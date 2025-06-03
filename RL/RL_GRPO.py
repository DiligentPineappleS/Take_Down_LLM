import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
from collections import defaultdict
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from pathlib import Path
import random
import re
from trl import GRPOTrainer

import numpy as np
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

def get_answer(text):
    """从响应中提取答案部分"""
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return match.group(1).strip() if match else ""

def reward_format(responses):
    """检查响应格式是否正确"""
    pattern = r"<thinking>.*?</thinking>\s*<answer>.*?</answer>"
    return [2.0 if re.search(pattern, res, re.DOTALL) else 0.0 for res in responses]

def reward_think_format(responses):
    """检查思考步骤格式"""
    pattern = r"<thinking>(.*?)</thinking>"
    return [1.0 if re.search(pattern, res, re.DOTALL) else 0.0 for res in responses]

def reward_correctness(responses, answers):
    """检查答案正确性"""
    extracted_answers = [get_answer(res) for res in responses]
    return [1.0 if ans.strip() == ref.strip() else 0.0 for ans, ref in zip(extracted_answers, answers)]



class GRPO(nn.Module):
    def __init__(self, model_name,beta,eps_min,eps_max,num_generations,max_length,batch_size,lr,train_data,eval_data,scale_rewards,gamma,device):
        super(GRPO,self).__init__()
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.ref_model  =  AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        for param in self.ref_model.parameters():
            param.requires_grad = False


        self.reward_model = ["think_format","format","correctness"]

        self.beta = beta
        self.scale_rewards = scale_rewards
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.num_generations = num_generations
        self.max_length =max_length
        self.batch_size = batch_size
        self.lr = lr
        self.train_data = train_data
        self.eval_data = eval_data
        self.gamma = gamma
        self.reward_weights = torch.ones(len(self.reward_model), dtype=torch.float32)
        self.optimizer = torch.optim.AdamW(self.model.parameters,lr=self.lr)

        self.metrics = defaultdict(list)
        
    def get_reward(self,response_texts,answer_texts):

        reward_outputs = torch.zeros(len(response_texts), len(self.reward_model), device=self.device)

        for i,reward_func in enumerate(self.reward_model):
            if "format" ==reward_func:
                format_reward = reward_format(response_texts)
                reward_outputs[:,i] =  torch.tensor([reward if reward is not None else torch.nan for reward in format_reward], dtype=torch.float32, device=self.device)
            elif "think_format" ==reward_func:
                think_format_reward = reward_think_format(response_texts)
                reward_outputs[:,i] =  torch.tensor([reward if reward is not None else torch.nan for reward in think_format_reward], dtype=torch.float32, device=self.device)
            elif "correctness" ==reward_func:
                correctness_reward = reward_correctness(response_texts,answer_texts)
                reward_outputs[:,i] =  torch.tensor([reward if reward is not None else torch.nan for reward in correctness_reward], dtype=torch.float32, device=self.device)
            else:
                continue

        # # 将奖励列表转换为张量
        # format_tensor = torch.tensor(format_reward, dtype=torch.float32, device=self.device)
        # think_tensor = torch.tensor(think_format_reward, dtype=torch.float32, device=self.device)
        # correct_tensor = torch.tensor(correctness_reward, dtype=torch.float32, device=self.device)
        # # 堆叠成 [batch_size, 3] 张量
        # torch.stack([format_tensor, think_tensor, correct_tensor], dim=1)

        # Apply weights to each reward function's output and sum
        rewards = (reward_outputs * self.reward_weights.to(self.device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(std_grouped_rewards, torch.zeros_like(std_grouped_rewards))

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)
        return advantages



        
    def generation_response(self, prompt,answer):
        # Tokenize prompt: [batch_size, seq_len] -> [batch_size, prompt_len]
        inputs = self.tokenizer(prompt,padding = True,truncation = True,max_length = self.max_length//2, return_tensor = "pt").to(self.device)
        output = self.model.generate(inputs.input_ids,attention_mask = inputs.attention_mask,max_length= self.max_length,do_sample = True,temperature = 0.7,top_p = 0.9,
                                     pad_token_id = self.tokenizer.pad_token_id,eos_token_id=self.tokenizer.eos_token_id,num_return_sequences = self.num_generations)
        responses, masks = [], []
        prompt_len = inputs.input_ids.shape[1]  # 输入序列长度
        # 处理每个生成序列 (batch_size * num_generations)

        for seq in output:
            # 找到第一个 eos_token 的位置（在生成部分）
            eos_pos = (seq==self.tokenizer.eos_pos_id).nonzero()
            cut_pos = eos_pos[0]+1 if len(eos_pos)>0  else len(seq)
            # 截取生成部分（不含输入）
            completion = seq[prompt_len:cut_pos]
            responses.append(completion)
            # 创建掩码（生成部分全为有效）
            mask = torch.ones(len(completion), dtype=torch.bool)
            masks.append(mask)
        
        completion_ids = nn.utils.rnn.pad_sequence(responses, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        prompt_ids = inputs.input_ids.repeat_interleave(self.num_generations, 0),
        prompt_mask = inputs.attention_mask.repeat_interleave(self.num_generations, 0),
        completion_ids = nn.utils.rnn.pad_sequence(responses, batch_first=True, padding_value=self.tokenizer.pad_token_id),
        completion_mask = nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)               
        
        input_ids = torch.cat([prompt_ids,completion_ids],dim=1)
        attention_mask =  torch.cat([prompt_mask,completion_mask],dim=1)
        completion_ids_list = [[id.item() for id, m in zip(row, mask_row) if m] for row, mask_row in zip(completion_ids, completion_mask)]
        logits_to_keep = completion_ids.size(1) 
        old_per_token_logps = self.get_per_token_logps(self.model, output, attention_mask, logits_to_keep)
        completions_text = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        repeated_answer = [sample for sample in answer for _ in range(self.num_generations)]
        reward_format(completions_text)
        advantages = self.get_reward(completions_text,repeated_answer)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
        }


        # 返回字典
        return {
            "prompt_ids": inputs.input_ids.repeat_interleave(self.num_generations, 0),
            "prompt_mask": inputs.attention_mask.repeat_interleave(self.num_generations, 0),
            "completion_ids": nn.utils.rnn.pad_sequence(responses, batch_first=True, padding_value=self.tokenizer.pad_token_id),
            "completion_mask": nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)
        }
    
    def process_advantages(self,batch_rewards):
        # 矩阵变换：
        rewards = batch_rewards.view(self.batch_size,self.num_generations,self.max_length)
        advantages = torch.zeros_like(rewards)
        future_rewards = torch.zeros(self.batch_size,self.num_generations)
        for i in range(self.max_length-1,-1,-1):
            future_rewards = rewards[:,:,i] + self.gamma
            advantages[:,:,i] = (future_rewards - future_rewards.mean(dim=1,keepdim = True))/ future_rewards.std(dim=1,keepdim = True)
        return advantages.view(self.batch_size*self.num_generations,-1)

    def responce_advanges(self,batch_rewards):
        rewards = batch_rewards.view(self.batch_size,self.num_generations)
        mean = rewards.mean(dim=1,keepdim=True)
        std = rewards.std(dim=1,keepdim=True)
        advantages = (rewards - mean) /std
        return advantages.view(self.batch_size*self.num_generations,1).expand(-1,self.max_length)

    def selective_log_softmax(logits, index):
        if logits.dtype in [torch.float32, torch.float64]:
            selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
            logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
            per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
        else:
            per_token_logps = []
            for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
                row_logps = F.log_softmax(row_logits, dim=-1)
                row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
                per_token_logps.append(row_per_token_logps)
            per_token_logps = torch.stack(per_token_logps)
        return per_token_logps    
    
    def get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        batch_size = self.batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]
            logits = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch, logits_to_keep=logits_to_keep + 1).logits
            logits = logits[:, :-1, :]  
            input_ids_batch = input_ids_batch[:, -logits_to_keep:]
            logits = logits[:, -logits_to_keep:]
            logits = logits / self.temperature
            logps = self.selective_log_softmax(logits, input_ids_batch)  # compute logprobs for the input tokens
            all_logps.append(logps)
        return torch.cat(all_logps, dim=0)
    
    def train(self,epochs):
        self.model.train()
        for epoch in tqdm(range(epochs),desc=f"train EPOCH {epoch}"):
            batch_data = random.sample(self.train_data, self.batch_size)
            batch_prompt = batch_data["prompt"]
            batch_answer = batch_data["answer"]
            batch_prompt = torch.tensor(batch_prompt).repeat_interleave(self.num_generations,dim=0).to(device=self.device)
            batch_prompt = torch.tensor(batch_prompt).repeat_interleave(self.num_generations,dim=0).to(device=self.device)

            with torch.no_grad():
                responces =  self.generation_responce(batch_prompt,batch_answer)
                completion_mask = responces["completion_ids"]
                input_ids = torch.cat([responces["prompt_ids"],responces["completion_ids"]],dim=1)
                attention_mask =  torch.cat([responces["prompt_mask"],responces["completion_mask"]],dim=1)
                logits_to_keep = responces["completion_ids"].size(1) 
                token_log_probs = self.get_per_token_logps(self.model, input_ids, attention_mask, logits_to_keep)
                ref_token_log_probs = self.get_per_token_logps(self.ref_model, input_ids, attention_mask, logits_to_keep)
                per_token_kl = (torch.exp(ref_token_log_probs - token_log_probs) - (ref_token_log_probs - token_log_probs) - 1)
                advantages = responces["advantages"]

                old_per_token_logps = (token_log_probs.detach() if responces["old_per_token_logps"] is None else responces["old_per_token_logps"])
                coef_1 = torch.exp(token_log_probs - old_per_token_logps)
                coef_2 = torch.clamp(coef_1, 1 - self.eps_min, 1 + self.eps_max)

                per_token_loss1 = coef_1 * advantages.unsqueeze(1)
                per_token_loss2 = coef_2 * advantages.unsqueeze(1)
                per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
                if self.beta != 0.0:
                    per_token_loss = per_token_loss + self.beta * per_token_kl
                loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
                self.optimizer.zero_grad()
                loss.backward() 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
                self.optimizer.step()
            return loss
                    

    def evaluate(self, eval_data=None, num_samples=None):
        """
        评估模型在验证集上的性能
        Args:
            eval_data: 可选，指定评估数据集（默认使用self.eval_data）
            num_samples: 可选，限制评估样本数量
        Returns:
            metrics: 包含各项评估指标的字典
        """ 
        # 设置模型为评估模式
        self.model.eval()
        
        # 确定评估数据集
        eval_data = eval_data or self.eval_data
        if num_samples and len(eval_data) > num_samples:
            eval_data = random.sample(eval_data, num_samples)
        
        # 初始化指标收集器
        metrics = {
            'total_reward': 0.0,
            'format_correct': 0,
            'think_format_correct': 0,
            'answer_correct': 0,
            'total_samples': 0
        }
        
        # 按批次处理评估数据
        with torch.no_grad():
            for i in range(0, len(eval_data), self.batch_size):
                batch = eval_data[i:i+self.batch_size]
                prompts = [item['prompt'] for item in batch]
                answers = [item['answer'] for item in batch]
                
                # 生成响应
                gen_results = self.generation_response(prompts, answers)
                responses = self.tokenizer.batch_decode(
                    gen_results['completion_ids'], 
                    skip_special_tokens=True
                )
                
                # 计算各项奖励
                format_rewards = reward_format(responses)
                think_rewards = reward_think_format(responses)
                correctness_rewards = reward_correctness(responses, answers)
                
                # 计算总奖励（使用与训练相同的权重）
                rewards = torch.tensor([
                    format_rewards[i] * self.reward_weights[0].item() +
                    think_rewards[i] * self.reward_weights[1].item() +
                    correctness_rewards[i] * self.reward_weights[2].item()
                    for i in range(len(responses))
                ])
                
                # 更新指标
                batch_size = len(batch)
                metrics['total_reward'] += rewards.sum().item()
                metrics['format_correct'] += sum(1 for r in format_rewards if r > 0)
                metrics['think_format_correct'] += sum(1 for r in think_rewards if r > 0)
                metrics['answer_correct'] += sum(1 for r in correctness_rewards if r > 0)
                metrics['total_samples'] += batch_size
        
        # 计算平均指标
        if metrics['total_samples'] > 0:
            metrics['avg_reward'] = metrics['total_reward'] / metrics['total_samples']
            metrics['format_accuracy'] = metrics['format_correct'] / metrics['total_samples']
            metrics['think_accuracy'] = metrics['think_format_correct'] / metrics['total_samples']
            metrics['answer_accuracy'] = metrics['answer_correct'] / metrics['total_samples']
        
        # 恢复模型为训练模式
        self.model.train()
        
        return metrics




    def evaluate_model(self, eval_data=None, num_samples=None, compute_ppl=True):
        """
        全面评估模型性能，包括：
        - 奖励指标（格式、思考步骤、正确性）
        - 语言质量指标（困惑度、BLEU、METEOR）
        - 内容质量指标（多样性、重复率）
        - 答案准确度
        
        Args:
            eval_data: 评估数据集（默认使用self.eval_data）
            num_samples: 限制评估样本数量
            compute_ppl: 是否计算困惑度（计算成本较高）
        
        Returns:
            metrics: 包含各项评估指标的字典
        """
        self.model.eval()
        eval_data = eval_data or self.eval_data
        if num_samples and len(eval_data) > num_samples:
            eval_data = random.sample(eval_data, num_samples)
        
        # 初始化指标收集器
        metrics = {
            'total_reward': 0.0,
            'format_accuracy': 0.0,
            'think_accuracy': 0.0,
            'answer_accuracy': 0.0,
            'perplexity': 0.0,
            'bleu_score': 0.0,
            'meteor_score': 0.0,
            'distinct_1': 0.0,
            'distinct_2': 0.0,
            'repetition_rate': 0.0,
            'total_samples': 0
        }
        
        # 用于困惑度计算的临时存储
        all_losses = []
        smooth = SmoothingFunction().method1
        
        with torch.no_grad():
            for i in tqdm(range(0, len(eval_data), self.batch_size), 
                        desc="Evaluating", total=len(eval_data)//self.batch_size+1):
                batch = eval_data[i:i+self.batch_size]
                prompts = [item['prompt'] for item in batch]
                answers = [item['answer'] for item in batch]
                
                # 生成响应
                gen_results = self.generation_response(prompts, answers)
                responses = self.tokenizer.batch_decode(
                    gen_results['completion_ids'], 
                    skip_special_tokens=True
                )
                
                # 提取纯答案部分
                gen_answers = [get_answer(res) for res in responses]
                
                # 1. 计算基础奖励指标
                format_rewards = reward_format(responses)
                think_rewards = reward_think_format(responses)
                correctness_rewards = reward_correctness(responses, answers)
                
                # 2. 计算困惑度（语言流畅性）
                if compute_ppl:
                    batch_ppl = self.calculate_perplexity(
                        gen_results['input_ids'],
                        gen_results['attention_mask'],
                        gen_results['prompt_ids'].shape[1]  # prompt长度
                    )
                    metrics['perplexity'] += batch_ppl.sum().item()
                
                # 3. 计算BLEU和METEOR（内容相关性）
                batch_bleu = 0.0
                batch_meteor = 0.0
                
                for gen_ans, ref_ans in zip(gen_answers, answers * self.num_generations):
                    gen_tokens = word_tokenize(gen_ans.lower())
                    ref_tokens = word_tokenize(ref_ans.lower())
                    
                    if gen_tokens and ref_tokens:
                        # BLEU计算
                        batch_bleu += sentence_bleu(
                            [ref_tokens], 
                            gen_tokens, 
                            smoothing_function=smooth
                        )
                        
                        # METEOR计算
                        batch_meteor += meteor_score(
                            [ref_tokens], 
                            gen_tokens
                        )
                
                metrics['bleu_score'] += batch_bleu
                metrics['meteor_score'] += batch_meteor
                
                # 4. 计算多样性和重复率
                all_tokens = []
                repeated_count = 0
                for res in responses:
                    tokens = word_tokenize(res.lower())
                    all_tokens.extend(tokens)
                    
                    # 检查连续重复
                    for j in range(1, len(tokens)):
                        if tokens[j] == tokens[j-1]:
                            repeated_count += 1
                
                # 5. 更新基础指标
                batch_size = len(batch) * self.num_generations
                metrics['total_reward'] += sum(
                    format_rewards[i] * self.reward_weights[0].item() +
                    think_rewards[i] * self.reward_weights[1].item() +
                    correctness_rewards[i] * self.reward_weights[2].item()
                    for i in range(len(responses))
                )
                metrics['format_accuracy'] += sum(1 for r in format_rewards if r > 0)
                metrics['think_accuracy'] += sum(1 for r in think_rewards if r > 0)
                metrics['answer_accuracy'] += sum(1 for r in correctness_rewards if r > 0)
                metrics['repetition_rate'] += repeated_count / max(1, len(all_tokens))
                metrics['total_samples'] += batch_size
                
                # 6. 更新多样性指标（基于整个评估集）
                if all_tokens:
                    token_counts = Counter(all_tokens)
                    metrics['distinct_1'] = len(token_counts) / max(1, len(all_tokens))
                    
                    # 计算bigram多样性
                    bigrams = [tuple(all_tokens[i:i+2]) for i in range(len(all_tokens)-1)]
                    bigram_counts = Counter(bigrams)
                    metrics['distinct_2'] = len(bigram_counts) / max(1, len(bigrams))
        
        # 计算平均指标
        if metrics['total_samples'] > 0:
            metrics['avg_reward'] = metrics['total_reward'] / metrics['total_samples']
            metrics['format_accuracy'] /= metrics['total_samples']
            metrics['think_accuracy'] /= metrics['total_samples']
            metrics['answer_accuracy'] /= metrics['total_samples']
            metrics['repetition_rate'] /= (len(eval_data) // self.batch_size + 1)
            
            if compute_ppl:
                metrics['perplexity'] = torch.exp(torch.tensor(
                    metrics['perplexity'] / metrics['total_samples']
                )).item()
            
            metrics['bleu_score'] /= metrics['total_samples']
            metrics['meteor_score'] /= metrics['total_samples']
        
        self.model.train()
        return metrics

    def calculate_perplexity(self, input_ids, attention_mask, prompt_length):
        """
        计算生成文本的困惑度
        Args:
            input_ids: 完整输入序列 (prompt + completion)
            attention_mask: 注意力掩码
            prompt_length: prompt部分的长度
        
        Returns:
            ppl: 每个序列的困惑度
        """
        # 准备标签 - 右移一位
        labels = input_ids.clone()
        labels[:, :prompt_length] = -100  # 忽略prompt部分
        
        # 前向传播
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # 计算每个序列的困惑度
        seq_loss = torch.zeros(input_ids.size(0), device=self.device)
        for i in range(input_ids.size(0)):
            # 只考虑生成部分的损失
            valid_mask = (labels[i] != -100) & (attention_mask[i] == 1)
            valid_loss = loss[i] * valid_mask.float()
            seq_loss[i] = valid_loss.sum() / valid_mask.sum().clamp(min=1)
        
        return torch.exp(seq_loss)


import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
import re
from tqdm import tqdm

class MathReasoningDataset(Dataset):
    """数学推理数据集，包含问题和参考答案"""
    def __init__(self, data_path, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(data_path)
        
    def load_data(self, data_path):
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        processed = []
        for item in raw_data:
            # 构建带格式的提示
            processed.append({
                'prompt': item["prompt"],
                'answer': item['answer']
            })
        return processed
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class TrainingConfig:
    def __init__(self):
        self.model_name = "gpt2"  # 基础模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.beta = 0.1  # KL散度系数
        self.eps_min = 0.1  # PPO clip范围下限
        self.eps_max = 0.1  # PPO clip范围上限
        self.num_generations = 4  # 每个提示生成的响应数量
        self.max_length = 256  # 最大生成长度
        self.batch_size = 4  # 训练批次大小
        self.lr = 1e-5  # 学习率
        self.gamma = 0.99  # 奖励折扣因子
        self.scale_rewards = True  # 是否缩放奖励
        self.epochs = 20  # 训练轮数
        self.eval_interval = 2  # 评估间隔（轮数）
        self.save_dir = Path("checkpoints")  # 模型保存目录
        self.save_dir.mkdir(exist_ok=True)

def train():
    # 加载配置
    config = TrainingConfig()
    
    # 加载数据集
    train_dataset = MathReasoningDataset("data/train_math.json", tokenizer=None)
    eval_dataset = MathReasoningDataset("data/eval_math.json", tokenizer=None)
    
    # 初始化模型
    model = GRPO(
        model_name=config.model_name,
        beta=config.beta,
        eps_min=config.eps_min,
        eps_max=config.eps_max,
        num_generations=config.num_generations,
        max_length=config.max_length,
        batch_size=config.batch_size,
        lr=config.lr,
        train_data=train_dataset.data,
        eval_data=eval_dataset.data,
        scale_rewards=config.scale_rewards,
        gamma=config.gamma,
        device=config.device
    )
    
    # 训练循环
    best_score = -float('inf')
    for epoch in range(config.epochs):
        print(f"\n{'='*40} Epoch {epoch+1}/{config.epochs} {'='*40}")
        
        # 训练阶段
        model.train()
        total_loss = 0
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size,
            shuffle=True
        )
        
        for batch in tqdm(train_loader, desc="Training"):
            prompts = batch['prompt']
            answers = batch['answer']
            
            # 生成响应并计算损失
            loss = model.train(prompts, answers)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Training Loss: {avg_loss:.4f}")
        
        # 评估阶段
        if (epoch + 1) % config.eval_interval == 0 or epoch == config.epochs - 1:
            eval_metrics = model.evaluate_model(num_samples=50)
            
            # 打印评估结果
            print("\nEvaluation Metrics:")
            print(f"  Avg Reward: {eval_metrics['avg_reward']:.4f}")
            print(f"  Format Accuracy: {eval_metrics['format_accuracy']:.2%}")
            print(f"  Think Accuracy: {eval_metrics['think_accuracy']:.2%}")
            print(f"  Answer Accuracy: {eval_metrics['answer_accuracy']:.2%}")
            print(f"  Perplexity: {eval_metrics['perplexity']:.2f}")
            print(f"  BLEU Score: {eval_metrics['bleu_score']:.4f}")
            print(f"  Distinct-2: {eval_metrics['distinct_2']:.4f}")
            
            # 保存最佳模型
            score = eval_metrics['avg_reward'] + eval_metrics['answer_accuracy']
            if score > best_score:
                best_score = score
                model.save_model(config.save_dir / f"best_model_epoch{epoch+1}.pt")
                print(f"Saved best model with score {score:.4f}")
                
            # 保存检查点
            model.save_model(config.save_dir / f"checkpoint_epoch{epoch+1}.pt")
    
    print("Training completed!")


# 6. 运行训练
if __name__ == "__main__":

    """
    prompt:"";
    answer : "
        <thinking>
        推理过程...
        </thinking>
        <answer>
        最终答案
        </answer>
        "
    """
    train()