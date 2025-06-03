import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from transformers import AutoModelForCausalLM,AutoTokenizer
from tqdm import tqdm
from collections import defaultdict
class DPODataset(Dataset):
    def __init__(self,data,tokenizer,max_lenght=512):
        """
        数据处理，生成输入格式
        Attribute:
            prompts:提示词
            chosen:优质答案
            rejected:劣质答案
        数据格式要求:
            每条样本都应当包含:prompt\chosen\rejected
        """
        super(DPODataset,self).__init__()
        self.tokenizer = tokenizer
        self.max_lenght = max_lenght
        self.prompts = [f"[INST] {d['prompt']} [/INST]" for d in data]
        self.chosen = [d['chosen'] for d in data]
        self.rejected = [d['rejected'] for d in data]
        self.len = len(self.prompts)
    def tokenizer_process(self,prompt,responce):
        """
        拼接提示词和回答，并进行分词处理
        输入：
            prompt:提示词
            responce:对应的回答
        输出：
            input_ids:分词后的tokenid序列
            attention_mask:掩码序列
        """
        diag = prompt + responce
        return self.tokenizer(diag,max_length = self.max_lenght,truncation = True,padding = "max_length",return_tensors="pt")
    def get_tokenizer(self,ids):
        """
        生成单个训练样本 : 
        输入：索引ID
        输出：
            chosen_ids: 优选答案序列的token id
            rejected_ids: 劣制答案序列的token id
            attention_mask: 对应优质答案序列的掩码
        """
        chosen_tokenizer = self.tokenizer_process(self.prompts[ids],self.chosen[ids])
        rejected_tokenizer = self.tokenizer_process(self.prompts[ids],self.rejected[ids]) 
        return {
            "chosen_ids":chosen_tokenizer["input_ids"],
            "rejected_ids":rejected_tokenizer["input_ids"],
            "attention_mask":chosen_tokenizer["attention_mask"]
        }

class DPO(nn.Module):
    """
    DPO实现:
    DPO 通过重新参数化将复杂的强化学习问题转化为简单的分类任务，直接在策略模型和参考模型之间优化偏好概率;
    通过对比策略模型和参考模型的输出差异，隐式学习奖励信号;
    参考模型作为基准,通过KL散度约束(beta控制强度)防止策略模型过度优化到偏好数据的噪声中
    beta越小,保持回答多样性,但偏好对齐效果减弱。beta越大,对齐效果越好,多样性会差
    核心依赖公式：
        L_DPO = -E[log Sigmoid(beta(log(policy_chosen/policy_rejected) - log(ref_chosen/ref_rejected)))]
        其中: log(policy_chosen/policy_rejected):策略模型偏好程度
            log(ref_chosen/ref_rejected):参考模型偏好程度
    """

    def __init__(self,model,ref_model,tokenizer,train_data,eval_data,beta,lr,batch_size,max_length,eval_interval):
        """
        DPO初始化
        输入：
            model：待训练的策略模型
            ref_model： 参考模型
            tokenizer：分词器
            train_data：训练数据
            beta：DPO温度系数，控制KL散度
            lr：学习率
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.ref_model = ref_model.to(self.device)
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.eval_data = eval_data
        self.beta = beta
        self.lr = lr
        self.batch_size = batch_size
        self.max_length = max_length
         # 冻结参考模型参数
        for param in self.ref_model.parameters():
            param.requires_grad = False
        # 定义优化器
        self.optimizer = torch.optim.AdamW(self.model.parameters,lr=self.lr)
        # DPO数据加载
        self.data_set = DPODataset(self.train_data,self.tokenizer,self.max_length)
        self.data_loader = DataLoader(self.data_set,batch_size=self.batch_size,shuffle=True)
        self.eval_interval = eval_interval  # 评估间隔
        self.best_eval_loss = float('inf')
        
    def log_probs(self,model,input_ids,attention_mask):
        """
        计算给定模型对序列的对数概率
        输入：
            model: 待评估的模型
            input_ids: 输入序列tokenid
            attention_mask: 注意力掩码
        输出: 
            序列对数概率
        
        计算步骤分解：
            1. 模型前向得到logits（未归一化的预测值）
            2. 计算每个位置的log_softmax（归一化对数概率）
            3. 通过gather获取实际token的对数概率
            4. 应用mask并求和得到序列总对数概率
        
        数学公式：
            log P(y|x) = Σ_{t=1}^L log P(y_t | x, y_{<t}) * mask_t
        """
        with torch.no_grad():
            outputs = model(input_ids,attention_mask = attention_mask)
            logits = outputs.logits()
        # 计算对数概率：log_softmax沿词汇表维度
        logits_probs = torch.log_softmax(logits,dim=-1)
        # 获取实际token的对数概率
        ## 取出第一个词后的所有tokenid，并增加一个维度
        ## 模型预测的第 i 个位置对应输入序列的第 i+1 个 token，概率对应第i个位置；所以token位置提取第一个位置之后的，预测概率对应的到第i个位置
        ## (batch, seq_len-1, 1)
        label = input_ids[:,1:].unsqueeze(-1)
        ## 取出截止最后一个词的预测概率；
        logits_select = logits_probs[:,-1:,:]
        # 合并 (batch, seq_len-1)
        token_logits_select = torch.gather(logits_select,2,label).squeeze(-1)
        mask = attention_mask[:,1:].float()
        sequence_log_prob = (token_logits_select * mask).sum(dim=1)
        return sequence_log_prob
    
    def dpo_loss(self,policy_chosen_probs,policy_rejected_probs,ref_chosen_probs,ref_rejected_probs):
        policy_diff_probs = policy_chosen_probs - policy_rejected_probs
        ref_diff_probs = ref_chosen_probs - ref_rejected_probs
        logits = policy_diff_probs - ref_diff_probs
        loss = -F.logsigmoid(self.beta , logits).mean()
        return loss
    def train(self,epoch):
        self.model.train()
        total_loss = 0 
        for batch_idx,batch_data in enumerate (tqdm(self,DataLoader,desc = f"Epoch{epoch}")):
            
            chosen_ids = batch_data["chosen_ids"]
            rejected_ids = batch_data["rejected_ids"]
            attention_mask = batch_data["attention_mask"]
            policy_chosen_probs = self.log_probs(self.model,chosen_ids,attention_mask)
            policy_rejected_probs = self.log_probs(self.model,rejected_ids,attention_mask)
            with torch.no_grad():
                ref_chosen_probs = self.log_probs(self.ref_model,chosen_ids,attention_mask)
                ref_rejected_probs = self.log_probs(self.ref_model,chosen_ids,attention_mask)
            loss = self.dpo_loss(policy_chosen_probs,policy_rejected_probs,ref_chosen_probs,ref_rejected_probs)
            total_loss = total_loss + loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            #为了防止梯度爆炸，采用限制梯度L2范数的方法，max_norm最大允许的梯度范数
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=1.0)
            self.optimizer.step()
        # 每个eopch评估
        eval_metrics = self.evaluate(self.eval_data)
        print(f"Step {batch_idx+1} Evaluation:")
        for k, v in eval_metrics.items():
            print(f"{k}: {v:.4f}")
        # 保存最佳模型
        if eval_metrics["eval_loss"] < self.best_eval_loss:
            self.save_model()
            self.best_eval_loss = eval_metrics["eval_loss"]
        return total_loss/len(self,DataLoader)

    def evaluate(self, eval_data):
        """
        综合评估模型性能
        返回指标：
            - 验证损失
            - 偏好胜率（Preference Win Rate）
            - 生成质量（困惑度）
            - 生成多样性（distinct-2）
        """
        self.model.eval()
        total_loss = 0
        win_count = 0
        total_samples = 0
        all_generated = []
        
        with torch.no_grad():
            for batch in tqdm(eval_data, desc="Evaluating"):
                # 计算验证损失
                chosen_ids = batch["chosen_ids"].to(self.device)
                rejected_ids = batch["rejected_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                policy_chosen = self.log_probs(self.model, chosen_ids, attention_mask)
                policy_rejected = self.log_probs(self.model, rejected_ids, attention_mask)
                ref_chosen = self.log_probs(self.ref_model, chosen_ids, attention_mask)
                ref_rejected = self.log_probs(self.ref_model, rejected_ids, attention_mask)
                
                loss = self.dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
                total_loss += loss.item()
                
                # 计算偏好胜率
                policy_ratio = (policy_chosen - policy_rejected)
                ref_ratio = (ref_chosen - ref_rejected)
                win_count += (policy_ratio > ref_ratio).sum().item()
                total_samples += len(policy_ratio)
                
                # 生成文本用于质量评估
                generated = self.generate_text(batch["prompt_ids"])
                all_generated.extend(generated)
        
        # 计算指标
        eval_loss = total_loss / len(eval_data)
        win_rate = win_count / total_samples
        perplexity = self.calculate_perplexity(all_generated)
        diversity = self.calculate_diversity(all_generated)
        
        return {
            "eval_loss": eval_loss,
            "win_rate": win_rate,
            "perplexity": perplexity,
            "diversity": diversity
        }
    
    def generate_text(self, prompt_ids, max_length=50):
        """生成文本用于质量评估"""
        self.model.eval()
        generated = []
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=prompt_ids.to(self.device),
                max_length=max_length,
                num_beams=1,
                do_sample=True,
                top_p=0.9
            )
            for seq in outputs:
                text = self.tokenizer.decode(seq, skip_special_tokens=True)
                generated.append(text)
        return generated
    
    def calculate_perplexity(self, texts):
        """计算困惑度"""
        total_logprob = 0
        total_tokens = 0
        
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                log_probs = torch.log_softmax(outputs.logits, dim=-1)
                token_probs = log_probs[0, :-1].gather(1, inputs.input_ids[0, 1:].unsqueeze(-1)).squeeze()
                total_logprob += token_probs.sum().item()
                total_tokens += len(token_probs)
        
        return torch.exp(torch.tensor(-total_logprob / total_tokens)).item()
    
    def calculate_diversity(self, texts, n=2):
        """计算distinct-n多样性指标"""
        ngram_counts = defaultdict(int)
        total_ngrams = 0
        
        for text in texts:
            tokens = text.split()
            for i in range(len(tokens)-n+1):
                ngram = tuple(tokens[i:i+n])
                ngram_counts[ngram] += 1
                total_ngrams += 1
        
        return len(ngram_counts) / total_ngrams if total_ngrams > 0 else 0

    def save_model(self, path="best_model"):
        """保存模型和分词器"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        

if __name__ == "__main__":
    # 初始化组件
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
    tokenizer.pad_token = tokenizer.eos_token  # 设置pad token
    
    # 加载模型=
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 示例训练数据
    train_data = [
        {
            "prompt": "如何学习人工智能？",
            "chosen": "学习人工智能需要掌握数学基础、编程技能",
            "rejected": "随便看看视频就能学会，不需要动手实践。"
        }
    ]
    
    # 初始化训练器
    dpo_trainer = DPO(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_data=train_data,
        beta=0.1,
        lr=5e-6,
        batch_size=2
    )
    
    # 执行训练
    for epoch in range(3):
        dpo_trainer.train(epoch)
    
    # 保存模型
    model.save_pretrained("./dpo_finetuned_model")


        



