
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from tqdm import tqdm
import numpy as np
import os
import gc
import json
from datetime import datetime

# 设置随机种子保证可复现
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cuda.matmul.allow_tf32 = True  # 启用TF32加速


class PPOTrainer:
    def __init__(self, 
                 policy_model, 
                 ref_model, 
                 reward_model,
                 tokenizer,
                 optimizer,
                 config):
        """
        PPO训练器初始化
        
        参数:
        policy_model: 待优化的策略模型
        ref_model: 参考模型（冻结）
        reward_model: 人类偏好奖励模型
        tokenizer: 分词器
        optimizer: 优化器
        config: 训练配置
        """
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.config = config
        
        # 确保模型在正确的设备上
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_model.to(self.device)
        self.ref_model.to(self.device)
        self.reward_model.to(self.device)
        
        # 冻结参考模型和奖励模型
        self._freeze_model(self.ref_model)
        self._freeze_model(self.reward_model)
        
        # 创建经验缓冲区
        self.buffer = {
            "query_ids": [],
            "response_ids": [],
            "logprobs": [],
            "values": [],
            "rewards": [],
            "masks": []
        }

        # 初始化KL控制器参数
        self.kl_coef = config.get("init_kl_coef", 0.1)
        self.target_kl = config.get("target_kl", 6.0)
        self.adaptive_kl = config.get("adaptive_kl", True)

    def _freeze_model(self, model):
        """冻结模型参数"""
        for param in model.parameters():
            param.requires_grad = False

    def generate_response(self, query_ids, log_prob_callback=True):
        """
        使用策略模型生成响应
        
        参数:
        query_ids: 输入查询的token ids
        log_prob_callback: 是否计算对数概率
        
        返回:
        response_ids: 生成的响应token ids
        response_logprobs: 每一步的对数概率
        response_values: 每一步的状态价值估计
        """
        self.policy_model.eval()
        
        # 创建注意力掩码
        attention_mask = query_ids.ne(self.tokenizer.pad_token_id)
        
        # 生成参数配置
        generation_config = {
            "input_ids": query_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": self.config["max_new_tokens"],
            "do_sample": True,
            "top_p": self.config["top_p"],
            "temperature": self.config["temperature"],
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": log_prob_callback,
            "output_attentions": False,
            "output_hidden_states": False
        }
        
        # 生成响应
        with torch.no_grad():
            outputs = self.policy_model.generate(**generation_config)
            response_ids = outputs.sequences
            
            # 初始化响应部分
            response_logprobs = []
            response_values = []
            
            # 收集对数概率和价值估计
            if log_prob_callback:
                # 获取每一步的logits
                logits = torch.stack(outputs.scores, dim=1)
                
                # 提取生成token的对数概率
                generated_tokens = response_ids[:, query_ids.size(1):]
                probs = F.softmax(logits, dim=-1)
                response_logprobs = torch.log(probs.gather(2, generated_tokens.unsqueeze(2))).squeeze(2)
                
                # 计算价值估计
                value_inputs = torch.cat([query_ids, generated_tokens], dim=1)
                value_mask = value_inputs.ne(self.tokenizer.pad_token_id)
                value_outputs = self.policy_model(value_inputs, attention_mask=value_mask)
                response_values = value_outputs.value[:, query_ids.size(1)-1:-1]
                
                # 处理价值输出维度
                if response_values.dim() > 2:
                    response_values = response_values.squeeze(2)
        
        # 提取响应部分 (移除查询)
        response_only_ids = response_ids[:, query_ids.size(1):]
        
        # 创建响应掩码 (区分实际内容和填充)
        response_mask = response_only_ids.ne(self.tokenizer.pad_token_id)
        
        return response_only_ids, response_logprobs, response_values, response_mask

    def compute_human_preference_reward(self, query_ids, response_ids):
        """
        计算基于人类偏好的奖励
        
        参数:
        query_ids: 查询的token ids
        response_ids: 响应的token ids
        
        返回:
        rewards: 奖励值张量 [batch_size]
        """
        self.reward_model.eval()
        
        # 拼接查询和响应
        full_input_ids = torch.cat([query_ids, response_ids], dim=1)
        attention_mask = full_input_ids.ne(self.tokenizer.pad_token_id)
        
        # 解码为文本用于奖励模型
        texts = []
        for i in range(full_input_ids.size(0)):
            text = self.tokenizer.decode(
                full_input_ids[i][attention_mask[i].bool()], 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            texts.append(text)
        
        # 计算奖励 - 使用奖励模型评估人类偏好
        with torch.no_grad():
            rewards = []
            for text in texts:
                # 实际应用中替换为您的奖励模型逻辑
                # 以下是模拟的奖励组件：
                coherence = self._text_coherence(text)  # 文本连贯性
                relevance = self._query_relevance(text, self.tokenizer.decode(query_ids[0], skip_special_tokens=True))  # 相关性
                engagement = self._engagement_score(text)  # 吸引力
                
                # 组合权重 (可根据需要调整)
                total_reward = (
                    0.4 * coherence + 
                    0.4 * relevance + 
                    0.2 * engagement
                )
                
                # 加入奖励模型输出
                # 实际应用中取消下一行注释
                # total_reward += self.reward_model(text).item()
                
                rewards.append(total_reward)
            
            # 添加长度惩罚
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            
            # 长度惩罚系数 (防止过短响应)
            token_counts = (response_ids != self.tokenizer.pad_token_id).sum(dim=1).float()
            length_penalty = torch.clamp(token_counts / self.config["min_length_penalty"], max=1.0)
            rewards *= length_penalty
            
        return rewards

    def _text_coherence(self, text):
        """模拟文本连贯性评分"""
        # 实际应用中用NLP模型替代
        if len(text) < 20:
            return 0.3
        if "however" in text or "therefore" in text or "furthermore" in text:
            return 0.9
        return 0.7

    def _query_relevance(self, text, query):
        """模拟查询相关性评分"""
        # 实际应用中用相似度模型替代
        if len(query) > 0 and any(word in text for word in query.split()[:3]):
            return 0.8
        return 0.6

    def _engagement_score(self, text):
        """模拟用户参与度评分"""
        # 实际应用中用情感模型替代
        if "!" in text or "?" in text or ("amazing" in text or "important" in text):
            return 0.9
        return 0.65

    def compute_advantages(self, rewards, values, masks):
        """
        计算广义优势估计
        
        参数:
        rewards: 每步奖励 [batch_size]
        values: 价值估计 [batch_size, seq_len]
        masks: 响应掩码 [batch_size, seq_len]
        
        返回:
        advantages: 优势估计 [batch_size, seq_len]
        returns: 回报 [batch_size, seq_len]
        """
        # 确保输入为张量
        rewards = rewards.unsqueeze(1).repeat(1, values.size(1)) * masks.float()
        masks = masks.float()
        gamma = self.config["gamma"]
        lam = self.config["lambda"]
        
        # 准备输出
        advantages = torch.zeros_like(values)
        returns = torch.zeros_like(values)
        last_gaelam = 0
        
        # 时间步计算 (从后向前)
        for t in reversed(range(values.size(1))):
            next_value = values[:, t+1] if t < values.size(1)-1 else 0
            delta = rewards[:, t] + gamma * next_value * masks[:, t] - values[:, t]
            advantages[:, t] = last_gaelam = delta + gamma * lam * masks[:, t] * last_gaelam
            
            # 计算回报
            returns[:, t] = advantages[:, t] + values[:, t]
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns

    def kl_penalty(self, query_ids, response_ids, logprobs):
        """
        计算策略模型和参考模型之间的KL散度惩罚
        
        参数:
        query_ids: 查询的token ids
        response_ids: 响应的token ids
        logprobs: 策略模型生成的对数概率
        
        返回:
        kl_div: KL散度值 [batch_size]
        """
        self.ref_model.eval()
        self.policy_model.eval()
        
        # 拼接查询和响应
        input_ids = torch.cat([query_ids, response_ids], dim=1)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        # 计算参考模型的对数概率
        with torch.no_grad():
            outputs = self.ref_model(input_ids, attention_mask=attention_mask)
            ref_logits = outputs.logits
            
            # 只考虑响应部分 (偏移一个token)
            ref_logits = ref_logits[:, query_ids.size(1)-1:-1, :]
            ref_logprobs = F.log_softmax(ref_logits, dim=-1)
            
            # 提取参考模型生成token的对数概率
            ref_logprobs = torch.gather(
                ref_logprobs, 
                2, 
                response_ids.unsqueeze(2)
            ).squeeze(2)
            
            # 计算KL散度: KL(Policy || Ref)
            kl_div = logprobs - ref_logprobs  # log(policy) - log(ref)
            kl_div = torch.sum(kl_div, dim=1)  # 按序列求和
            
        return kl_div.mean()  # 返回批次平均值

    def ppo_loss(self, old_logprobs, new_logprobs, advantages, kl_div, clip_eps=0.2):
        """
        计算PPO损失
        
        参数:
        old_logprobs: 生成时的旧对数概率
        new_logprobs: 当前策略的新对数概率
        advantages: 优势估计
        kl_div: KL散度值
        clip_eps: PPO截断阈值
        
        返回:
        policy_loss: 策略损失
        value_loss: 价值损失
        loss: 总损失
        """
        # 概率比
        ratio = torch.exp(new_logprobs - old_logprobs)
        
        # 策略损失 (截断)
        pg_losses1 = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        policy_loss = torch.max(pg_losses1, pg_losses2).mean()
        
        # KL惩罚损失
        kl_loss = self.kl_coef * kl_div
        
        # 总损失
        total_loss = policy_loss + kl_loss
        
        # 添加熵奖励 (可选)
        if self.config.get("entropy_coef", 0) > 0:
            entropy = -torch.exp(new_logprobs) * new_logprobs
            entropy_loss = -self.config["entropy_coef"] * entropy.mean()
            total_loss += entropy_loss
        
        return {
            "policy_loss": policy_loss,
            "kl_loss": kl_loss,
            "total_loss": total_loss,
            "kl_div": kl_div.item()
        }

    def update_kl_coef(self, kl_div):
        """根据KL散度动态调整KL惩罚系数"""
        if not self.adaptive_kl:
            return
        
        # 计算调整
        kl_diff = kl_div - self.target_kl
        adjustment = self.config.get("kl_adjust_rate", 0.01) * kl_diff
        
        # 更新KL系数
        self.kl_coef = max(0.0, self.kl_coef + adjustment)
        
        # 记录调整
        print(f"KL divergence: {kl_div:.4f}, KL coef updated to {self.kl_coef:.4f}")
        return self.kl_coef

    def train_step(self, batch):
        """
        执行单次训练步骤
        
        参数:
        batch: 输入批次数据
        
        返回:
        loss_dict: 包含各项损失的字典
        """
        self.policy_model.train()
        query_ids = batch["input_ids"].to(self.device)
        query_mask = batch["attention_mask"].to(self.device)
        
        # ================== 轨迹收集 ================== 
        # 生成响应
        response_ids, old_logprobs, old_values, response_mask = self.generate_response(
            query_ids, log_prob_callback=True
        )
        
        # 计算奖励
        rewards = self.compute_human_preference_reward(query_ids, response_ids)
        
        # 计算优势和回报
        advantages, returns = self.compute_advantages(
            rewards, 
            old_values, 
            response_mask
        )
        
        # 记录轨迹
        self.buffer["query_ids"].append(query_ids)
        self.buffer["response_ids"].append(response_ids)
        self.buffer["logprobs"].append(old_logprobs)
        self.buffer["values"].append(old_values)
        self.buffer["rewards"].append(rewards)
        self.buffer["masks"].append(response_mask)
        
        # ================== 策略优化 ==================
        loss_dict = None
        
        # 当缓冲区达到批次大小时更新策略
        if len(self.buffer["query_ids"]) >= self.config["buffer_size"]:
            # 计算KL散度
            kl_div = self.kl_penalty(
                torch.cat(self.buffer["query_ids"], dim=0),
                torch.cat(self.buffer["response_ids"], dim=0),
                torch.cat(self.buffer["logprobs"], dim=0)
            )
            
            # 动态调整KL系数
            self.update_kl_coef(kl_div.item())
            
            # 执行多次PPO更新
            for _ in range(self.config["ppo_epochs"]):
                # 获取当前策略在新数据上的对数概率
                new_logprobs, values = self._current_policy_logprobs(
                    torch.cat(self.buffer["query_ids"], dim=0),
                    torch.cat(self.buffer["response_ids"], dim=0),
                    torch.cat(self.buffer["masks"], dim=0)
                )
                
                # 计算损失
                advantages_flat = torch.cat([
                    a.view(-1)[m.view(-1) > 0] 
                    for a, m in zip(self.buffer["advantages"], self.buffer["masks"])
                ])
                
                old_logprobs_flat = torch.cat([
                    lp.view(-1)[m.view(-1) > 0] 
                    for lp, m in zip(self.buffer["logprobs"], self.buffer["masks"])
                ])
                
                new_logprobs_flat = torch.cat([
                    lp.view(-1)[m.view(-1) > 0] 
                    for lp, m in zip(new_logprobs, self.buffer["masks"])
                ])
                
                # PPO损失
                loss_dict = self.ppo_loss(
                    old_logprobs_flat,
                    new_logprobs_flat,
                    advantages_flat,
                    kl_div
                )
                
                # 价值损失 (可选)
                if self.config.get("value_coef", 0) > 0:
                    returns_flat = torch.cat([
                        r.view(-1)[m.view(-1) > 0] 
                        for r, m in zip(self.buffer["returns"], self.buffer["masks"])
                    ])
                    
                    values_flat = torch.cat([
                        v.view(-1)[m.view(-1) > 0] 
                        for v, m in zip(values, self.buffer["masks"])
                    ])
                    
                    value_loss = F.mse_loss(values_flat, returns_flat)
                    loss_dict["total_loss"] += self.config["value_coef"] * value_loss
                    loss_dict["value_loss"] = value_loss.item()
                
                # 反向传播
                self.optimizer.zero_grad()
                loss_dict["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy_model.parameters(), 
                    self.config["max_grad_norm"]
                )
                self.optimizer.step()
            
            # 清空缓冲区
            self._clear_buffer()
        
        return loss_dict or {"total_loss": 0.0}

    def _current_policy_logprobs(self, query_ids, response_ids, masks):
        """
        获取当前策略在响应上的对数概率
        
        参数:
        query_ids: 查询token ids
        response_ids: 响应token ids
        masks: 响应掩码
        
        返回:
        logprobs: 对数概率
        values: 价值估计
        """
        self.policy_model.eval()
        
        # 拼接查询和响应
        input_ids = torch.cat([query_ids, response_ids], dim=1)
        full_mask = torch.cat([
            query_ids.ne(self.tokenizer.pad_token_id), 
            masks
        ], dim=1)
        
        # 前向传播
        with torch.no_grad():
            outputs = self.policy_model(input_ids, attention_mask=full_mask)
            logits = outputs.logits
            values = outputs.value if hasattr(outputs, "value") else None
            
            # 只考虑响应部分的logits (偏移查询长度)
            response_logits = logits[:, query_ids.size(1)-1:-1, :]
            
            # 计算响应token的对数概率
            logprobs = F.log_softmax(response_logits, dim=-1)
            logprobs = torch.gather(
                logprobs, 
                dim=2, 
                index=response_ids.unsqueeze(2)
            ).squeeze(2)
            
            # 应用掩码
            logprobs = logprobs * masks
            if values is not None:
                values = values[:, query_ids.size(1)-1:-1] * masks.unsqueeze(-1)
        
        return logprobs, values

    def _clear_buffer(self):
        """清空经验缓冲区"""
        for key in self.buffer:
            self.buffer[key] = []

    def save_checkpoint(self, step, path="checkpoints", metadata=None):
        """
        保存模型检查点
        
        参数:
        step: 当前训练步数
        path: 保存路径
        metadata: 额外元数据
        """
        if not os.path.exists(path):
            os.makedirs(path)
        
        # 模型保存路径
        model_path = os.path.join(path, f"step_{step}")
        
        # 保存模型
        self.policy_model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # 保存训练状态
        checkpoint = {
            "step": step,
            "optimizer_state": self.optimizer.state_dict(),
            "kl_coef": self.kl_coef,
            "config": self.config,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        torch.save(checkpoint, os.path.join(model_path, "training_state.pt"))
        print(f"Saved checkpoint at step {step} to {model_path}")


class HumanPreferenceDataset(Dataset):
    def __init__(self, tokenizer, prompts_path="human_preference_prompts.json", max_length=256):
        """
        人类偏好提示数据集
        
        参数:
        tokenizer: 分词器
        prompts_path: 提示文件路径
        max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载提示数据
        if os.path.exists(prompts_path):
            with open(prompts_path, "r") as f:
                self.prompts = json.load(f)
        else:
            # 使用示例提示
            self.prompts = [
                "Write a comprehensive review of the latest smartphone features",
                "Explain quantum computing in simple terms",
                "Compose a haiku about the changing seasons",
                "Describe the benefits of renewable energy sources",
                "Write a persuasive paragraph about why education is important",
                "What are the potential risks of artificial intelligence?",
                "Tell me an inspiring short story about overcoming adversity",
                "Compare and contrast traditional and online education systems"
            ]
            print(f"Using built-in prompts. Save prompts to {prompts_path} for persistence.")
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx % len(self.prompts)]  # 循环使用提示
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze()
        }
    
    def save_prompts(self, path):
        """保存提示到文件"""
        with open(path, "w") as f:
            json.dump(self.prompts, f, indent=2)


def create_model(model_name="Qwen/Qwen2-7B", device_map="auto", use_lora=True):
    """创建模型实例并配置LoRA"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True
    )
    
    # 添加LoRA适配器 (高效微调)
    if use_lora:
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 针对Qwen2结构
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def main():
    # 训练配置
    config = {
        # PPO参数
        "lr": 2e-5,                       # 学习率
        "gamma": 0.99,                     # 折扣因子
        "lambda": 0.95,                    # GAE参数
        "init_kl_coef": 0.1,               # 初始KL惩罚系数
        "target_kl": 6.0,                  # KL目标值
        "adaptive_kl": True,               # 启用自适应KL控制
        "kl_adjust_rate": 0.01,            # KL调整速率
        "entropy_coef": 0.01,              # 熵奖励系数
        "value_coef": 0.5,                 # 价值损失权重
        "max_grad_norm": 1.0,              # 梯度裁剪阈值
        "clip_eps": 0.2,                   # PPO截断阈值
        
        # 训练参数
        "batch_size": 4,                   # 训练批次大小
        "buffer_size": 16,                 # 经验缓冲区大小
        "ppo_epochs": 2,                   # 每次缓冲区更新的PPO epoch数
        "total_steps": 200,                # 总训练步数
        "save_steps": 50,                  # 保存检查点的步数间隔
        
        # 生成参数
        "max_new_tokens": 128,             # 最大生成token数
        "min_length_penalty": 32,          # 最小长度奖励阈值
        "temperature": 0.7,                # 采样温度
        "top_p": 0.92,                     # 核采样阈值
        
        # 安全约束
        "repetition_penalty": 1.1,         # 重复惩罚因子
        "no_repeat_ngram_size": 3,          # 禁止重复n-gram大小
    }
    
    print("Initializing models...")
    
    # 1. 初始化模型
    base_model_name = "Qwen/Qwen2-7B"
    policy_model, tokenizer = create_model(base_model_name, use_lora=True)
    
    # 使用相同模型作为参考模型和奖励模型 (实际应用中奖励模型应不同)
    ref_model, _ = create_model(base_model_name, use_lora=False)
    reward_model, _ = create_model(base_model_name, use_lora=False)
    
    # 2. 准备优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, policy_model.parameters()),
        lr=config["lr"],
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    # 3. 准备数据集
    dataset = HumanPreferenceDataset(tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=config["batch_size"],
        shuffle=True
    )
    
    # 4. 初始化PPO训练器
    trainer = HumanPreferencePPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        config=config
    )
    
    print("Starting PPO training...")
    
    # 5. 训练循环
    step = 0
    progress_bar = tqdm(total=config["total_steps"], desc="PPO Training")
    
    while step < config["total_steps"]:
        for batch in dataloader:
            loss_dict = trainer.train_step(batch)
            
            # 更新进度条
            if loss_dict and "total_loss" in loss_dict:
                loss_val = loss_dict["total_loss"]
                if isinstance(loss_val, torch.Tensor):
                    loss_val = loss_val.item()
                
                progress_bar.set_postfix({
                    "loss": f"{loss_val:.4f}",
                    "kl_coef": f"{trainer.kl_coef:.4f}"
                })
            
            step += 1
            progress_bar.update(1)
            
            # 保存检查点
            if step % config["save_steps"] == 0:
                trainer.save_checkpoint(step, metadata={
                    "loss": loss_dict,
                    "step": step
                })
            
            if step >= config["total_steps"]:
                break
    
    progress_bar.close()
    
    # 6. 保存最终模型
    final_path = "human_preference_tuned_qwen"
    policy_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    # 内存优化
    torch.cuda.empty_cache()
    gc.collect()
    
    # 启动训练
    main()