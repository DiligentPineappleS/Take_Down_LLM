import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed
)
from peft import LoraConfig, get_peft_model
import os
import json
import gc
from tqdm import tqdm
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# 设置随机种子
set_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True  # 启用TF32加速
torch.autograd.set_detect_anomaly(True)  # 检测异常梯度

class PreferenceDataset(Dataset):
    """处理三元组偏好数据集"""
    def __init__(self, tokenizer, data_path, max_length=256, debug=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载数据集
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        if debug:
            self.data = self.data[:100]  # 调试时只使用少量样本
        
        print(f"Loaded {len(self.data)} preference samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        prompt = sample["prompt"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]
        
        # 编码提示
        prompt_enc = self.tokenizer(
            prompt, 
            max_length=self.max_length // 2,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        
        # 编码优质的响应
        chosen_enc = self.tokenizer(
            chosen, 
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        
        # 编码劣质的响应
        rejected_enc = self.tokenizer(
            rejected, 
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        
        return {
            "prompt": prompt,
            "prompt_input_ids": prompt_enc["input_ids"],
            "prompt_attention_mask": prompt_enc.get("attention_mask", [1]*len(prompt_enc["input_ids"])),
            "chosen_input_ids": chosen_enc["input_ids"],
            "chosen_attention_mask": chosen_enc.get("attention_mask", [1]*len(chosen_enc["input_ids"])),
            "rejected_input_ids": rejected_enc["input_ids"],
            "rejected_attention_mask": rejected_enc.get("attention_mask", [1]*len(rejected_enc["input_ids"])),
        }

class DPOTrainer(Trainer):
    """自定义DPO训练器"""
    def __init__(self, *args, beta=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta  # DPO损失的温度参数
        self.loss_history = []
        self.accuracy_history = []
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """计算DPO损失"""
        # 获取提示和响应
        prompt_input_ids = inputs["prompt_input_ids"]
        chosen_input_ids = inputs["chosen_input_ids"]
        rejected_input_ids = inputs["rejected_input_ids"]
        
        # 创建完整的输入序列 (prompt + response)
        chosen_sequences = [
            prompt_input_ids[i] + chosen_input_ids[i] for i in range(len(prompt_input_ids))
        ]
        rejected_sequences = [
            prompt_input_ids[i] + rejected_input_ids[i] for i in range(len(prompt_input_ids))
        ]
        
        # 找到响应开始的位置 (用于掩码响应部分)
        response_starts = [len(prompt_input_ids[i]) for i in range(len(prompt_input_ids))]
        
        # 填充批次
        chosen_padded = self.tokenizer.pad(
            [{"input_ids": seq} for seq in chosen_sequences],
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.args.device)
        
        rejected_padded = self.tokenizer.pad(
            [{"input_ids": seq} for seq in rejected_sequences],
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.args.device)
        
        # 创建注意力掩码
        chosen_attention_mask = chosen_padded["attention_mask"]
        rejected_attention_mask = rejected_padded["attention_mask"]
        
        # 前向传播优质的响应
        chosen_outputs = model(
            input_ids=chosen_padded["input_ids"],
            attention_mask=chosen_attention_mask,
            output_hidden_states=False,
            output_attentions=False
        )
        
        # 前向传播劣质的响应
        with torch.no_grad():
            rejected_outputs = model(
                input_ids=rejected_padded["input_ids"],
                attention_mask=rejected_attention_mask,
                output_hidden_stances=False,
                output_attentions=False
            )
        
        # 获取对数概率 (logits)
        chosen_logits = chosen_outputs.logits
        rejected_logits = rejected_outputs.logits
        
        # 准备标签 (偏移一个位置)
        chosen_labels = chosen_padded["input_ids"][:, 1:].contiguous()
        chosen_logits = chosen_logits[:, :-1, :].contiguous()
        rejected_labels = rejected_padded["input_ids"][:, 1:].contiguous()
        rejected_logits = rejected_logits[:, :-1, :].contiguous()
        
        # 掩码掉提示部分 (只计算响应的损失)
        mask_indices = []
        for i in range(len(chosen_sequences)):
            mask_length = len(chosen_sequences[i]) - response_starts[i] - 1
            mask = [0] * (len(chosen_sequences[i]) - mask_length - 1) + [1] * mask_length
            mask_indices.append(mask)
            
        # 填充掩码并转换为张量
        mask_tensor = torch.tensor([
            mask + [0] * (chosen_logits.size(1) - len(mask)) 
            for mask in mask_indices
        ], dtype=torch.bool, device=self.args.device)
        
        # 计算优质的序列的损失 (只计算响应部分)
        chosen_losses = F.cross_entropy(
            chosen_logits.view(-1, chosen_logits.size(-1)),
            chosen_labels.view(-1),
            reduction='none'
        ).view(chosen_logits.size(0), chosen_logits.size(1))
        
        chosen_loss = (chosen_losses * mask_tensor).sum() / mask_tensor.sum()
        
        # 计算劣质的序列的损失 (只计算响应部分)
        rejected_losses = F.cross_entropy(
            rejected_logits.view(-1, rejected_logits.size(-1)),
            rejected_labels.view(-1),
            reduction='none'
        ).view(rejected_logits.size(0), rejected_logits.size(1))
        
        rejected_loss = (rejected_losses * mask_tensor).sum() / mask_tensor.sum()
        
        # 计算DPO损失
        dpo_loss = -F.logsigmoid(self.beta * (rejected_loss - chosen_loss))
        
        # 记录损失和准确率
        self.loss_history.append(dpo_loss.item())
        self.accuracy_history.append(torch.mean((rejected_loss > chosen_loss).float()).item())
        
        return (dpo_loss, chosen_outputs) if return_outputs else dpo_loss
    
    def plot_training_history(self, output_dir):
        """绘制训练历史图表"""
        plt.figure(figsize=(12, 5))
        
        # 损失图表
        plt.subplot(1, 2, 1)
        plt.plot(self.loss_history, label="DPO Loss")
        plt.title("Training Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # 准确率图表
        plt.subplot(1, 2, 2)
        plt.plot(self.accuracy_history, label="Preference Accuracy", color="green")
        plt.title("Preference Accuracy")
        plt.xlabel("Step")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_history.png"))
        print("Saved training history plot")
    
    def save_model(self, output_dir):
        """保存模型"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存模型
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # 保存训练状态
        torch.save({
            "loss_history": self.loss_history,
            "accuracy_history": self.accuracy_history
        }, os.path.join(output_dir, "training_state.pt"))
        
        print(f"Saved model to {output_dir}")

def create_model(model_name, use_lora=True):
    """创建模型并应用LoRA"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 应用LoRA适配器
    if use_lora:
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],  # Qwen2的注意力投影层
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer

def collate_fn(batch, tokenizer, max_length):
    """数据整理函数"""
    # 创建批处理数据
    collated = {
        "prompt": [],
        "prompt_input_ids": [],
        "prompt_attention_mask": [],
        "chosen_input_ids": [],
        "chosen_attention_mask": [],
        "rejected_input_ids": [],
        "rejected_attention_mask": [],
    }
    
    for item in batch:
        for key in collated:
            if key in item:
                collated[key].append(item[key])
    
    # 对输入序列进行填充
    def pad_sequence(sequences, pad_value):
        max_len = max(len(seq) for seq in sequences)
        padded = []
        for seq in sequences:
            padding = max_len - len(seq)
            padded.append(seq + [pad_value] * padding)
        return torch.tensor(padded)
    
    # 填充所有序列到最大长度
    pad_token_id = tokenizer.pad_token_id
    
    # 填充提示序列
    padded_prompt = pad_sequence(collated["prompt_input_ids"], pad_token_id)
    collated["prompt_input_ids"] = padded_prompt
    
    # 使用1填充注意力掩码
    padded_attn = pad_sequence(
        [mask + [0] * (padded_prompt.size(1) - len(mask)) 
         for mask in collated["prompt_attention_mask"]], 
        0
    )
    collated["prompt_attention_mask"] = padded_attn
    
    # 填充优质的响应
    padded_chosen = pad_sequence(collated["chosen_input_ids"], pad_token_id)
    collated["chosen_input_ids"] = padded_chosen
    
    padded_attn = pad_sequence(
        [mask + [0] * (padded_chosen.size(1) - len(mask)) 
         for mask in collated["chosen_attention_mask"]], 
        0
    )
    collated["chosen_attention_mask"] = padded_attn
    
    # 填充劣质的响应
    padded_rejected = pad_sequence(collated["rejected_input_ids"], pad_token_id)
    collated["rejected_input_ids"] = padded_rejected
    
    padded_attn = pad_sequence(
        [mask + [0] * (padded_rejected.size(1) - len(mask)) 
         for mask in collated["rejected_attention_mask"]], 
        0
    )
    collated["rejected_attention_mask"] = padded_attn
    
    return collated

def train_dpo(model, tokenizer, dataset, output_dir="dpo_model", beta=0.1, **kwargs):
    """训练DPO模型"""
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=kwargs.get("batch_size", 2),
        gradient_accumulation_steps=kwargs.get("gradient_accumulation", 4),
        learning_rate=kwargs.get("lr", 5e-6),
        num_train_epochs=kwargs.get("epochs", 3),
        report_to="none",
        logging_dir="./logs",
        logging_steps=10,
        save_steps=kwargs.get("save_steps", 500),
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        max_steps=kwargs.get("max_steps", -1)
    )
    
    # 自定义数据整理器
    collate_function = lambda batch: collate_fn(batch, tokenizer, kwargs.get("max_length", 256))
    
    # 创建DPO训练器
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        beta=beta,
        train_dataset=dataset,
        data_collator=collate_function,
        tokenizer=tokenizer
    )
    
    # 开始训练
    print("Starting DPO training...")
    trainer.train()
    
    # 保存模型和训练历史
    trainer.save_model(output_dir)
    trainer.plot_training_history(output_dir)
    
    return trainer

def generate_sample(model, tokenizer, prompt, max_length=128):
    """生成响应样本"""
    input_ids = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=256
    ).input_ids.to(model.device)
    
    # 生成响应
    outputs = model.generate(
        input_ids,
        max_length=max_length + input_ids.size(1),
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True)

def compare_results(model, tokenizer, dataset, num_samples=5):
    """对比训练前后的生成结果"""
    print("\n=== 生成结果对比 ===")
    
    # 优质样本
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for idx in indices:
        sample = dataset[idx]
        prompt = sample["prompt"]
        
        print(f"\nPrompt: {prompt}")
        
        # 生成预训练模型的响应
        with torch.no_grad():
            base_response = generate_sample(model.base_model, tokenizer, prompt)
        
        # 生成DPO优化后的响应
        optimized_response = generate_sample(model, tokenizer, prompt)
        
        print(f"Base Model Response: {base_response}")
        print(f"DPO Model Response:  {optimized_response}")
        
        # 原始优质
        print(f"Original Chosen:    {sample['chosen']}")
        print(f"Original Rejected:  {sample['rejected']}")

def main():
    # 配置
    config = {
        "model_name": "Qwen/Qwen2-7B",
        "data_path": "preference_data.json",  # 三元组偏好数据
        "output_dir": "dpo_tuned_qwen",
        "use_lora": True,
        "beta": 0.1,  # DPO损失参数
        "batch_size": 2,
        "gradient_accumulation": 4,
        "lr": 5e-6,
        "epochs": 2,
        "max_steps": -1,  # 所有步骤
        "max_length": 256,  # 最大序列长度
        "debug": False     # 调试模式
    }
    
    # 1. 加载模型和分词器
    print("Loading model and tokenizer...")
    model, tokenizer = create_model(config["model_name"], config["use_lora"])
    
    # 2. 准备数据集
    print("Loading dataset...")
    dataset = PreferenceDataset(
        tokenizer, 
        config["data_path"], 
        max_length=config["max_length"],
        debug=config["debug"]
    )
    
    # 3. DPO训练
    print("Training with DPO...")
    dpo_trainer = train_dpo(
        model,
        tokenizer,
        dataset,
        output_dir=config["output_dir"],
        beta=config["beta"],
        batch_size=config["batch_size"],
        gradient_accumulation=config["gradient_accumulation"],
        lr=config["lr"],
        epochs=config["epochs"],
        max_steps=config["max_steps"],
        max_length=config["max_length"]
    )
    
    # 4. 结果对比
    compare_results(model, tokenizer, dataset)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    # 清理缓存
    torch.cuda.empty_cache()
    gc.collect()
    
    # 启动训练
    main()