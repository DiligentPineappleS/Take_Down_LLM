import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader 
from transformers import AutoModelForCausalLM,AutoTokenizer,Adamw,get_linear_schedule_with_warmup
from tqdm import tqdm
import json
import argparse
import os

parse = argparse.ArgumentParser()
parse.add_argument("--model_name",type=str,default="qwen/qwen2.0-7B",help="模型名称")
parse.add_argument("--datapath",type=str,default="./",help="训练数据的路径")
parse.add_argument("--lora_rank",type=int,default=8,help="lora秩")
parse.add_argument("--alpha",type=int,default=32,help="lora缩放因子")
parse.add_argument("--target_modules",type=list,default=["q_proj","v_proj"],help="lora微调的应用模块")
parse.add_argument("--lr",type=float,default=1e-5,help="学习率")
parse.add_argument("--batch_size",type=int,default=32,help="模型训练批大小")
parse.add_argument("--epoches",type=int,default=10,help="训练的轮数")
parse.add_argument("--save_dir",type=str,default="./save_model",help="模型保存路径")
parse.add_argument("--max_len",type=int,default=1024,help="序列最大长度")
parse.add_argument("--warmup",type=int,default=1024,help="序列最大长度")
parse.add_argument("--max_grad_norm",type=float,default=1,help="序列最大长度")
parse.add_argument("--seed",type=int,default=64,help="序列最大长度")



class LoraLayer(nn.Module):
    def __init__(self, model_layers,rank,alpha):
        super(LoraLayer,self).__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha/rank
        
        self.input_dim = model_layers.in_features
        self.output_dim =  model_layers.output_features

        self.layer_weight = nn.Parameter(model_layers.weight.data.detach().clone(),requires_grad=False)
        if model_layers.bias is not None:
            self.layer_bias =  nn.Parameter(model_layers.bias.data.detach().clone(),requires_grad=False)
        else:
            self.layer_bias = None
        # 初始化LoRA的A和B矩阵
        # A矩阵:初始化为随机高斯分布 (in_features, rank)
        # B矩阵:初始化为零 (rank, out_features) 
        self.lora_a = nn.Parameter(torch.empty(self.input_dim,self.rank))
        self.lora_b =  nn.Parameter(torch.empty(self.rank,self.output_dim))
        # 防止​​梯度消失/爆炸​​问题;保持每一层输出的​​方差不变​
        torch.nn.init.kaiming_uniform_(self.lora_a)
        torch.nn.init.zeros_(self.rank,self.output_dim)
    def forward(self,x):
        layer_output = F.linear(x,weight=self.layer_weight,bias=self.layer_bias)
        lora_output = torch.matmul(torch.matmul(x,self.lora_a),self.lora_b)*self.scaling
        return layer_output + lora_output
    def merge_weight(self):

        lora_matrix = torch.matmul(self.lora_a,self.lora_b)*self.scaling.transpose()
        return lora_matrix + self.layer_weight



class Lora(nn.Module):
    def __init__(self, model,target_modules,rank,alpha):
        super(Lora,self).__init__()

        self.model = model
        self.target_modules = target_modules
        self.rank = rank
        self.alpha = alpha

        self.lora_modules = {}

        for name,module in list(self.model.named_modules()):
            if any(target_module in name for target_module in self.target_modules ):
                if isinstance(module,nn.Linear):
                    lora_layer = LoraLayer(model_layers=module,rank=self.rank,alpha=self.alpha)
                    parent_name = self.get_parent_name(self.model,name)
                    if parent_name is not None:
                        child_name = name.split(".")[-1]
                        setattr(parent_name,child_name,lora_layer)
                        self.lora_modules[name] = lora_layer
        for param in model.parameters():
            param.resquires_grad = False
        for param in self.lora_modules.parameters():
            param.resquires_grad = True

    def get_parent_name(self,model,name):
        name_lsit = name.split(".")
        if len(name_lsit)==1:
            return model
        name_parent = ".".join(name_lsit[:-1])
        parent_name = model
        for temp_name in name_parent:
            parent = getattr(parent_name,temp_name,None)
            if parent is None:
                return None
        return parent_name

    def computer_loss(self,model_logits,target,attention_mask):
        attention_mask_move =None
        if attention_mask:
            attention_mask_move = attention_mask[...,1:].contiguous()
        model_logits_move = model_logits[...,:-1,:].contiguous()
        target_move =  target[...,1:].contiguous()
        per_token_loss = F.cross_entropy(model_logits_move.view(-1,model_logits_move.size(-1)),target_move.view(-1),reduction= 'none')
        per_token_loss = per_token_loss.view(target_move.size())
        if not attention_mask_move:
            per_token_loss = per_token_loss*attention_mask_move
            loss = per_token_loss.sum()
            num_tokens = max(attention_mask_move.sum(),1)
        else:
            loss = per_token_loss.sum()
            num_tokens = per_token_loss.numel()
        return loss/num_tokens
    

    def forward(self,input_ids,attention_mask,target,**kwarg):
        assert input_ids is not None,"input_ids canot be None"
        if not attention_mask:
            attention_mask = torch.ones_like(input_ids)
        model_output = self.model(input_ids = input_ids,attention_mask = attention_mask,**kwarg)
        if not target:
            return model_output
        model_output_logits = model_output.logits if hasattr(model_output,'logits') else model_output[0]
        loss = self.computer_loss(model_output_logits,target,attention_mask)
        result  =  {"loss":loss,"logits":model_output_logits}
        if isinstance(model_output,dict):
            for key,value in model_output.items():
                if key not in ["loss","logits"]:
                    result[key] = value
        else:
            for attr in ['hidden_states', 'attentions', 'past_key_values']:
                if hasattr(model_output, attr):
                    setattr(result, attr, getattr(model_output, attr))
        return result
    def save_lora(self,save_path):
        os.makedirs(save_path,exist_ok=True)
        states_lora = {}
        for name , module in self.lora_modules.items():
            states_lora[f"{name}.lora_a"] = module.lora_a
            states_lora[f"{name}.lora_b"] = module.lora_b
        torch.save(states_lora,save_path)
    def save_merge_model(self,save_path):
        merge_model = self.model

        for name,loralayer in self.lora_modules.items():
            parent_name = self.get_parent_name(merge_model,name)
            if parent_name:
                child_name = name.split(".")[-1]
                model_module = getattr(parent_name,child_name)
                model_module.weight.data = loralayer.merge_weight()
                if hasattr(loralayer,'layer_bias') or loralayer.layer_bias is not None:
                    model_module.bias.data = loralayer.layer_bias


    
        merge_model.save_pretained(save_path)



class TextDataset(Dataset):
    """文本数据集类"""
    def __init__(self, tokenizer, file_path, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据集
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.json'):
                self.data = json.load(f)
            elif file_path.endswith('.jsonl'):
                self.data = [json.loads(line) for line in f]
        
        # 检测文本字段
        if 'text' in self.data[0]:
            self.text_field = 'text'
        elif 'prompt' in self.data[0]:
            self.text_field = 'prompt'
        else:
            for key in self.data[0]:
                if isinstance(self.data[0][key], str):
                    self.text_field = key
                    break
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx][self.text_field]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }

def train_epoch(model, tokenizer, train_loader, optimizer, scheduler, device, epoch):
    """训练单个epoch的函数式实现"""
    model.train()
    total_loss = 0
    
    for step, batch in enumerate(train_loader):
        # 移动到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播和计算损失
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        
        # 反向传播
        loss.backward()
        
        # 参数更新
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # 记录损失
        total_loss += loss.item()
        
        # 打印进度
        if step % 50 == 0:
            avg_loss = total_loss / (step + 1)
            print(f"Epoch {epoch+1} | Step {step}/{len(train_loader)} | Loss: {avg_loss:.4f}")
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} completed | Average Loss: {avg_loss:.4f}")
    return avg_loss

def main():
    """主训练函数"""
    args = parse_args()
    set_seed()
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
    ).to(device)
    
    # 应用LoRA
    print("Applying LoRA...")
    lora_model = LoRAModelWrapper(
        model, 
        args.target_modules, 
        args.lora_rank, 
        args.alpha
    ).to(device)
    
    # 准备数据集
    print("Loading dataset...")
    train_dataset = TextDataset(tokenizer, args.dataset_path, args.max_seq_length)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    # 优化器（只优化LoRA参数）
    params_to_optimize = []
    for module in lora_model.lora_modules.values():
        params_to_optimize += list(module.parameters())
    
    optimizer = AdamW(params_to_optimize, lr=args.learning_rate)
    
    # 学习率调度器
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 训练循环
    print("Starting training...")
    for epoch in range(args.num_epochs):
        avg_loss = train_epoch(
            lora_model, 
            tokenizer, 
            train_loader, 
            optimizer, 
            scheduler, 
            device, 
            epoch
        )
        
        # 保存检查点
        epoch_dir = os.path.join(args.save_dir, f"epoch_{epoch+1}")
        lora_model.save_lora_weights(epoch_dir)
        
        # 简单测试
        test_input = "用户：深度学习的核心概念是什么？"
        inputs = tokenizer(test_input, return_tensors="pt").to(device)
        outputs = lora_model(input_ids=inputs['input_ids'])
        last_token_logits = outputs.logits[0, -1, :]
        top5 = torch.topk(torch.softmax(last_token_logits, dim=-1), 5)
        print("\nTest output:")
        print("Input:", test_input)
        print("Top 5 next tokens:", tokenizer.convert_ids_to_tokens(top5.indices.tolist()))
    
    # 保存最终模型
    final_lora_dir = os.path.join(args.save_dir, "final_lora")
    final_merged_dir = os.path.join(args.save_dir, "final_merged_model")
    
    lora_model.save_lora_weights(final_lora_dir)
    lora_model.merge_and_save(final_merged_dir)
    
    print(f"\nTraining complete! Models saved at {args.save_dir}")
    
    # 最终测试
    test_input = "用户：如何学习深度学习？"
    inputs = tokenizer(test_input, return_tensors="pt").to(device)
    outputs = lora_model.generate(
        input_ids=inputs['input_ids'],
        max_length=200,
        do_sample=True,
        temperature=0.7
    )
    print("\nFinal test output:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()








