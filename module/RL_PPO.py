
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer

class PPO(nn.Module):
    def __init__(self, model_name,ref_model_name,reward_model,ppo_epoch,batch_size,max_length,gamma = 0.99,
                  lam = 0.95,clip_eps = 0.2,kl_coef=0.1,vf_coef = 0.5,lr = 2e-5,grad_clip = 1.0):
        super(PPO).__init__()

