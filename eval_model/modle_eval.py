

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from nltk import ngrams
from nltk.util import everygrams
from collections import Counter

class LLMEvaluator:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def compute_perplexity(self, texts: List[str], batch_size: int = 16) -> float:
        """计算困惑度，支持大批量自动分割"""
        self.model.eval()
        total_loss = 0
        total_length = 0

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Computing PPL"):
                batch = texts[i:i+batch_size]
                inputs = self.tokenizer(
                    batch, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                ).to(self.device)
                
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                total_loss += loss.item() * inputs["input_ids"].size(1)
                total_length += inputs["input_ids"].size(1)
        
        return np.exp(total_loss / total_length)

    def compute_diversity(self, generated_texts: List[str], n_gram_range: Tuple[int, int] = (1, 4)) -> Dict:
        """计算文本多样性指标，包含n-gram统计和重复率"""
        diversity_metrics = {}
        all_ngrams = defaultdict(list)
        
        # 计算不同n-gram的unique比例
        for n in range(n_gram_range[0], n_gram_range[1]+1):
            ngram_counts = Counter()
            total_ngrams = 0
            
            for text in generated_texts:
                tokens = self.tokenizer.tokenize(text)
                text_ngrams = list(ngrams(tokens, n))
                ngram_counts.update(text_ngrams)
                total_ngrams += len(text_ngrams)
            
            unique_ngrams = len(ngram_counts)
            diversity_metrics[f'unique_{n}gram_ratio'] = unique_ngrams / total_ngrams if total_ngrams > 0 else 0
            diversity_metrics[f'repetition_{n}gram'] = 1 - diversity_metrics[f'unique_{n}gram_ratio']
        
        # 计算自重复率
        duplicate_ratio = len(generated_texts) / len(set(generated_texts)) - 1
        diversity_metrics['exact_duplicate_ratio'] = max(0.0, duplicate_ratio)
        
        return diversity_metrics

    def evaluate_on_dataset(self, dataset_name: str, subset: str = None, 
                          split: str = 'test', max_samples: int = 1000) -> Dict:
        """在指定开源数据集上进行评估"""
        dataset = self._load_dataset(dataset_name, subset, split)
        samples = self._preprocess_dataset(dataset, dataset_name, max_samples)
        
        # 计算困惑度
        ppl = self.compute_perplexity(samples['text'])
        
        # 生成文本多样性评估
        generated_texts = self.generate_texts(num_samples=100, max_length=100)
        diversity = self.compute_diversity(generated_texts)
        
        return {
            'dataset': dataset_name,
            'perplexity': ppl,
            'diversity_metrics': diversity,
            'sample_size': len(samples['text'])
        }

    def _load_dataset(self, name: str, subset: str, split: str):
        """加载常见开源数据集"""
        dataset_mapping = {
            'wikitext': ('wikitext', 'wikitext-103-raw-v1'),
            'c4': ('c4', 'en'),
            'pile': ('EleutherAI/pile', None)
        }
        
        if name in dataset_mapping:
            dataset_args = dataset_mapping[name]
            return load_dataset(dataset_args[0], dataset_args[1], split=split, streaming=True)
        
        return load_dataset(name, subset, split=split, streaming=True)

    def _preprocess_dataset(self, dataset, dataset_name: str, max_samples: int) -> Dict:
        """数据集预处理标准化"""
        samples = {'text': []}
        
        # 不同数据集的字段处理
        text_columns = {
            'wikitext': 'text',
            'c4': 'text',
            'pile': 'text'
        }
        
        column = text_columns.get(dataset_name, 'text')
        count = 0
        
        for item in dataset:
            if count >= max_samples:
                break
            if item[column].strip():
                samples['text'].append(item[column])
                count += 1
        
        return samples

    def generate_texts(self, num_samples: int = 100, max_length: int = 100) -> List[str]:
        """生成评估用文本"""
        generated_texts = []
        
        for _ in range(num_samples):
            inputs = self.tokenizer(
                "", 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=50
            ).to(self.device)
            
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                do_sample=True,
                top_p=0.95,
                temperature=0.9
            )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_texts.append(generated)
        
        return generated_texts

    @staticmethod
    def visualize_results(results: Dict):
        """可视化评估结果"""
        # 困惑度可视化
        plt.figure(figsize=(12, 6))
        datasets = [r['dataset'] for r in results]
        ppls = [r['perplexity'] for r in results]
        plt.bar(datasets, ppls)
        plt.title('Perplexity Comparison')
        plt.ylabel('Perplexity')
        plt.xticks(rotation=45)
        plt.show()

        # 多样性可视化
        n_grams = sorted([k for k in results[0]['diversity_metrics'] if 'unique' in k])
        plt.figure(figsize=(12, 6))
        
        for metric in n_grams:
            values = [r['diversity_metrics'][metric] for r in results]
            plt.plot(datasets, values, marker='o', label=metric)
        
        plt.title('Diversity Metrics')
        plt.legend()
        plt.xticks(rotation=45)
        plt.ylabel('Ratio')
        plt.show()

if __name__ == "__main__":
    # 使用示例
    evaluator = LLMEvaluator("")
    
    # 定义要评估的数据集
    datasets_to_evaluate = [
        {"name": "wikitext"},
        {"name": "c4", "subset": "en"},
        {"name": "EleutherAI/pile", "subset": None}
    ]
    
    results = []
    for dataset in datasets_to_evaluate:
        result = evaluator.evaluate_on_dataset(
            dataset["name"], 
            subset=dataset.get("subset"),
            max_samples=500
        )
        results.append(result)
        print(f"Results for {dataset['name']}:")
        print(f"Perplexity: {result['perplexity']:.2f}")
        print(f"Diversity (1-gram): {result['diversity_metrics']['unique_1gram_ratio']:.2%}")
        print("="*50)
    
    # 结果可视化
    evaluator.visualize_results(results)