from datasketch import MinHash, MinHashLSH
from nltk import ngrams
import re
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import NearestNeighbors

# 数据去重
class DataDeduplicator:
    def __init__(self, threshold=0.85):
        """
        文本去重处理器
        :param threshold: 相似度阈值，>此值视为重复
        """
        self.threshold = threshold
        
    def _simhash(self, text, hash_bits=64):
        """
        SimHash算法实现 (局部敏感哈希)
        生成文本hash，相似文本的海明距离小
        加权位运算，文本特征映射到固定位数的二进制签名
        """
        # 步骤1: 文本向量化
        vectorizer = HashingVectorizer(n_features=2**hash_bits)
        vectors = vectorizer.fit_transform([text]).toarray()[0]
        
        # 步骤2: 加权特征处理 (此处简化，实际可优化)
        binary_signature = np.zeros(hash_bits, dtype=int)
        for i, val in enumerate(vectors):
            if val != 0:  # 只处理非零特征
                # 计算位权重：特征值大小影响权重
                weight = abs(val)
                binary_signature[i % hash_bits] += weight
        
        # 步骤3: 生成二进制指纹
        return ''.join(['1' if x > 0 else '0' for x in binary_signature])
    
    def _hamming_distance(self, hash1, hash2):
        """计算海明距离"""
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    
    def semantic_dedup(self, texts):
        """
        语义去重：基于MinHash LSH
        检测语义相似的文档，解决同义不同词问题
        使用Jaccard相似度和局部敏感哈希快速检索
        """
        # 创建LSH索引
        lsh = MinHashLSH(threshold=self.threshold, num_perm=128)
        minhashes = []
        
        for idx, text in enumerate(texts):
            # 创建MinHash对象
            m = MinHash(num_perm=128)
            
            # 提取n-grams特征 (n=3)
            tokens = re.sub(r'[^\w\s]', '', text.lower()).split()
            for ng in ngrams(tokens, 3):
                m.update(''.join(ng).encode('utf-8'))
            
            # 存储MinHash并加入索引
            minhashes.append(m)
            lsh.insert(idx, m)
        
        # 检测并移除重复项
        unique_idx = set()
        duplicates = set()
        
        for idx in range(len(texts)):
            if idx not in duplicates:
                unique_idx.add(idx)
                # 查找相似项
                similar = lsh.query(minhashes[idx])
                duplicates.update(set(similar) - {idx})
                
        return [texts[i] for i in unique_idx]
    
    def deduplicate(self, texts):
        """双阶段去重：先SimHash快速去重，再语义精确去重"""
        # 第一阶段：SimHash去重
        simhashes = {}
        stage1_unique = []
        
        for text in texts:
            sh = self._simhash(text)
            
            # 检查是否有相似哈希存在
            duplicate_found = False
            for ref_hash in simhashes:
                if self._hamming_distance(sh, ref_hash) < (1 - self.threshold) * len(sh):
                    duplicate_found = True
                    break
                    
            if not duplicate_found:
                stage1_unique.append(text)
                simhashes[sh] = True
        
        # 第二阶段：语义精确去重
        return self.semantic_dedup(stage1_unique)

## 精确哈希去重
## 使用sha256进行去重
import hashlib
from collections import defaultdict
def content_hash_deduplicate(texts):
    """精确哈希去重"""
    seen_hashes = set()
    unique_texts = []
    
    for text in texts:
        # 使用SHA256生成内容哈希
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        
        # 检查是否已存在
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            unique_texts.append(text)
    
    return unique_texts

## 滑动窗口局部哈希去重
## 使用md5进行去重
def sliding_window_deduplicate(texts, window_size=100):
    """滑动窗口局部哈希去重"""
    seen_windows = defaultdict(set)
    unique_texts = []
    
    for text in texts:
        # 跳过空文本
        if not text: continue
        
        # 拆分文本为重叠窗口
        windows = []
        words = text.split()
        for i in range(len(words) - window_size + 1):
            window = " ".join(words[i:i+window_size])
            win_hash = hashlib.md5(window.encode()).hexdigest()
            windows.append(win_hash)
        
        # 检查是否有重复窗口
        duplicate = False
        for win_hash in windows:
            if win_hash in seen_windows.get(len(words), set()):
                duplicate = True
                break
                
        if not duplicate:
            unique_texts.append(text)
            seen_windows[len(words)].update(windows)
            
    return unique_texts



import mmh3
from datasketch import MinHash, MinHashLSH
import numpy as np

class MultiLevelMinHash:
    def __init__(self, num_perm=128, bands=10, rows=12):
        """多级MinHash优化结构"""
        self.lsh = MinHashLSH(num_perm=num_perm, params=(bands, rows))
        self.minhashes = []
        self.signatures = []
        self.texts = []
        
    def add_text(self, text):
        """添加文本到索引"""
        # 创建MinHash对象
        m = MinHash(num_perm=self.lsh.num_perm)
        
        # 提取n-grams特征
        words = text.split()
        shingles = set()
        for i in range(len(words)-2):
            shingle = " ".join(words[i:i+3])
            shingles.add(shingle)
        
        # 添加到MinHash
        for s in shingles:
            m.update(s.encode('utf-8'))
        
        # 计算签名向量
        signature = np.array([m.hashvalues[i] for i in range(self.lsh.num_perm)])
        
        # 添加到索引
        idx = len(self.texts)
        self.minhashes.append(m)
        self.signatures.append(signature)
        self.texts.append(text)
        self.lsh.insert(idx, m)
        
    def deduplicate(self):
        """执行去重"""
        duplicates = set()
        
        for idx in range(len(self.texts)):
            # 如果已被标记为重复则跳过
            if idx in duplicates:
                continue
                
            # 查找相似项
            similar = self.lsh.query(self.minhashes[idx])
            for similar_idx in similar:
                if similar_idx != idx:
                    # 余弦相似度验证
                    dot = np.dot(self.signatures[idx], self.signatures[similar_idx])
                    norm1 = np.linalg.norm(self.signatures[idx])
                    norm2 = np.linalg.norm(self.signatures[similar_idx])
                    similarity = dot / (norm1 * norm2)
                    
                    if similarity > 0.8:  # 相似度阈值
                        duplicates.add(similar_idx)
        
        # 返回唯一文本
        return [self.texts[i] for i in range(len(self.texts)) if i not in duplicates]

## 加权MinHash去重
from datasketch import WeightedMinHashGenerator

class WeightedMinHashDeduplicator:
    def __init__(self, dim=1000):
        """加权MinHash去重，支持带权特征"""
        self.wm_gen = WeightedMinHashGenerator(dim=dim)
        self.lsh = MinHashLSH()
        self.text_weights = {}
        
    def _text_to_weights(self, text):
        """将文本转换为加权特征向量"""
        # 此处使用TF-IDF简化版，实际应替换为完整实现
        words = text.split()
        word_count = {}
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1
            
        # 归一化
        max_count = max(word_count.values()) if word_count else 1
        return {word: count/max_count for word, count in word_count.items()}
    
    def add_text(self, text):
        """添加文本"""
        weights = self._text_to_weights(text)
        wm = self.wm_gen.minhash(weights)
        idx = len(self.text_weights)
        self.text_weights[idx] = weights
        self.lsh.insert(idx, wm)
        
    def deduplicate(self, threshold=0.8):
        """执行去重"""
        # 类似MinHash去重流程...



import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from sklearn.neighbors import NearestNeighbors

class SiameseDeduplicator:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """孪生网络语义去重器"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.embeddings = []
        self.texts = []
    
    def embed_texts(self, texts):
        """生成文本嵌入向量"""
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]  # CLS token
            embeddings = torch.nn.functional.normalize(embeddings)
            
        return embeddings.cpu().numpy()
    
    def add_texts(self, texts):
        """批量添加文本"""
        if not texts:
            return
            
        # 分批处理避免OOM
        batch_size = 128
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.embed_texts(batch_texts)
            self.embeddings.extend(batch_embeddings)
            self.texts.extend(batch_texts)
            
    def deduplicate(self, threshold=0.85):
        """语义相似性去重"""
        # 创建索引
        index = NearestNeighbors(n_neighbors=10, metric="cosine")
        embeddings_array = np.array(self.embeddings)
        index.fit(embeddings_array)
        
        # 检测重复
        duplicates = set()
        for idx, emb in enumerate(embeddings_array):
            if idx in duplicates:
                continue
                
            # 查找最相似项
            distances, indices = index.kneighbors([emb], n_neighbors=10)
            for i, dist in zip(indices[0][1:], distances[0][1:]):  # 跳过自己
                similarity = 1 - dist
                if similarity > threshold:
                    duplicates.add(i)
                    
        return [self.texts[i] for i in range(len(self.texts)) if i not in duplicates]





import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class GraphDeduplicator:
    def __init__(self, sim_threshold=0.8):
        """基于图的去重方法"""
        self.graph = nx.Graph()
        self.text_map = {}
        self.sim_threshold = sim_threshold
        
    def add_texts(self, texts, embeddings):
        """添加文本及其嵌入向量"""
        for idx, (text, emb) in enumerate(zip(texts, embeddings)):
            self.text_map[idx] = text
            self.graph.add_node(idx, embedding=emb)
    
    def build_similarity_edges(self):
        """构建相似度边"""
        # 获取所有嵌入
        embeddings = [self.graph.nodes[idx]['embedding'] for idx in self.graph.nodes]
        embeddings = np.array(embeddings)
        
        # 计算相似度矩阵
        sim_matrix = cosine_similarity(embeddings)
        
        # 添加相似度边
        for i in range(sim_matrix.shape[0]):
            for j in range(i+1, sim_matrix.shape[1]):
                if sim_matrix[i][j] > self.sim_threshold:
                    self.graph.add_edge(i, j, weight=sim_matrix[i][j])
    
    def deduplicate(self):
        """通过连通分量去重"""
        # 找出所有连通分量
        components = list(nx.connected_components(self.graph))
        
        unique_texts = []
        for comp in components:
            # 每个分量选择一个代表性文本
            representative = self.select_representative(comp)
            unique_texts.append(self.text_map[representative])
            
        return unique_texts
    
    def select_representative(self, node_set):
        """选择中心度最高的作为代表"""
        subgraph = self.graph.subgraph(node_set)
        centrality = nx.degree_centrality(subgraph)
        return max(centrality, key=centrality.get)

import re
import emoji
from transformers import pipeline

class QualityFilter:
    def __init__(self, max_symbol_ratio=0.1, min_length=20):
        """文本质量过滤器"""
        self.max_symbol_ratio = max_symbol_ratio  # 符号最大占比
        self.min_length = min_length  # 最小单词数
        self.toxic_classifier = pipeline(
            "text-classification", 
            model="unitary/toxic-bert"
        )
        
    def rule_based_filter(self, text):
        """
        基于规则的质量过滤
        物理意义：移除低信息密度内容
        原理：启发式规则+统计特征分析
        """
        # 规则1：删除HTML标签
        cleaned = re.sub(r'<[^>]+>', '', text)
        
        # 规则2：删除URLs
        cleaned = re.sub(r'https?://\S+', '', cleaned)
        
        # 规则3：移除连续空白
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # 规则4：计算非文字符号比例
        chars_count = len(cleaned)
        if chars_count == 0:
            return False
            
        symbol_count = sum(
            1 for char in cleaned 
            if not char.isalnum() and not char.isspace()
        )
        symbol_ratio = symbol_count / chars_count
        
        # 规则5：表情符号处理
        has_emoji = bool(emoji.get_emoji_regexp().search(cleaned))
        
        # 规则6：单词数量检查
        word_count = len(cleaned.split())
        
        # 决定是否保留
        return (
            word_count >= self.min_length and 
            symbol_ratio <= self.max_symbol_ratio and
            not has_emoji
        )
    
    def toxicity_filter(self, text, threshold=0.7):
        """
        有毒内容分类器
        物理意义：确保训练数据安全无害
        原理：预训练BERT模型微调
        """
        result = self.toxic_classifier(text, top_k=1)
        return result[0]['score'] < threshold
        
    def full_filter(self, texts):
        """完整质量过滤流程"""
        filtered_texts = []
        
        for text in texts:
            if self.rule_based_filter(text) and self.toxicity_filter(text):
                filtered_texts.append(text)
                
        return filtered_texts


from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast
import numpy as np
import faiss

class TokenizationSystem:
    def __init__(self, vocab_size=30000):
        """分词与向量化系统"""
        self.tokenizer = None
        self.vocab_size = vocab_size
        
    def train_bpe(self, corpus_files):
        """
        训练BPE分词器
        物理意义：创建子词词汇表解决OOV问题
        原理：贪心算法迭代合并高频字符对
        """
        # 初始化空白BPE分词器
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        
        # 添加预分词器 (按空格分割)
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        
        # 添加解码器 (还原原始文本)
        tokenizer.decoder = decoders.BPEDecoder()
        
        # 训练分词器
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        )
        tokenizer.train(files=corpus_files, trainer=trainer)
        
        # 封装为Hugging Face兼容格式
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            padding_side='right',
            truncation_side='right',
            model_max_length=512
        )
        self.tokenizer.add_special_tokens({
            'pad_token': '[PAD]',
            'cls_token': '[CLS]',
            'sep_token': '[SEP]',
            'unk_token': '[UNK]',
            'mask_token': '[MASK]'
        })
        
    def tokenize_batch(self, texts):
        """
        批量文本分词
        物理意义：将文本转换为模型可处理的数字序列
        返回：{
            'input_ids': 编码后的token id列表,
            'attention_mask': 注意力掩码
        }
        """
        return self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'  # 可选：'np'为NumPy数组
        )
    
    def build_faiss_index(self, embeddings):
        """
        创建FAISS向量索引
        物理意义：支持高效相似度搜索
        原理：向量量化+倒排索引
        """
        # 获取向量维度
        dim = embeddings.shape[1]
        
        # 创建量化器 (IVF：倒排文件索引)
        quantizer = faiss.IndexFlatL2(dim)
        
        # 创建IVF索引 (nlist为分区数)
        index = faiss.IndexIVFFlat(quantizer, dim, 100)
        
        # 训练索引 (需要代表性样本)
        index.train(embeddings)
        
        # 添加向量到索引
        index.add(embeddings)
        return index
    
    def semantic_search(self, query_embedding, index, top_k=5):
        """在FAISS索引中执行相似度搜索"""
        # 计算查询向量的范数
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding /= norm  # 归一化
        
        # FAISS搜索 (返回距离和索引)
        distances, indices = index.search(np.array([query_embedding]), top_k)
        return indices[0], distances[0]