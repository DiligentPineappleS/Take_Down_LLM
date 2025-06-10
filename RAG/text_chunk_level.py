import re
import json
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Any, Callable, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
from nltk.tokenize import sent_tokenize
import jieba
import spacy
from transformers import pipeline

# 初始化核心组件（实际应用中应考虑延迟加载）
jieba.initialize()
nlp = spacy.load("zh_core_web_sm")  # 中文处理模型
summarizer = pipeline("summarization", model="csebuetnlp/mT5_multilingual_XLSum")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

class OptimizedMetadataChunker:
    
    def __init__(self, 
                 base_chunk_size: int = 512,
                 max_metadata_ratio: float = 0.25,
                 content_header: str = "内容摘要",
                 metadata_header: str = "文档属性",
                 max_metadata_items: int = 4,
                 relevance_threshold: float = 0.45,
                 use_gpu: bool = False):
        """
        参数:
          base_chunk_size: 基础块大小 (默认512)
          max_metadata_ratio: 元数据最大内容占比 (0-1, 默认0.25)
          content_header: 内容摘要标题 (默认"内容摘要")
          metadata_header: 元数据标题 (默认"文档属性")
          max_metadata_items: 每个块最大元数据项数 (默认4)
          relevance_threshold: 元数据相关性阈值 (0-1, 默认0.45)
          use_gpu: 是否使用GPU加速
        """
        self.base_chunk_size = base_chunk_size
        self.max_metadata_ratio = max_metadata_ratio
        self.content_header = content_header
        self.metadata_header = metadata_header
        self.max_metadata_items = max_metadata_items
        self.relevance_threshold = relevance_threshold
        
        # 设置计算设备
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        # 加载模型到指定设备
        self.embedding_model = embedding_model.to(self.device)
        self.summarizer.device = 0 if "cuda" in self.device else -1
    
    def preprocess_metadata(self, metadata: dict) -> dict:
        """
        元数据智能预处理:
        1. 自动分类核心/扩展元数据
        2. 清理和标准化数据
        3. 提取元数据关键词
        """
        # 核心元数据标识（优先级最高）
        core_keys = {'标题', '作者', '机构', '时间', '年份', '日期', '类型', '类别', '领域', '主题', '文档ID'}
        
        # 自动检测元数据项的重要性
        processed = {
            'core': {},
            'extended': {},
            'keywords': []
        }
        
        # 提取元数据关键词
        metadata_keywords = set()
        
        for key, value in metadata.items():
            if not value or (isinstance(value, str) and not value.strip()):
                continue
                
            # 规范化键名（去除非中文字符）
            clean_key = re.sub(r'[^\u4e00-\u9fa5]', '', key)
            
            # 转换值为字符串
            if isinstance(value, list):
                value_str = "、".join(map(str, value))
            else:
                value_str = str(value)
            
            # 使用spacy提取实体
            doc = nlp(value_str)
            entities = [ent.text for ent in doc.ents]
            
            # 核心元数据检测逻辑
            if clean_key in core_keys:
                processed['core'][clean_key] = value_str
                # 提取实体作为关键词
                metadata_keywords.update(entities)
            else:
                processed['extended'][clean_key] = value_str
                
                # 提取名词和重要实体
                nouns = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
                metadata_keywords.update(nouns[:min(3, len(nouns))])
        
        # 关键词去重和排序
        if metadata_keywords:
            processed['keywords'] = sorted(metadata_keywords, key=len, reverse=True)
        
        return processed
    
    def extract_keywords(self, text: str, top_n: int = 8) -> List[str]:
        """
        使用混合方法提取关键词:
        1. TF-IDF统计
        2. 实体识别
        3. 基于嵌入的关键词聚类
        """
        # TF-IDF方法
        words = [word for word in jieba.cut(text) if len(word) > 1]
        word_counts = Counter(words)
        tfidf_words = [word for word, count in word_counts.most_common(top_n * 2)]
        
        # 实体识别
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents]
        entity_counts = Counter(entities)
        entities = [ent for ent, count in entity_counts.most_common(top_n)]
        
        # 嵌入聚类关键词
        if len(text) > 200:
            # 使用嵌入进行关键词聚类
            candidate_words = list(set(tfidf_words + entities))
            
            if candidate_words:
                # 获取嵌入向量
                word_embeddings = self.embedding_model.encode(candidate_words)
                
                # 平均文本嵌入作为聚类中心
                text_embedding = self.embedding_model.encode([text])[0]
                
                # 计算每个词与文本的相关性
                similarities = cosine_similarity([text_embedding], word_embeddings)[0]
                sorted_indices = np.argsort(similarities)[::-1]
                
                # 选择最相关的关键词
                embedding_keywords = [candidate_words[i] for i in sorted_indices[:top_n]]
            else:
                embedding_keywords = []
        else:
            embedding_keywords = tfidf_words[:top_n]
        
        # 组合三种方法的关键词
        final_keywords = list(set(tfidf_words[:top_n//2] + entities[:top_n//2] + embedding_keywords[:top_n]))
        
        # 限制关键词数量并排序
        return sorted(final_keywords, key=len, reverse=True)[:top_n]
    
    def calculate_metadata_relevance(self, 
                                    chunk_embedding: np.ndarray, 
                                    metadata_items: List[Tuple[str, str]]) -> List[Tuple[str, float]]:
        """
        基于嵌入计算元数据项与块内容的相关性
        """
        if not metadata_items:
            return []
            
        # 准备元数据文本
        metadata_texts = [f"{key}: {value}" for key, value in metadata_items]
        
        # 获取元数据嵌入
        metadata_embeddings = self.embedding_model.encode(metadata_texts)
        
        # 计算相似度
        similarities = cosine_similarity([chunk_embedding], metadata_embeddings)[0]
        
        # 返回带分数的元数据项
        return list(zip(metadata_items, similarities))
    
    def generate_content_summary(self, text: str, max_length: int = 80) -> str:
        """
        智能摘要生成:
        1. 使用微调的多语言摘要模型
        2. 自动适应长度限制
        3. 备用传统方法
        """
        # 对于长文本使用先进摘要模型
        if len(text) > 300:
            try:
                # 使用多语言摘要模型
                summary = self.summarizer(
                    text, 
                    max_length=min(200, int(max_length * 1.5)), 
                    min_length=min(30, max_length // 2),
                    do_sample=False
                )[0]['summary_text']
                
                # 后处理：确保不超过目标长度
                if len(summary) > max_length:
                    # 使用句子截断而非单词截断
                    sentences = re.split(r'[。！？]', summary)
                    truncated = ""
                    for sentence in sentences:
                        if len(truncated) + len(sentence) < max_length - 3:  # 为省略号留空间
                            truncated += sentence + "。"
                        else:
                            break
                    summary = truncated.rstrip("。") + "..." if truncated else summary[:max_length] + "..."
                
                return summary
            except Exception as e:
                print(f"摘要生成错误: {e}, 使用后备方法")
        
        # 后备方法：提取关键句
        sentences = re.split(r'[。！？]', text)
        if not sentences:
            return text[:max_length]
        
        # 简单策略：取最长/最复杂的句子
        longest_sentence = max(sentences, key=lambda s: len(s))
        
        if len(longest_sentence) > max_length:
            # 尝试找到包含最多名词的句子
            noun_counts = []
            for s in sentences:
                doc = nlp(s)
                noun_count = sum(1 for token in doc if token.pos_ in ["NOUN", "PROPN"])
                noun_counts.append((s, noun_count))
            
            if noun_counts:
                best_sentence = max(noun_counts, key=lambda x: x[1])[0]
                return best_sentence[:max_length]
            else:
                return longest_sentence[:max_length]
        else:
            return longest_sentence
    
    def hybrid_chunking(self, text: str) -> List[str]:
        """
        自适应混合分块策略:
        1. 对技术文档使用结构分块
        2. 对长文本使用分层分块
        3. 对对话使用语义分块
        4. 通用使用递归分块
        """
        # 检测文档类型
        if re.search(r'(```|#|import|function|class|接口|API)', text):
            # 技术文档: 代码/API文档
            return self._chunk_by_structure(text)
        elif len(text) > 5000:
            # 超长文本: 分层分块
            return self._hierarchical_chunking(text)
        elif re.search(r'(用户|客户|客服|咨询)|(\?|\？|：)', text):
            # 对话内容: 语义分块
            return self._semantic_chunking(text)
        else:
            # 通用文档: 递归分块
            return self._recursive_chunking(text)
    
    def _chunk_by_structure(self, text: str) -> List[str]:
        """结构感知分块 (标题/代码块)"""
        # 按主要标题分割
        chunks = []
        current_chunk = ""
        lines = text.split('\n')
        
        for line in lines:
            # 标题检测 (Markdown风格)
            if re.match(r'^(#{1,6}|={3,}|-{3,})', line):
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = line + "\n"
            # 代码块检测
            elif re.match(r'^\s*(```|~~~|代码示例|import|def|function)', line):
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # 处理过大的块
        result = []
        for chunk in chunks:
            if len(chunk) > self.base_chunk_size * 1.5:
                subchunks = self._recursive_chunking(chunk)
                result.extend(subchunks)
            else:
                result.append(chunk)
        
        return result
    
    def _hierarchical_chunking(self, text: str) -> List[str]:
        """分层分块: 全局结构 + 局部语义"""
        # 第一层: 全局分块 (1024字符)
        global_chunks = self._fixed_chunking(text, chunk_size=1024, overlap=0.1)
        
        result = []
        for global_chunk in global_chunks:
            # 第二层: 基于内容类型分块
            if re.search(r'(#{1,6}|import|function)', global_chunk):
                subchunks = self._chunk_by_structure(global_chunk)
            else:
                subchunks = self._recursive_chunking(global_chunk)
            
            result.extend(subchunks)
        
        return result
    
    def _semantic_chunking(self, text: str) -> List[str]:
        """语义分块: 句子嵌入聚类"""
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            return [text]
        
        # 获取句子嵌入
        embeddings = self.embedding_model.encode(sentences)
        
        chunks = []
        current_chunk = []
        current_emb = None
        
        for i, sentence in enumerate(sentences):
            # 第一个句子
            if i == 0:
                current_chunk.append(sentence)
                current_emb = embeddings[i]
                continue
            
            # 计算当前句与当前块的相似度
            sim = cosine_similarity([current_emb], [embeddings[i]])[0][0]
            
            # 合并条件
            if sim >= 0.7 or len(" ".join(current_chunk)) + len(sentence) < self.base_chunk_size:
                current_chunk.append(sentence)
                # 更新平均嵌入
                current_emb = (current_emb * len(current_chunk) + embeddings[i]) / (len(current_chunk) + 1)
            else:
                # 保存当前块
                chunks.append(" ".join(current_chunk))
                # 开始新块
                current_chunk = [sentence]
                current_emb = embeddings[i]
        
        # 添加最后一个块
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _recursive_chunking(self, text: str) -> List[str]:
        """递归分块策略"""
        separators = ['\n\n', '\n', '。', '；', '？', '！', '，']
        
        # 尝试在最大长度内找到合适的分隔符
        if len(text) <= self.base_chunk_size:
            return [text]
        
        # 寻找最合适的分隔符
        for sep in separators:
            if sep in text:
                # 找到最近的分离点
                parts = text.split(sep)
                chunks = []
                current = ""
                
                for part in parts:
                    part_with_sep = part + sep
                    if len(current) + len(part_with_sep) <= self.base_chunk_size:
                        current += part_with_sep
                    else:
                        if current:
                            chunks.append(current)
                        current = part_with_sep
                
                if current:
                    chunks.append(current)
                
                # 检查是否需要进一步分割
                final_chunks = []
                for chunk in chunks:
                    if len(chunk) > self.base_chunk_size * 1.2:
                        final_chunks.extend(self._recursive_chunking(chunk))
                    else:
                        final_chunks.append(chunk)
                
                return final_chunks
        
        # 没有找到合适分隔符: 使用固定分块
        return self._fixed_chunking(text, chunk_size=self.base_chunk_size)
    
    def _fixed_chunking(self, text: str, chunk_size: int, overlap: float = 0.1) -> List[str]:
        """固定长度分块"""
        chunks = []
        start = 0
        step = max(1, int(chunk_size * (1 - overlap)))
        total_length = len(text)
        
        while start < total_length:
            end = min(start + chunk_size, total_length)
            chunks.append(text[start:end])
            
            if end == total_length:
                break
                
            start += step
        
        return chunks
    
    def optimize_chunk(self, chunk_text: str, metadata: dict, chunk_id: int, total_chunks: int) -> Dict[str, Any]:
        """
        优化单个分块:
        1. 生成内容摘要
        2. 提取关键词
        3. 选择最相关的元数据
        4. 构建块间关系
        """
        # 0. 预处理文本
        clean_text = re.sub(r'\s+', ' ', chunk_text).strip()
        
        # 1. 生成内容摘要
        summary_length = min(120, max(50, int(len(clean_text) * 0.25)))
        summary = self.generate_content_summary(clean_text, summary_length)
        
        # 2. 提取内容关键词
        content_keywords = self.extract_keywords(clean_text)
        
        # 3. 获取块嵌入（用于相关性计算）
        chunk_embedding = self.embedding_model.encode([clean_text])[0]
        
        # 4. 合并元数据关键词与内容关键词
        all_keywords = list(set(content_keywords + metadata.get('keywords', [])))
        
        # 5. 准备元数据项（核心 + 扩展）
        candidate_metadata = []
        candidate_metadata.extend([(k, v) for k, v in metadata['core'].items()])
        candidate_metadata.extend([(k, v) for k, v in metadata['extended'].items()])
        
        # 6. 基于相关性选择元数据
        if candidate_metadata:
            ranked_metadata = self.calculate_metadata_relevance(chunk_embedding, candidate_metadata)
            
            # 选择高于阈值的项目
            selected = [
                (key, value) for (key, value), score in ranked_metadata 
                if score >= self.relevance_threshold
            ][:self.max_metadata_items]
            
            # 如果相关性选择不足，补充核心元数据
            if not selected and metadata['core']:
                selected = [(k, v) for k, v in metadata['core'].items()][:min(2, self.max_metadata_items)]
        else:
            selected = []
        
        # 7. 计算元数据预算
        metadata_length = sum(len(f"{key}{value}") for key, value in selected)
        max_metadata_length = int(len(clean_text) * self.max_metadata_ratio)
        
        # 8. 修剪元数据以符合预算
        final_metadata = {}
        for key, value in selected:
            if metadata_length <= max_metadata_length:
                final_metadata[key] = value
            else:
                # 截断过长的元数据值
                if len(value) > 80:
                    final_metadata[key] = value[:80] + "..."
                    break
                else:
                    final_metadata[key] = value
                    metadata_length += len(value)
        
        # 9. 构建块关系
        relations = []
        if chunk_id > 0:
            relations.append({"type": "前序块", "chunk_id": chunk_id - 1})
        if chunk_id < total_chunks - 1:
            relations.append({"type": "后序块", "chunk_id": chunk_id + 1})
        
        # 10. 返回结构化块
        return {
            "chunk_id": chunk_id,
            "content": clean_text,
            "summary": summary,
            "metadata": final_metadata,
            "keywords": all_keywords,
            "relations": relations,
            "embedding": chunk_embedding.tolist() if "cuda" not in self.device else None
        }
    
    def __call__(self, text: str, metadata: dict) -> List[Dict[str, Any]]:
        """
        主处理流程
        """
        # 1. 预处理元数据
        processed_metadata = self.preprocess_metadata(metadata)
        
        # 2. 基础分块
        base_chunks = self.hybrid_chunking(text)
        
        # 3. 优化每个分块
        enriched_chunks = []
        total_chunks = len(base_chunks)
        
        for i, chunk_text in enumerate(base_chunks):
            if len(chunk_text.strip()) < 20:  # 过滤过小的块
                continue
                
            optimized_chunk = self.optimize_chunk(
                chunk_text, 
                processed_metadata,
                i,
                total_chunks
            )
            enriched_chunks.append(optimized_chunk)
        
        # 4. 添加文档级信息（在第一个块中）
        if enriched_chunks:
            # 添加文档摘要
            doc_summary = self.generate_content_summary(text, 150)
            
            # 添加核心元数据到第一个块
            enriched_chunks[0]['doc_metadata'] = {
                k: v for k, v in processed_metadata['core'].items()
            }
            enriched_chunks[0]['doc_summary'] = doc_summary
            
            # 添加文档级关键词
            enriched_chunks[0]['doc_keywords'] = processed_metadata.get('keywords', [])
        
        return enriched_chunks