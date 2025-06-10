import re
import json
import uuid
import os
import chromadb
import numpy as np
from datetime import datetime
from typing import List, Dict, Callable, Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import jieba
import jieba.analyse
import spacy
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import torch
from transformers import pipeline
import pdfplumber
import docx
import markdown
from bs4 import BeautifulSoup
from collections import Counter
import pytz

# 多格式文档加载器:自动生成文本文档标签元数据
class MultiFormatDocumentLoader:    
    def __init__(self):
        pass
    
    def load(self, file_path: str):
        """加载文档并自动生成元数据"""
        file_path = os.path.abspath(file_path)
        filename = os.path.basename(file_path)
        file_ext = os.path.splitext(filename)[1].lower()
        
        # 基础元数据
        metadata = {
            "source_file": file_path,
            "filename": filename,
            "file_type": file_ext[1:],  # 去除点号
            "load_time": datetime.now(pytz.utc).isoformat(),
            "file_size": os.path.getsize(file_path)
        }
        
        # 读取文件内容
        if file_ext == '.txt':
            content = self._read_txt(file_path)
        elif file_ext == '.pdf':
            content = self._read_pdf(file_path)
        elif file_ext == '.docx':
            content = self._read_docx(file_path)
        elif file_ext == '.md':
            content = self._read_markdown(file_path)
        elif file_ext == '.html' or file_ext == '.htm':
            content = self._read_html(file_path)
        else:
            # 尝试通用文本读取
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except:
                content = ""
        
        # 自动提取标题和作者
        metadata.update(self.auto_extract_metadata(content))
        
        return {
            "content": content,
            "metadata": metadata
        }
    ### 从内容中自动提取元数据
    def auto_extract_metadata(self, content: str) -> Dict[str, str]:
        metadata = {}
        
        # 提取标题
        title_patterns = [
            r'^#+\s+(.+)',  # Markdown标题
            r'<title>(.*?)<\/title>',  # HTML标题
            r'标题[:：]\s*(.+)',  # 中文标题
            r'Title[:：]\s*(.+)'   # 英文标题
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
            if match and len(match.group(1).strip()) > 3:
                metadata['title'] = match.group(1).strip()
                break
        
        # 提取作者
        author_patterns = [
            r'作者[:：]\s*(.+)',
            r'Author[:：]\s*(.+)',
            r'By[:：]\s*(.+)',
            r'[\n](.*?)[\n](责任编辑|编审)'  # 文档末尾签名
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
            if match and len(match.group(1).strip()) > 2:
                metadata['author'] = match.group(1).strip()
                break
        
        # 提取日期
        date_patterns = [
            r'日期[:：]\s*(\d{4}-\d{2}-\d{2})',
            r'Date[:：]\s*(\d{4}-\d{2}-\d{2})',
            r'(\d{4}年\d{1,2}月\d{1,2}日)',
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{4}/\d{1,2}/\d{1,2})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, content)
            if match:
                metadata['doc_date'] = match.group(1)
                break
        
        # 提取文档类型:关键词匹配；具体需要根据内容设计，有些可以通过标题来判断
        doc_type_keywords = {
            "合同": ["合同", "协议", "agreement", "contract"],
            "报告": ["报告", "分析", "report", "analysis"],
            "手册": ["手册", "指南", "manual", "guide"],
            "研究": ["研究", "论文", "survey", "study", "论文"]
        }
        
        for doc_type, keywords in doc_type_keywords.items():
            if any(kw in content for kw in keywords):
                metadata['doc_type'] = doc_type
                break
        
        return metadata
    
    def _read_txt(self, file_path: str) -> str:
        """读取文本文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _read_pdf(self, file_path: str) -> str:
        """读取PDF文件"""
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _read_docx(self, file_path: str) -> str:
        """读取Word文档"""
        doc = docx.Document(file_path)
        text = []
        for para in doc.paragraphs:
            text.append(para.text)
        return "\n".join(text)
    
    def _read_markdown(self, file_path: str) -> str:
        """读取Markdown文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            md_text = f.read()
            return markdown.markdown(md_text)  # 转换为HTML后再提取文本
    
    def _read_html(self, file_path: str) -> str:
        """读取HTML文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            # 提取主体文本
            for element in soup(["script", "style"]):
                element.decompose()
            return soup.get_text()





# 初始化核心组件
jieba.initialize()
nlp = spacy.load("zh_core_web_sm")
#关键词提取器
class EnhancedKeywordExtractor:
    def __init__(self, use_embedding=True):
        self.use_embedding = use_embedding
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 加载预训练模型
        if use_embedding:
            self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2').to(self.device)
    def extract(self, text: str, top_n: int = 5):
        if len(text) < 30:
            return self.short_text_keywords(text, top_n)
        # 多策略提取
        methods = [self.textrank_extract,self.pos_based_extract,self.named_entity_extract]
        if self.use_embedding:
            methods.append(self.embedding_cluster_extract)
        keyword_sets = [method(text, top_n) for method in methods]
        # 融合和去重
        return self.fuse_keywords(keyword_sets, top_n)
    ### TextRank算法提取关键词
    def textrank_extract(self, text: str, top_n: int):
        try:
            return jieba.analyse.textrank(text, topK=top_n*2, withWeight=False, allowPOS=('n', 'nr', 'ns', 'v', 'a'))
        except:
            return []
    ### 基于词性分布提取关键词
    def pos_based_extract(self, text: str, top_n: int):
        words = jieba.posseg.cut(text)
        candidate_keywords = []
        for word, flag in words:
            # 重点考虑名词、动词和形容词
            if flag.startswith('n') or flag.startswith('v') or flag == 'a':
                candidate_keywords.append(word)
        # 基于词频统计
        counter = Counter(candidate_keywords)
        return [kw for kw, count in counter.most_common(top_n*2)]
    ### 命名实体提取
    def named_entity_extract(self, text: str, top_n: int):
        doc = nlp(text)
        entities = set()
        for ent in doc.ents:
            # 过滤长度合适的实体
            if 1 < len(ent.text) < 10:
                entities.add(ent.text)
        return list(entities)[:top_n]
    ### 基于嵌入聚类提取关键词
    def embedding_cluster_extract(self, text: str, top_n: int):
        
        if not hasattr(self, 'embedding_model'):
            return []
        # 分句处理
        sentences = [sent for sent in re.split(r'[。！？；]+', text) if len(sent) > 10]
        if not sentences:
            return []
        # 生成句子嵌入
        embeddings = self.embedding_model.encode(sentences)
        # 自适应聚类数量
        k = min(5, max(1, len(sentences)//3))
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(embeddings)
        # 获取每个聚类的中心句
        centroids = kmeans.cluster_centers_
        
        # 为每个聚类寻找代表性词语
        cluster_keywords = []
        for i in range(k):
            # 找到最近的中心句
            distances = np.linalg.norm(embeddings - centroids[i], axis=1)
            closest_idx = np.argmin(distances)
            closest_sentence = sentences[closest_idx]
            
            # 提取关键词
            cluster_keywords.extend(self._textrank_extract(closest_sentence, top_n=2))
        
        return cluster_keywords[:top_n]
    ### 融合不同方法提取的关键词
    def fuse_keywords(self, keyword_sets: List[List[str]], top_n: int):
        
        # 计分机制：出现次数越多排名越高
        keyword_scores = Counter()
        for kw_set in keyword_sets:
            for i, kw in enumerate(kw_set):
                # 越靠前的词权重越高
                keyword_scores[kw] += (len(kw_set) - i) * 0.5
        
        # 按分数排序并去重
        sorted_keywords = [kw for kw, score in keyword_scores.most_common()]
        unique_keywords = []
        for kw in sorted_keywords:
            # 去重：包含关系处理（如"人工智能"和"智能"）
            if not any(existing != kw and kw in existing for existing in unique_keywords):
                unique_keywords.append(kw)
        
        return unique_keywords[:top_n]
    ### 短文本关键词提取方法
    def short_text_keywords(self, text: str, top_n: int):
        """"""
        # 使用简单词性分析
        words = jieba.posseg.cut(text)
        keywords = []
        for word, flag in words:
            if flag.startswith('n'):
                keywords.append(word)
        
        # 添加高频名词
        counter = Counter(keywords)
        return [kw for kw, count in counter.most_common(top_n)]


# 分级分块引擎
class HierarchicalChunker:
    def __init__(self, min_sentence_length: int = 20,max_sentence_length: int = 100,chunking_thresholds: Dict[str, int] = {"topic": 0.6,"paragraph": 0.4,"semantic": 0.3}):
        # 配置参数
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length
        self.chunking_thresholds = chunking_thresholds
        
        # 加载模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2').to(self.device)
        self.keyword_extractor = EnhancedKeywordExtractor()
        try:
            self.summarizer = pipeline("summarization", model="csebuetnlp/mT5_multilingual_XLSum")
            if "cuda" in self.device:
                self.summarizer.device = 0
        except:
            self.summarizer = self._fallback_summary
            print("使用后备摘要方法")
    
    def process_document(self, text: str):
        """处理文档并返回分级分块结构"""
        # 文档级处理
        doc_summary = self._generate_summary(text, max_length=120)
        doc_keywords = self.keyword_extractor.extract(text, top_n=10)
        doc_entities = self._extract_entities(text)
        doc_structure = self._analyze_structure(text)
        
        # 主题级分块
        topics = self.detect_topics(text)
        
        return {
            "level": "document",
            "content": text,
            "summary": doc_summary,
            "keywords": doc_keywords,
            "entities": doc_entities,
            "structure": doc_structure,
            "chunks": topics
        }
    
    def detect_topics(self, text: str):
        """主题级分块 - 识别文档主要主题划分"""
        paragraphs = self._split_paragraphs(text)
        para_embeddings = self.embedding_model.encode(paragraphs)
        
        k = max(2, min(10, len(paragraphs) // 5))
        kmeans = KMeans(n_clusters=k, random_state=42)
        topic_labels = kmeans.fit_predict(para_embeddings)
        
        topics = []
        for topic_id in set(topic_labels):
            topic_paras = [paragraphs[i] for i, label in enumerate(topic_labels) if label == topic_id]
            topic_text = " ".join(topic_paras)
            
            topic_keywords = self.keyword_extractor.extract(topic_text, top_n=5)
            topic_summary = self._generate_summary(topic_text, max_length=80)
            topic_entities = self._extract_entities(topic_text)
            
            topics.append({
                "level": "topic",
                "id": f"topic_{topic_id}",
                "summary": topic_summary,
                "keywords": topic_keywords,
                "entities": topic_entities,
                "chunks": [self._process_paragraph(para, f"topic_{topic_id}") for para in topic_paras]
            })
        
        return topics
    
    def _process_paragraph(self, paragraph: str, topic_id: str):
        """段落级处理"""
        semantic_units = self._split_semantic_units(paragraph)
        
        return {
            "level": "paragraph",
            "content": paragraph,
            "summary": self._generate_summary(paragraph, max_length=60),
            "keywords": self.keyword_extractor.extract(paragraph, top_n=3),
            "entities": self._extract_entities(paragraph),
            "parent_id": topic_id,
            "chunks": semantic_units
        }
    
    def _split_semantic_units(self, text: str):
        """语义级分块 - 基于语义边界划分"""
        sentences = self._split_sentences(text)
        if len(sentences) < 2:
            return [{
                "level": "semantic",
                "content": text,
                "keywords": self.keyword_extractor.extract(text, top_n=3),
                "entities": self._extract_entities(text),
                "chunks": [self._process_sentence(text)]
            }]
        
        sentence_embeddings = self.embedding_model.encode(sentences)
        semantic_units = []
        current_unit = []
        current_emb = None
        
        for i, sentence in enumerate(sentences):
            sentence_emb = sentence_embeddings[i]
            
            if not current_unit:
                current_unit.append(sentence)
                current_emb = sentence_emb
                continue
            
            similarity = cosine_similarity([current_emb], [sentence_emb])[0][0]
            
            if similarity > self.chunking_thresholds["semantic"]:
                current_unit.append(sentence)
                current_emb = (current_emb * len(current_unit) + sentence_emb) / (len(current_unit) + 1)
            else:
                unit_text = " ".join(current_unit)
                semantic_units.append({
                    "level": "semantic",
                    "content": unit_text,
                    "keywords": self.keyword_extractor.extract(unit_text, top_n=3),
                    "entities": self._extract_entities(unit_text),
                    "chunks": [self._process_sentence(s) for s in current_unit]
                })
                current_unit = [sentence]
                current_emb = sentence_emb
        
        if current_unit:
            unit_text = " ".join(current_unit)
            semantic_units.append({
                "level": "semantic",
                "content": unit_text,
                "keywords": self.keyword_extractor.extract(unit_text, top_n=3),
                "entities": self._extract_entities(unit_text),
                "chunks": [self._process_sentence(s) for s in current_unit]
            })
        
        return semantic_units
    
    def _process_sentence(self, sentence: str):
        """语句级处理"""
        return {
            "level": "sentence",
            "content": sentence,
            "keywords": self.keyword_extractor.extract(sentence, top_n=2),
            "entities": self._extract_entities(sentence),
            "tokens": [token.text for token in nlp(sentence)]
        }
    
    def to_flat_chunks(self, document: Dict[str, Any], include_levels: list = ["paragraph", "semantic"]):
        """将分级结构转换为扁平块列表，用于向量存储"""
        chunks = []
        
        def extract_chunks(node):
            if node["level"] in include_levels:
                # 为每个块创建唯一ID
                chunk_id = node.get("id", str(uuid.uuid4())[:8])
                parent_id = node.get("parent_id", "")
                
                chunks.append({
                    "id": chunk_id,
                    "content": node["content"],
                    "summary": node.get("summary", ""),
                    "keywords": node.get("keywords", []),
                    "entities": node.get("entities", []),
                    "level": node["level"],
                    "parent_id": parent_id
                })
            
            if "chunks" in node:
                for child in node["chunks"]:
                    extract_chunks(child)
        
        extract_chunks(document)
        return chunks

    # ----------------- 辅助方法 -----------------
    def _split_paragraphs(self, text: str):
        return [para.strip() for para in re.split(r'\n\n+', text) if para.strip()]
    
    def _split_sentences(self, text: str):
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents 
                if self.min_sentence_length < len(sent.text) < self.max_sentence_length]
    
    def _extract_entities(self, text: str):
        if not text: return []
        doc = nlp(text)
        entities = list(set([ent.text for ent in doc.ents]))
        return entities
    
    def _generate_summary(self, text: str, max_length: int = 80) -> str:
        if len(text) < max_length * 2:
            return text[:max_length]
        
        try:
            if callable(self.summarizer) and not hasattr(self.summarizer, 'model'):
                return self.summarizer(text, max_length)
            
            return self.summarizer(
                text, 
                max_length=max_length, 
                min_length=min(30, max_length // 2),
                do_sample=False
            )[0]['summary_text']
        except Exception as e:
            print(f"摘要生成错误: {e}")
            return self._fallback_summary(text, max_length)
    
    def _fallback_summary(self, text: str, max_length: int) -> str:
        """后备摘要方法"""
        sentences = re.split(r'[。！？]', text)
        if len(sentences) > 1:
            return sentences[0] + "..." + sentences[-1]
        elif sentences:
            return sentences[0]
        return text[:max_length]
    
    def _analyze_structure(self, text: str):
        sections = re.findall(r'#+\s*(.+?)\n', text)  # Markdown标题
        para_count = len(self._split_paragraphs(text))
        sent_count = len(self._split_sentences(text))
        
        return {
            "sections": sections,
            "paragraph_count": para_count,
            "sentence_count": sent_count,
            "avg_sentence_length": sum(len(sent) for sent in self._split_sentences(text)) / max(1, sent_count)
        }


class ChromaVectorStore:
    """Chroma向量存储管理器"""
    
    def __init__(self, 
                 chroma_path: str = "./chroma_db",
                 collection_name: str = "hierarchical_chunks",
                 embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 use_gpu: bool = True):
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        
        # 初始化Chroma客户端
        self.client = chromadb.PersistentClient(path=chroma_path)
        
        # 初始化嵌入模型
        self.embedding_model = SentenceTransformer(embedding_model_name)
        if use_gpu and torch.cuda.is_available():
            self.embedding_model = self.embedding_model.to("cuda")
        
        # 获取或创建集合
        self.collection = self._get_or_create_collection()
    
    def _get_or_create_collection(self):
        """获取或创建Chroma集合"""
        try:
            collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=embedding_functions.DefaultEmbeddingFunction()
            )
            print(f"使用现有集合: {self.collection_name}")
            return collection
        except:
            print(f"创建新集合: {self.collection_name}")
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=embedding_functions.DefaultEmbeddingFunction()
            )
    
    def store_document(self, document_id: str, flat_chunks: List[Dict], metadata: dict = {}):
        """将文档的扁平块存储到Chroma"""
        if not flat_chunks:
            print("错误: 没有可存储的块")
            return 0
        
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk in flat_chunks:
            # 为嵌入生成文本
            embedding_text = f"{chunk['summary']} | {' '.join(chunk['keywords'])} | {chunk['content'][:200]}"
            
            # 生成嵌入向量
            embedding = self.embedding_model.encode(embedding_text).tolist()
            
            # 准备元数据
            chunk_metadata = {
                "document_id": document_id,
                "chunk_id": chunk["id"],
                "level": chunk["level"],
                "parent_id": chunk.get("parent_id", ""),
                "keywords": json.dumps(chunk["keywords"], ensure_ascii=False),
                "entities": json.dumps(chunk["entities"], ensure_ascii=False),
                "timestamp": datetime.now(pytz.utc).isoformat(),
                **metadata
            }
            
            ids.append(chunk["id"])
            embeddings.append(embedding)
            documents.append(chunk["content"][:1000])  # 只存储内容前1000字符
            metadatas.append(chunk_metadata)
        
        # 存储到Chroma
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"已存储文档 {document_id} 的 {len(flat_chunks)} 个块")
        return len(flat_chunks)
    
    def query(self, query_text: str, n_results: int = 5, 
              where: dict = None, include: list = ["documents", "metadatas", "distances"]):
        """查询向量数据库"""
        # 生成查询嵌入
        query_embedding = self.embedding_model.encode(query_text).tolist()
        
        # 执行查询
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=include
        )
        
        # 格式化结果
        formatted_results = []
        for i in range(n_results):
            try:
                content = results["documents"][0][i]
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                
                formatted_results.append({
                    "content": content,
                    "metadata": metadata,
                    "similarity": 1 - distance,  # 距离转换为相似度
                    "source_document": metadata.get("document_id", "")
                })
            except IndexError:
                break
        
        return formatted_results
    
    def search_by_metadata(self, metadata_filter: dict, n_results: int = 5):
        """根据元数据过滤条件进行搜索"""
        return self.collection.get(
            where=metadata_filter,
            limit=n_results
        )
    
    def delete_document(self, document_id: str):
        """删除文档的所有块"""
        document_chunks = self.collection.get(
            where={"document_id": document_id}
        )
        
        if document_chunks["ids"]:
            self.collection.delete(ids=document_chunks["ids"])
            return len(document_chunks["ids"])
        return 0
    
    def update_document(self, document_id: str, flat_chunks: List[Dict], metadata: dict = {}):
        """更新文档内容 - 删除旧块并存储新块"""
        self.delete_document(document_id)
        return self.store_document(document_id, flat_chunks, metadata)
    
    def get_collection_stats(self):
        """获取集合统计信息"""
        stats = self.collection.count()
        print(f"集合包含 {stats} 个块")
        return stats


class RAGSystem:
    
    def __init__(self, chroma_path: str = "./chroma_db",collection_name: str = "hierarchical_rag",min_sentence_length: int = 128,max_sentence_length: int = 512):
        # 初始化组件
        self.chunker = HierarchicalChunker(min_sentence_length=min_sentence_length,max_sentence_length=max_sentence_length)
        self.vector_store = ChromaVectorStore(chroma_path=chroma_path,collection_name=collection_name)
        self.document_loader = MultiFormatDocumentLoader()
    
    def ingest_document(self, file_path: str, custom_metadata: dict = None):
        """从文件系统加载并处理文档"""
        # 加载文档
        doc_data = self.document_loader.load(file_path)
        content = doc_data["content"]
        auto_metadata = doc_data["metadata"]
        
        # 合并元数据
        metadata = auto_metadata.copy()
        if custom_metadata:
            metadata.update(custom_metadata)
        
        # 生成文档ID
        document_id = metadata.get("document_id", f"doc_{uuid.uuid4().hex[:8]}")
        metadata["document_id"] = document_id
        
        # 分级分块处理
        hierarchical_doc = self.chunker.process_document(content)
        
        # 转换为扁平块
        flat_chunks = self.chunker.to_flat_chunks(hierarchical_doc)
        
        # 存储到向量库
        self.vector_store.store_document(document_id, flat_chunks, metadata)
        
        return {"document_id": document_id,"hierarchical": hierarchical_doc,"flat_chunks": flat_chunks,"metadata": metadata}
    
    def ingest_folder(self, folder_path: str, file_types: list = None, custom_metadata: dict = None):
        """批量处理整个文件夹中的文档"""
        processed = []
        skipped = []
        
        # 支持的扩展名映射
        supported_ext = {
            '.txt': '文本文件',
            '.pdf': 'PDF文档',
            '.docx': 'Word文档',
            '.md': 'Markdown文档',
            '.html': 'HTML文档',
            '.htm': 'HTML文档'
        }
        
        # 如果没有指定文件类型，则处理所有支持的文件
        if not file_types:
            file_types = list(supported_ext.keys())
        
        # 扫描文件夹
        for root, _, files in os.walk(folder_path):
            for filename in files:
                file_ext = os.path.splitext(filename)[1].lower()
                
                # 检查文件类型
                if file_ext in file_types:
                    file_path = os.path.join(root, filename)
                    try:
                        result = self.ingest_document(file_path, custom_metadata)
                        processed.append({
                            "file_path": file_path,
                            "document_id": result["document_id"]
                        })
                    except Exception as e:
                        skipped.append({
                            "file_path": file_path,
                            "error": str(e)
                        })
                else:
                    skipped.append({
                        "file_path": os.path.join(root, filename),
                        "reason": f"不支持的格式: {file_ext}"
                    })
        
        return {
            "processed": processed,
            "skipped": skipped
        }
    
    def retrieve(self, query: str, n_results: int = 5, metadata_filter: dict = None):
        """检索相关块"""
        return self.vector_store.query(query, n_results=n_results, where=metadata_filter)
    
    def rag_generate(self, query: str, llm_function: Callable, n_contexts: int = 3, metadata_filter: dict = None):
        """完整的RAG流程"""
        # 检索相关上下文
        contexts = self.retrieve(query, n_results=n_contexts, metadata_filter=metadata_filter)
        
        # 构建提示
        prompt = "基于以下上下文回答用户问题:\n\n"
        for i, ctx in enumerate(contexts):
            source_meta = ctx['metadata']
            source_info = f"{source_meta.get('title', '无标题')} ({source_meta.get('filename', '')})"
            
            prompt += f"### 上下文 {i+1} [相似度: {ctx['similarity']:.2f}, 来源: {source_info}]\n"
            prompt += f"{ctx['content'][:300]}...\n\n"
        
        prompt += f"### 用户问题: {query}\n\n回答:"
        
        # 调用LLM生成回答
        response = llm_function(prompt)
        
        return {
            "query": query,
            "contexts": contexts,
            "prompt": prompt,
            "response": response
        }
    
    def delete_document(self, document_id: str):
        """删除文档"""
        self.vector_store.delete_document(document_id)
        return {"document_id": document_id, "deleted": True}
    
    def update_document(self, document_id: str, new_file_path: str, custom_metadata: dict = None):
        """从文件更新文档内容"""
        # 加载文档
        doc_data = self.document_loader.load(new_file_path)
        content = doc_data["content"]
        auto_metadata = doc_data["metadata"]
        
        # 合并元数据
        metadata = auto_metadata.copy()
        if custom_metadata:
            metadata.update(custom_metadata)
        metadata["document_id"] = document_id
        
        # 删除旧文档
        self.vector_store.delete_document(document_id)
        
        # 处理新内容
        hierarchical_doc = self.chunker.process_document(content)
        flat_chunks = self.chunker.to_flat_chunks(hierarchical_doc)
        
        # 存储新文档
        self.vector_store.store_document(document_id, flat_chunks, metadata)
        
        return {
            "document_id": document_id,
            "updated": True
        }
    
    def get_document_metadata(self, document_id: str):
        """获取文档元数据"""
        chunks = self.vector_store.collection.get(
            where={"document_id": document_id},
            include=["metadatas"]
        )
        
        if chunks.get("metadatas"):
            # 返回第一个块的元数据 (文档级元数据在所有块中相同)
            return chunks["metadatas"][0]
        
        return None