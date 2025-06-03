import re
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

class BaseTokenizer:
    """基础分词器类，包含预处理功能和公共方法"""
    def __init__(self):
        # 特殊标记
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        self.vocab = set()
    
    def preprocess(self, text: str) -> str:
        """文本预处理：小写化、去除多余空格、特殊字符处理"""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)  # 合并多个空格
        text = text.strip()
        return text
    
    def build_vocab(self, tokens: List[str]):
        """根据分词结果构建词表"""
        self.vocab = set(tokens)
        # 添加特殊标记
        self.vocab.add(self.unk_token)
        self.vocab.add(self.pad_token)
    
    def token_to_id(self, token: str) -> int:
        """将token转换为ID"""
        if token not in self.vocab:
            return self.vocab.index(self.unk_token) if self.unk_token in self.vocab else -1
        return list(self.vocab).index(token)
    
    def tokenize(self, text: str) -> List[str]:
        """分词方法（需要在子类实现）"""
        raise NotImplementedError("This method should be implemented in subclasses.")


class WordTokenizer(BaseTokenizer):
    """基于词的分词器"""
    def tokenize(self, text: str) -> List[str]:
        # 预处理文本
        text = self.preprocess(text)
        # 使用正则表达式按空格和标点分词
        tokens = re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)
        return tokens


class CharTokenizer(BaseTokenizer):
    """基于字符的分词器"""
    def tokenize(self, text: str) -> List[str]:
        # 预处理文本
        text = self.preprocess(text)
        # 拆分为单个字符
        return list(text)


class BPETokenizer(BaseTokenizer):
    """BPE分词器实现（Byte-Pair Encoding）"""
    def __init__(self, vocab_size: int = 10000):
        super().__init__()
        self.vocab_size = vocab_size
        self.merges: Dict[Tuple[str, str], str] = {}  # 存储合并规则
        self.word_end = "</w>"  # 词尾标记
    
    def _get_stats(self, vocab: Dict[str, int]) -> Counter:
        """统计相邻符号对出现频率"""
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] += freq
        return pairs

    def _merge_vocab(self, pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
        """合并指定的符号对并更新词汇表"""
        new_vocab = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        for word, freq in vocab.items():
            new_word = p.sub(''.join(pair), word)
            new_vocab[new_word] = freq
        
        return new_vocab

    def train(self, corpus: List[str]):
        """训练BPE分词器"""
        # 初始词汇统计
        vocab = Counter()
        for text in corpus:
            text = self.preprocess(text)
            # 按空格和标点初步分割
            words = re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)
            for word in words:
                # 在字符间添加空格，并在词尾添加结束标记
                encoded_word = " ".join(list(word)) + f" {self.word_end}"
                vocab[encoded_word] += 1
        
        # 基础词表：所有字符 + 特殊标记
        base_chars = set(char for word in vocab for char in word.replace(" ", "") if char != self.word_end)
        self.vocab = set(base_chars) | {self.unk_token, self.pad_token, self.word_end}
        
        # 迭代合并直到达到目标词表大小
        for i in range(self.vocab_size - len(self.vocab)):
            pairs = self._get_stats(vocab)
            if not pairs:
                break
                
            # 选择最高频的符号对
            best_pair = max(pairs, key=pairs.get)
            self.merges[best_pair] = "".join(best_pair)
            
            # 更新词表和合并规则
            vocab = self._merge_vocab(best_pair, vocab)
            
            # 更新vocab集合
            self.vocab.add("".join(best_pair))

    def tokenize(self, text: str) -> List[str]:
        """应用BPE分词"""
        if not self.merges:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        text = self.preprocess(text)
        # 按空格和标点初步分割
        words = re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)
        tokens = []
        
        for word in words:
            # 在字符间添加空格，并在词尾添加结束标记
            current = " ".join(list(word)) + f" {self.word_end}"
            current = current.split()
            
            # 应用合并规则
            while len(current) > 1:
                # 查找最长的可合并对
                merge_candidate = None
                max_length = 0
                
                for pair, merged in self.merges.items():
                    for i in range(len(current) - 1):
                        if (current[i], current[i+1]) == pair:
                            # 选择最长的可合并对
                            if len(merged) > max_length:
                                max_length = len(merged)
                                merge_candidate = (i, merged)
                
                if merge_candidate is None:
                    break
                
                i, merged = merge_candidate
                # 合并符号对
                current = current[:i] + [merged] + current[i+2:]
            
            # 移除词尾标记（如果有）
            current = [tok for tok in current if tok != self.word_end]
            tokens.extend(current)
        
        return tokens


class WordPieceTokenizer(BaseTokenizer):
    """WordPiece分词器实现"""
    def __init__(self, vocab_size: int = 10000):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_token_length = 20  # 单词最大长度限制
        self.prefix = "##"  # 子词前缀标记
    
    def _compute_likelihood(self, word_counts: Dict[str, int], token: str) -> float:
        """计算token的似然分数（简化版）"""
        if token not in word_counts:
            return 0.0
        # 简单实现：使用出现频率作为分数
        return word_counts[token] / sum(word_counts.values())
    
    def train(self, corpus: List[str]):
        """训练WordPiece分词器"""
        # 初始化词表
        char_counts = Counter()
        word_counts = Counter()
        
        # 统计字符和单词出现频率
        for text in corpus:
            text = self.preprocess(text)
            # 按空格和标点初步分割
            words = re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)
            for word in words:
                # 过滤过长的单词
                if len(word) > self.max_token_length:
                    continue
                
                word_counts[word] += 1
                for char in word:
                    char_counts[char] += 1
        
        # 初始词表：所有字符
        self.vocab = set(char for char, count in char_counts.items() if count > 1)
        self.vocab.update([self.unk_token, self.pad_token])
        
        # 添加高频单词
        for word, count in word_counts.items():
            if count > 10 and len(word) <= self.max_token_length:  # 设置阈值
                self.vocab.add(word)
        
        # 迭代学习子词
        while len(self.vocab) < self.vocab_size:
            subword_candidates = set()
            
            # 生成可能的子词组合
            for word in word_counts:
                if len(word) > 1:
                    # 生成所有可能的子词拆分
                    for i in range(1, len(word)):
                        candidate = word[:i]
                        # 只添加不在词表中的子词
                        if candidate not in self.vocab:
                            subword_candidates.add(candidate)
            
            if not subword_candidates:
                break
                
            # 选择最有可能改善整体似然的子词
            best_token = None
            best_score = -float('inf')
            
            for token in subword_candidates:
                # 计算添加该token后的似然
                new_vocab = self.vocab | {token}
                score = self._compute_likelihood(word_counts, token)
                
                if score > best_score:
                    best_score = score
                    best_token = token
            
            # 添加最佳token到词表
            if best_token:
                self.vocab.add(best_token)
    
    def tokenize(self, text: str) -> List[str]:
        """应用WordPiece分词"""
        if not self.vocab:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        text = self.preprocess(text)
        # 按空格和标点初步分割
        words = re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)
        tokens = []
        
        for word in words:
            # 如果单词已在词表中，直接使用
            if word in self.vocab:
                tokens.append(word)
                continue
                
            # 应用最长匹配策略进行分词
            start = 0
            sub_tokens = []
            word_len = len(word)
            
            while start < word_len:
                end = word_len
                cur_substr = None
                
                # 找最长的匹配子词
                while start < end:
                    substr = word[start:end]
                    # 添加前缀标记（如果不是单词开头）
                    if start > 0:
                        substr = self.prefix + substr
                    
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    
                    end -= 1
                
                # 如果找到匹配，添加到结果
                if cur_substr:
                    sub_tokens.append(cur_substr)
                    start = end
                else:
                    # 未找到匹配，作为未知token
                    sub_tokens.append(self.unk_token)
                    start += 1  # 前进一个字符
            
            tokens.extend(sub_tokens)
        
        return tokens


class UnigramTokenizer(BaseTokenizer):
    """基于Unigram的分词器实现"""
    def __init__(self, vocab_size: int = 10000):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_scores: Dict[str, float] = {}  # token及其分数
    
    def train(self, corpus: List[str], num_iterations: int = 5):
        """训练Unigram分词器（简化版EM算法）"""
        # 1. 初始词表（常用字符和符号）
        char_counts = Counter()
        for text in corpus:
            text = self.preprocess(text)
            words = re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)
            for word in words:
                for char in word:
                    char_counts[char] += 1
                # 添加单词本身
                char_counts[word] += 1
        
        # 初始词表包括最常见的字符和单词
        self.vocab = set(token for token, count in char_counts.most_common(self.vocab_size))
        self.vocab.update([self.unk_token, self.pad_token])
        
        # 2. EM迭代（简化版）
        for iteration in range(num_iterations):
            # E步：收集所有可能的切分方式
            token_counts = defaultdict(float)
            
            for text in corpus:
                text = self.preprocess(text)
                words = re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)
                
                for word in words:
                    # 获取所有可能的切分方式
                    segmentations = self._get_possible_segmentations(word)
                    
                    if not segmentations:
                        continue
                    
                    # 计算每个切分的概率（使用简单频率）
                    total = sum(len(seg) for seg in segmentations)
                    
                    # 统计token在切分中的出现
                    for seg in segmentations:
                        for token in seg:
                            token_counts[token] += len(seg) / total
            
            # M步：更新词表，移除低频token
            total_count = sum(token_counts.values())
            self.token_scores = {}
            
            # 计算每个token的分数（相对频率）
            for token in token_counts:
                if token in self.vocab:
                    self.token_scores[token] = token_counts[token] / total_count
            
            # 保留分数最高的token
            sorted_tokens = sorted(self.token_scores.items(), key=lambda x: x[1], reverse=True)
            self.vocab = set(token for token, score in sorted_tokens[:self.vocab_size])
            self.vocab.update([self.unk_token, self.pad_token])
    
    def _get_possible_segmentations(self, word: str) -> List[List[str]]:
        """获取单词所有可能的切分方式（递归实现）"""
        if len(word) == 0:
            return [[]]
            
        segmentations = []
        
        # 尝试所有可能的切分点
        for i in range(1, len(word) + 1):
            prefix = word[:i]
            
            # 如果前缀在词表中，递归处理剩余部分
            if prefix in self.vocab:
                for seg in self._get_possible_segmentations(word[i:]):
                    segmentations.append([prefix] + seg)
        
        return segmentations
    
    def tokenize(self, text: str) -> List[str]:
        """应用Unigram分词（使用Viterbi算法）"""
        if not self.token_scores:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        text = self.preprocess(text)
        # 按空格和标点初步分割
        words = re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)
        tokens = []
        
        for word in words:
            # 如果没有词表，整个单词作为token
            if not self.vocab:
                tokens.append(word)
                continue
                
            # 使用动态规划找到最佳切分
            n = len(word)
            # dp[i]存储到第i个字符的最佳切分和分数
            dp = [{"score": -float('inf'), "tokens": []} for _ in range(n + 1)]
            dp[0] = {"score": 0.0, "tokens": []}
            
            for i in range(1, n + 1):
                for j in range(i):
                    substring = word[j:i]
                    
                    # 检查子字符串是否是有效token
                    if substring in self.vocab:
                        token_score = self.token_scores.get(substring, 0.001)  # 给未知token一个小概率
                        # 简化处理：取对数避免下溢，这里直接使用score
                        new_score = dp[j]["score"] + token_score
                        
                        if new_score > dp[i]["score"]:
                            dp[i] = {
                                "score": new_score,
                                "tokens": dp[j]["tokens"] + [substring]
                            }
            
            tokens.extend(dp[n]["tokens"])
        
        return tokens


class SentencePieceTokenizer(BaseTokenizer):
    """SentencePiece风格的分词器实现"""
    def __init__(self, vocab_size: int = 10000):
        super().__init__()
        self.vocab_size = vocab_size
        self.piece_to_id: Dict[str, int] = {}
        self.id_to_piece: Dict[int, str] = {}
        self.algorithm = "bpe"  # 可用 "bpe" 或 "unigram"
    
    def _prepare_corpus(self, corpus: List[str]) -> List[str]:
        """准备语料：空格替换为边界标记"""
        processed_corpus = []
        for text in corpus:
            # 将空格替换为特殊符号 (U+2581)
            text = self.preprocess(text)
            text = re.sub(r'\s+', '▁', text)  # 使用▁替代空格
            processed_corpus.append(text)
        return processed_corpus
    
    def train(self, corpus: List[str], algorithm: str = "bpe"):
        """训练分词器"""
        self.algorithm = algorithm
        processed_corpus = self._prepare_corpus(corpus)
        
        if algorithm == "bpe":
            bpe_tokenizer = BPETokenizer(vocab_size=self.vocab_size)
            bpe_tokenizer.train(processed_corpus)
            self.vocab = bpe_tokenizer.vocab
        elif algorithm == "unigram":
            unigram_tokenizer = UnigramTokenizer(vocab_size=self.vocab_size)
            unigram_tokenizer.train(processed_corpus)
            self.vocab = unigram_tokenizer.vocab
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        # 构建ID映射
        self.piece_to_id = {piece: idx for idx, piece in enumerate(sorted(self.vocab))}
        self.id_to_piece = {idx: piece for piece, idx in self.piece_to_id.items()}
    
    def tokenize(self, text: str) -> List[str]:
        """应用分词"""
        if not self.vocab:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        # 预处理文本：空格替换为边界标记
        text = self.preprocess(text)
        text = re.sub(r'\s+', '▁', text)
        
        if self.algorithm == "bpe":
            # 使用BPE方法
            tokenizer = BPETokenizer()
            tokenizer.vocab = self.vocab
            tokenizer.merges = self._recreate_merges()
            tokens = tokenizer.tokenize(text)
        else:  # unigram
            # 使用Unigram方法
            tokenizer = UnigramTokenizer()
            tokenizer.vocab = self.vocab
            # 这里简化处理，实际应该存储token_scores
            tokens = tokenizer.tokenize(text)
        
        return tokens
    
    def _recreate_merges(self) -> Dict[Tuple[str, str], str]:
        """根据词表重建合并规则"""
        merges = {}
        seen = set()
        
        # 按照长度排序，优先合并更长的子词
        sorted_vocab = sorted(self.vocab, key=len, reverse=True)
        
        for piece in sorted_vocab:
            if len(piece) > 1 and piece not in seen:
                # 尝试找到可能的合并
                for i in range(1, len(piece)):
                    left = piece[:i]
                    right = piece[i:]
                    
                    if left in self.vocab and right in self.vocab:
                        merges[(left, right)] = piece
                        seen.add(piece)
                        break
        
        return merges
    
    def encode(self, text: str) -> List[int]:
        """将文本转换为ID序列"""
        tokens = self.tokenize(text)
        return [self.piece_to_id.get(token, self.piece_to_id.get(self.unk_token, 0)) for token in tokens]
    
    def decode(self, ids: List[int]) -> str:
        """将ID序列转换回文本"""
        tokens = [self.id_to_piece.get(id, self.unk_token) for id in ids]
        text = "".join(tokens).replace('▁', ' ')  # 恢复空格
        return text.strip()


# 使用示例
if __name__ == "__main__":
    # 示例文本
    text = "The Transformer architecture revolutionized NLP tasks. Let's build a language model!"
    
    # 1. 基于词的分词
    word_tokenizer = WordTokenizer()
    word_tokens = word_tokenizer.tokenize(text)
    print("Word Tokenization:", word_tokens)
    
    # 2. 基于字符的分词
    char_tokenizer = CharTokenizer()
    char_tokens = char_tokenizer.tokenize(text)
    print("\nChar Tokenization:", char_tokens[:10], "...")
    
    # 3. BPE分词
    bpe_tokenizer = BPETokenizer(vocab_size=50)
    corpus = [
        "The Transformer architecture revolutionized NLP tasks.",
        "Let's build a language model!",
        "Natural Language Processing is fascinating."
    ]
    bpe_tokenizer.train(corpus)
    bpe_tokens = bpe_tokenizer.tokenize(text)
    print("\nBPE Tokenization:", bpe_tokens)
    
    # 4. WordPiece分词
    wp_tokenizer = WordPieceTokenizer(vocab_size=1000)
    wp_tokenizer.train(corpus)
    wp_tokens = wp_tokenizer.tokenize(text)
    print("\nWordPiece Tokenization:", wp_tokens)
    
    # 5. Unigram分词
    unigram_tokenizer = UnigramTokenizer(vocab_size=1000)
    unigram_tokenizer.train(corpus)
    unigram_tokens = unigram_tokenizer.tokenize(text)
    print("\nUnigram Tokenization:", unigram_tokens)
    
    # 6. SentencePiece风格的分词 (使用BPE)
    sp_tokenizer = SentencePieceTokenizer(vocab_size=1000)
    sp_tokenizer.train(corpus, algorithm="bpe")
    sp_tokens = sp_tokenizer.tokenize(text)
    print("\nSentencePiece Tokenization (BPE):", sp_tokens)
    
    # 7. SentencePiece风格的分词 (使用Unigram)
    sp_tokenizer.train(corpus, algorithm="unigram")
    sp_tokens = sp_tokenizer.tokenize(text)
    print("\nSentencePiece Tokenization (Unigram):", sp_tokens)
    
    # 8. 编码和解码测试
    encoded = sp_tokenizer.encode(text)
    decoded = sp_tokenizer.decode(encoded)
    print("\nOriginal Text:", text)
    print("Encoded IDs:", encoded[:10], "...")
    print("Decoded Text:", decoded)