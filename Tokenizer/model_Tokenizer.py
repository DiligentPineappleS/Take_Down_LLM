from collections import Counter, defaultdict
import re
import sentencepiece as spm
from transformers import BertTokenizer

class BaseTokenizer:
    """基础分词器类，包含预处理功能"""
    def __init__(self):
        # 特殊标记
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
    
    def preprocess(self, text):
        """文本预处理：小写化、去除多余空格、特殊字符处理"""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)  # 合并多个空格
        text = text.strip()
        return text

class WordTokenizer(BaseTokenizer):
    """基于词的分词器"""
    def tokenize(self, text):
        text = self.preprocess(text)
        # 使用正则表达式按空格和标点分词
        tokens = re.findall(r'\b\w+\b|[.,!?;]', text)
        return tokens

class CharTokenizer(BaseTokenizer):
    """基于字符的分词器"""
    def tokenize(self, text):
        text = self.preprocess(text)
        # 拆分为单个字符
        return list(text)

class BPETokenizer(BaseTokenizer):
    """BPE分词器实现"""
    def __init__(self, vocab_size=10000):
        super().__init__()
        self.vocab_size = vocab_size
        self.vocab = []
        self.merges = {}  # 存储合并规则
        
    def _get_stats(self, vocab):
        """统计相邻符号对出现频率"""
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def train(self, corpus):
        """训练BPE分词器"""
        # 初始词表：所有字符 + 特殊标记
        vocab = Counter()
        for text in corpus:
            text = self.preprocess(text)
            words = re.findall(r'\b\w+\b|[.,!?;]', text)
            for word in words:
                vocab[' '.join(list(word)) + ' </w>'] += 1
        
        self.vocab = [char for char in set(''.join(vocab.keys()))] + [self.unk_token, self.pad_token]
        
        # 迭代合并直到达到目标词表大小
        while len(self.vocab) < self.vocab_size:
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            
            # 获取最高频符号对
            best_pair = max(pairs, key=pairs.get)
            new_symbol = ''.join(best_pair)
            
            # 更新词表和合并规则
            self.vocab.append(new_symbol)
            self.merges[best_pair] = new_symbol
            
            # 更新词汇
            new_vocab = {}
            for word, freq in vocab.items():
                w = word.split()
                new_word = []
                i = 0
                while i < len(w):
                    if i < len(w) - 1 and (w[i], w[i + 1]) == best_pair:
                        new_word.append(new_symbol)
                        i += 2
                    else:
                        new_word.append(w[i])
                        i += 1
                new_vocab[' '.join(new_word)] = freq
            vocab = new_vocab
    
    def tokenize(self, text):
        """应用BPE分词"""
        if not self.merges:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        text = self.preprocess(text)
        words = re.findall(r'\b\w+\b|[.,!?;]', text)
        tokens = []
        
        for word in words:
            current = list(word) + ['</w>']
            
            # 应用合并规则
            while True:
                pairs = set(zip(current[:-1], current[1:]))
                merge_candidate = None
                
                # 查找最长的可合并对
                for pair in pairs:
                    if pair in self.merges:
                        merge_candidate = pair
                        break  # 贪心策略：取第一个匹配
                
                if not merge_candidate:
                    break
                    
                merged = self.merges[merge_candidate]
                new_current = []
                i = 0
                while i < len(current):
                    if i < len(current) - 1 and (current[i], current[i + 1]) == merge_candidate:
                        new_current.append(merged)
                        i += 2
                    else:
                        new_current.append(current[i])
                        i += 1
                current = new_current
            
            tokens.extend(current)
        
        # 移除结束标记
        tokens = [t for t in tokens if t != '</w>']
        return tokens

class WordPieceTokenizer(BaseTokenizer):
    """WordPiece分词器实现"""
    def __init__(self, max_vocab_size=10000):
        super().__init__()
        self.max_vocab_size = max_vocab_size
        self.tokenizer = None
    
    def train(self, corpus):
        """使用transformers库训练WordPiece"""
        from tokenizers import Tokenizer, models, normalizers
        from tokenizers.processors import TemplateProcessing
        
        # 创建空的WordPiece模型
        tokenizer = Tokenizer(models.WordPiece(unk_token=self.unk_token))
        
        # 添加规范化处理器
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFD(),         # Unicode规范化
            normalizers.Lowercase(),   # 小写化
            normalizers.StripAccents() # 去除重音
        ])
        
        # 训练分词器
        tokenizer.train_from_iterator(
            iter(corpus), 
            vocab_size=self.max_vocab_size,
            min_frequency=2
        )
        
        # 添加特殊标记
        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", 1),
                ("[SEP]", 2),
            ]
        )
        
        self.tokenizer = tokenizer
    
    def tokenize(self, text):
        """分词函数"""
        if not self.tokenizer:
            raise ValueError("Tokenizer not trained. Call train() first.")
            
        text = self.preprocess(text)
        output = self.tokenizer.encode(text)
        return output.tokens

class SentencePieceTokenizer:
    """SentencePiece分词器封装"""
    def __init__(self, model_file='spm.model'):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_file)
    
    def tokenize(self, text):
        """分词函数"""
        return self.sp.encode_as_pieces(text)


# 使用示例
if __name__ == "__main__":
    # 示例文本
    text = "The Transformer architecture revolutionized NLP tasks. Let's build a language model!"
    
    # 基于词的分词
    word_tokenizer = WordTokenizer()
    print("Word Tokenization:", word_tokenizer.tokenize(text))
    
    # 基于字符的分词
    char_tokenizer = CharTokenizer()
    print("Char Tokenization:", char_tokenizer.tokenize(text))
    
    # BPE分词
    bpe_tokenizer = BPETokenizer(vocab_size=50)
    corpus = [
        "The Transformer architecture revolutionized NLP tasks.",
        "Let's build a language model!"
    ]
    bpe_tokenizer.train(corpus)
    print("BPE Tokenization:", bpe_tokenizer.tokenize(text))
    
    # WordPiece分词 (使用训练好的Mini-BERT)
    wp_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("WordPiece Tokenization:", wp_tokenizer.tokenize(text))
    
    # SentencePiece分词
    # 需要先训练模型，这里使用预训练模型示例
    sp_tokenizer = SentencePieceTokenizer()
    # 真实使用时需要先训练模型:
    # spm.SentencePieceTrainer.train(input='corpus.txt', model_prefix='spm', vocab_size=5000)
    print("SentencePiece Tokenization:", sp_tokenizer.tokenize(text))