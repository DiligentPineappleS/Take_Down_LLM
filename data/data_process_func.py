# 0. 统一转为UTF-8
import chardet
from ftfy import fix_text

def fix_mixed_encoding(text):
    """
    解决中文编码混合问题：
    1. 识别多种编码格式
    2. 修正错误编码字符
    3. 统一转为UTF-8
    """
    # 修正常见错误编码
    text = fix_text(text)
    
    # 深度编码检测
    result = chardet.detect(text.encode())
    encoding = result['encoding'] or 'utf-8'
    
    try:
        return text.encode(encoding).decode('utf-8', errors='replace')
    except:
        return text.encode('latin1').decode('gb18030', errors='replace')


# 1. ​​精确去重（MD5哈希）​

import hashlib
from pyspark.sql import functions as F

def text_md5(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

df = spark.read.text("hdfs://path/to/raw_data")
df = df.withColumn("md5", F.udf(text_md5)(F.col("value")))
df = df.dropDuplicates(["md5"])  # 基于哈希值去重

# 2. ​​语义相似去重（MinHash+LSH）​
from pyspark.ml.feature import MinHashLSH
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import Tokenizer, NGram

# 中文需先分词（以jieba分词为例）
tokenizer = Tokenizer(inputCol="text", outputCol="words")
df = tokenizer.transform(df)

# 生成MinHash签名
mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
df = mh.fit(df).transform(df)

# 查询相似文档（阈值0.8）
result = mh.approxSimilarityJoin(df, df, 0.8, distCol="JaccardDistance")
result.filter("datasetA.id < datasetB.id")  # 避免自对比



# 3.1. ​​规则清洗（正则表达式）​
import re

# 常见中文噪声模式
noise_patterns = [
    r"[\x00-\x1F\x7F]",          # 控制字符
    r"【.*?】|$$.*?$$",          # 广告标识
    r"(?:回复|转发):@\w+\s",      # 社交媒体噪声
    r"\\[uU][0-9a-fA-F]{4}"      # Unicode乱码
]

def clean_text(text):
    for pattern in noise_patterns:
        text = re.sub(pattern, "", text)
    # 特殊处理：过长重复字符（超过3次）
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)  
    return text.strip()



# 3.2. ​​低质量内容过滤（BERT分类模型）
from transformers import BertTokenizer, BertForSequenceClassification
import torch
# 加载预训练质量分类模型（示例：microsoft/xtremedistil-l6-h384-zh）
model = BertForSequenceClassification.from_pretrained("quality_model_zh")
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

def is_high_quality(text, threshold=0.9):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    prob = torch.softmax(outputs.logits, dim=1)[0][1].item()  # 质量概率
    return prob > threshold




#4.1 简繁统一​
from opencc import OpenCC

cc = OpenCC('t2s')  # 繁转简（s2t为简转繁）
text_simplified = cc.convert("微信支付優惠領取")


# 4.2 全角转半角​
def full_to_half(text):
    return ''.join([chr(ord(c) - 65248) if 65281 <= ord(c) <= 65374 else c for c in text])



import re

# 4.3 中文敏感脱敏
patterns = {
    "ID_CARD": r"[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]",
    "PHONE": r"(?<!\d)1[3-9]\d{9}(?!\d)",
    "BANK_CARD": r"\b[1-9]\d{9,18}\b"
}

def desensitize(text):
    # 电话号码脱敏
    text = re.sub(patterns["PHONE"], lambda m: m.group()[:3] + "****" + m.group()[-4:], text)
    # 身份证脱敏
    text = re.sub(patterns["ID_CARD"], lambda m: m.group()[:6] + "********" + m.group()[-4:], text)
    return text


# 5. 端到端文本噪声校正
from transformers import MarianMTModel, MarianTokenizer

tokenizer = MarianTokenizer.from_pretrained("noise-cleaner-zh")
model = MarianMTModel.from_pretrained("noise-cleaner-zh")

def correct_text_noise(text):
    batch = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    generated = model.generate(**batch)
    return tokenizer.decode(generated[0], skip_special_tokens=True)
