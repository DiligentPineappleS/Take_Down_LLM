
def fix_chunk(text,chunk_lenght=512,overlap= 0.3):
    """
    固定长度分块方法 - 按预设字符数切割文本
    
    原理:
      1. 从文本起始位置开始，按固定步长滑动窗口
      2. 每个窗口截取chunk_size长度的文本
      3. 相邻块保留overlap比例的重叠区域
      
    优点: 实现简单，处理速度快，适合非结构化文本
    缺点: 可能切断句子语义边界
    
    参数:
      text: 输入文本字符串
      chunk_size: 目标块字符数 (默认512)
      overlap: 块间重叠比例 (0-1, 默认0.1)
      
    返回:
      分块后的文本列表
    """
    chunks_list = []
    start_idx = 0
    steps = int(chunk_lenght*overlap)
    if steps<=0:
        steps=1
    
    while start_idx<len(text):
        end_idx = min(start_idx+chunk_lenght,len(text))
        chunk = text[start_idx:end_idx]
        chunks_list.append(chunk)

        if end_idx == len(text):
            break

        start_idx = end_idx - steps
    
    return chunks_list


def recursice_chunk(text,sep_list =['\n\n','\n','。','；','，','？','！'],max_chunk_length =512,min_chunk_length = 128):
    """
    递归分块方法 - 按文档层次结构分层切割
    原理:
      1. 尝试使用分隔符列表分割文本
      2. 优先使用高优先级分隔符(如段落)
      3. 对过大的块递归应用同样策略
    优点: 保留文档结构，适合多层级内容
    缺点: 依赖分隔符优先级设计
    参数:
      text: 输入文本字符串
      max_size: 最大块字符数 (默认512)
      separators: 分隔符优先级列表 (默认顺序: 段落>句子>词语)
      min_size: 最小块大小，小于此值尝试合并 (默认128)
      
    返回:
      分块后的文本列表
    """

    def split_recursive(text):
        if len(text)<=max_chunk_length:
            return [text]
        for sep in sep_list:
            if sep in text:
                text_list = text.split(sep)
                if all(len(temp_text)<= max_chunk_length for temp_text in text_list):
                    result = []
                    for i , text_chunk in enumerate(text_list):
                        if i <len(text_chunk)-1:
                            result.append(text_chunk+sep)
                        else:
                            result.append(text_chunk)
                    return result
    text_chunks = split_recursive(text)
    res_text_chunks = []
    current_chunk = ""
    for chunk in text_chunks:
        if len(current_chunk) + len(chunk) > max_chunk_length:
            if current_chunk:
                res_text_chunks.append(current_chunk)
                current_chunk = ""

        if len(chunk)>max_chunk_length:
            res_text_chunks.extend(split_recursive(chunk))
        else:
            if current_chunk and len(current_chunk)<min_chunk_length:
                current_chunk +=chunk
            else:
                if current_chunk:
                    res_text_chunks.append(current_chunk)
                current_chunk = chunk
    if current_chunk:
        res_text_chunks.append(current_chunk)
    res_chunks = []
    for chunk in res_text_chunks:
        if len(chunk)<min_chunk_length and res_chunks:
            res_chunks[-1]  = res_chunks[-1] + chunk
        else:
            res_chunks.append(chunk)
            
    return res_text_chunks


def docment_struture_chunk(text,struture_makers =  ['\n# ', '\n## ', '\n### ', '\n#### ', '\n**', '\n- ', '\n1. '],max_chunk_length =512):
    """
    文档结构分块方法 - 根据文档标记分块
    
    原理:
      1. 按行分割文本
      2. 识别结构标记(标题、列表等)
      3. 以结构标记为边界创建新块
    优点: 保留文档逻辑结构，适合技术文档
    缺点: 依赖文档格式规范性
    参数:
      text: 输入文本字符串
      structure_markers: 结构标记列表 (默认支持Markdown标题)
      max_size: 最大块大小，过大会递归分块 (默认768)
      
    返回:
      分块后的文本列表
    """
    text_list = text.split('\n')
    chunk_list =[]
    current_chunk = ""

    for temp_text in text_list:
        if any(marker in temp_text for marker in struture_makers):
            if current_chunk:
                chunk_list.append(current_chunk.strip())
                current_chunk = ""
            current_chunk = temp_text +" "
        else:
            current_chunk = current_chunk +temp_text +" "
    if current_chunk:
        chunk_list.append(current_chunk.strip())
    res_chunks = []
    for chunk in chunk_list:
        if len(chunk) > max_chunk_length:
            sub_chunk = recursice_chunk(chunk,max_chunk_length=max_chunk_length)
            res_chunks.extend(sub_chunk)
        else:
            res_chunks.append(chunk)
    return res_chunks


import nltk
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer,BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese")


def semantic_chunk(text,threhold=0.85,min_chunk_len=100,max_chunk_len = 512):
    """  
    原理:
      1. 将文本分割为句子
      2. 计算相邻句子的嵌入相似度
      3. 根据相似度阈值合并相关句子
    优点: 保持语义一致性，避免中断主题
    缺点: 计算开销大，需要BERT模型
    参数:
      text: 输入文本
      tokenizer: 分词器 (如BertTokenizer)
      model: 嵌入模型 (如BertModel)
      device: 计算设备 ('cpu'/'cuda')
      threshold: 相似度合并阈值 (0-1, 默认0.85)
      min_chunk_size: 最小块字符数 (默认100)
      max_chunk_size: 最大块字符数 (默认512)
      
    返回:
      分块后的文本列表
    """
    senttences = sent_tokenize(text)
    if len(senttences)<2:
        return [text]
    senttence_embedding = []
    for sent in senttences:
        sent_tokenize = tokenizer(sent,return_tensor = 'pt',padding = True,truncation = True,max_length=128)
        sent_tokenize = {k:v for k,v in sent_tokenize.items()}
        with torch.no_grad():
            responce = model(**sent_tokenize)
        sent_embedding = responce.last_hidden_state[:,0,:].cpu().numpy()
        senttence_embedding.append(sent_embedding)
    senttence_embedding = np.vstack(senttence_embedding)

    chunks_list =[]
    current_chunk = senttences[0]
    current_chunk_emb = senttence_embedding[0].reshape(1,-1)

    for i in range(1,len(senttences)):
        sent_emb = senttence_embedding[i].reshape(1,-1)
        similarity = cosine_similarity(current_chunk_emb,sent_emb)[0][0]
        if similarity>threhold and len(current_chunk)+len(senttences[i])<max_chunk_len:
            current_chunk += " " +senttences[i]
            current_chunk_emb = (current_chunk_emb *(i) +sent_emb)/(i+1)
        else:
            if len(current_chunk)>= min_chunk_len:
                chunks_list.append(current_chunk)
            current_chunk = senttences[i]
            current_chunk_emb = sent_emb
    if current_chunk and len(current_chunk)>= min_chunk_len:
        chunks_list.append(current_chunk)
    return chunks_list

def late_chunk(text,max_chunk_length,global_context_size):
    """
    后期分块 - 先获取全局上下文再切分
    原理:
      1. 用大窗口(全局上下文)分块获取背景信息
      2. 在全局块内应用精细分块方法
    优点: 解决长距离依赖问题
    缺点: 需要额外处理步骤
    参数:
      text: 输入文本
      chunk_size: 目标块大小 (默认512)
      global_context_size: 全局上下文大小 (默认1024)
      
    返回:
      分块后的文本列表
    """
    if  not text:
        return []

    global_chunks = fix_chunk(text,global_context_size,overlap=0.3)
    res_chunks = []
    for global_chunk in global_chunks:
        struct_chunks = docment_struture_chunk(global_chunk,max_chunk_length=max_chunk_length)
        for chunk in struct_chunks:
            if len(chunk) > max_chunk_length:
                res_chunks.extend(recursice_chunk(chunk,max_chunk_length=max_chunk_length))
            else:
                res_chunks.append(chunk)
    return res_chunks

def combine_chunk(text,max_chunk_length):
    """
    混合分块 - 组合多种策略
    原理:
      1. 首先尝试文档结构分块
      2. 对过大的块应用递归分块
      3. 对过小的块进行合并
    优点: 自适应各种文档类型
    参数:
      text: 输入文本
      chunk_size: 目标块大小 (默认512)  
    返回:
      分块后的文本列表
    """
    if not text:
        return []
    struct_chunks = docment_struture_chunk(text=text,max_chunk_length=max_chunk_length*2)

    res_chunks = []
    res_text_chunks = []
    current_chunk = ""

    for chunk in struct_chunks:
        if current_chunk and len(current_chunk) < max_chunk_length/2:
            current_chunk  = current_chunk + " " + chunk
        else:
            if current_chunk:
                res_text_chunks.append(current_chunk)
            current_chunk = chunk
    
    if current_chunk:
        res_text_chunks.append(current_chunk)
    
    for chunk in res_text_chunks:
        if len(chunk) > max_chunk_length*1.5:
            sub_chunks = recursice_chunk(chunk,max_chunk_length=max_chunk_length,min_chunk_length=max_chunk_length/4)
            res_chunks.extend(sub_chunks)
        else:
            res_chunks.append(chunk)
    return res_chunks

def metadata_chunk(text,metadata:dict,max_chunK_length=512):
    conbine_chunks = combine_chunk(text,max_chunK_length)
    metadata_list =[]
    for key ,value in metadata.items():
        metadata_list.append(f"{key.upper()}:{value}")
    res_chunks =[]
    idx = 0

    for i ,chunk in enumerate(conbine_chunks):
        metadata_chunks =[]
        chunk_nums = min(3,len(metadata_chunks) - idx)
        for j in range(chunk_nums):
            metadata_chunks.append(metadata_list[idx])
            idx = idx+1
        metadata_text = "\n".join(metadata_chunks)
        metadata_str = f"""
        [块ID: {i}]
        [元数据]:
        {metadata_text}
        [内容]:
        {chunk}
        """
        res_chunks.append(metadata_str)
    return res_chunks




