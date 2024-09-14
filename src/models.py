# models.py
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoModel, AutoTokenizer
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from FlagEmbedding import FlagReranker
from typing import List

from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from typing import List

"""
句子embedding模型
"""


# class SentenceEmbeddingModel():
#     # MODEL_NAME = "moka-ai/m3e-base"
#     MODEL_NAME = "../m3e-base"
#     # MODEL_NAME = "../gte_sentence-embedding_multilingual-base"
#     embeddings: HuggingFaceBgeEmbeddings
#
#     def __init__(self) -> None:
#         print(f"初始化SentenceEmbedding模型：{self.MODEL_NAME}")
#         # embedding model
#         model_name = self.MODEL_NAME
#         # model_kwargs = {'device': 'cpu'}
#         # encode_kwargs = {'normalize_embeddings': True}
#
#         # 手动加载模型和tokenizer，加入trust_remote_code参数
#
#         embeddings = HuggingFaceBgeEmbeddings(
#             model_name=model_name
#             # model_kwargs=model_kwargs,
#             # encode_kwargs=encode_kwargs
#         )
#         self.embeddings = embeddings
#
#     """embedding 单句"""
#
#     def embed_query(self, query: str) -> List[float]:
#         key_feat = self.embeddings.embed_query(query)
#         return key_feat
#
#     """embedding 一批句子"""
#
#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         list_feat = self.embeddings.embed_documents(texts)
#         return list_feat

class SentenceEmbeddingModel():
    MODEL_NAME = "../gte_sentence-embedding_multilingual-base"
    model: AutoModel
    tokenizer: AutoTokenizer
    output_dim: int = 768  # 输出嵌入维度

    def __init__(self) -> None:
        print(f"初始化SentenceEmbedding模型：{self.MODEL_NAME}")

        # 加载模型和tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME, trust_remote_code=True)

    """embedding 单句"""

    def embed_query(self, query: str) -> List[float]:
        # Tokenize输入文本
        inputs = self.tokenizer([query], max_length=8192, padding=True, truncation=True, return_tensors='pt')

        # 前向传播，获取模型输出
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 获取句子嵌入：取[CLS] token（通常是位置[0]）的hidden state，维度为self.output_dim
        embeddings = outputs.last_hidden_state[:, 0][:self.output_dim]

        # 对嵌入进行归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 返回句子嵌入
        return embeddings.squeeze().tolist()

    """embedding 一批句子"""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Tokenize输入文本
        inputs = self.tokenizer(texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')

        # 前向传播，获取模型输出
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 获取每个句子的嵌入，取[CLS] token（位置[0]）的hidden state，维度为self.output_dim
        embeddings = outputs.last_hidden_state[:, 0][:, :self.output_dim]

        # 对嵌入进行归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 返回每个句子的嵌入列表
        return embeddings.tolist()


"""
embedding重排序模型
"""


# class RerankModel():
#     # MODEL_NAME = 'BAAI/bge-reranker-base'
#     MODEL_NAME = '../gte_passage-ranking_multilingual-base'
#     reranker: FlagReranker
#
#     def __init__(self) -> None:
#         print(f"初始化Rerank模型：{self.MODEL_NAME}")
#         model_name = self.MODEL_NAME
#         reranker = FlagReranker(model_name)
#         self.reranker = reranker
#
#     """query跟一批句子做比较，返回相似度最高的 top_k 条"""
#
#     def rank(self, query: str, texts: List[str], top_k=3) -> List[str]:
#         pairs = [[query, text] for text in texts]
#         scores = self.reranker.compute_score(pairs)
#         combined = list(zip(scores, pairs))
#         sorted_combined = sorted(combined, reverse=True)
#         sorted_pairs = [item[1] for item in sorted_combined]
#         return sorted_pairs[:top_k]


import torch
from modelscope import AutoModelForSequenceClassification, AutoTokenizer

class RerankModel():
    MODEL_NAME = '../gte_passage-ranking_multilingual-base'
    model: AutoModelForSequenceClassification
    tokenizer: AutoTokenizer

    def __init__(self) -> None:
        print(f"初始化Rerank模型：{self.MODEL_NAME}")
        # 加载模型和tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME, trust_remote_code=True)
        self.model.eval()  # 将模型置于评估模式，不进行梯度更新

    """query跟一批句子做比较，返回相似度最高的 top_k 条"""
    def rank(self, query: str, texts: List[str], top_k=3) -> List[str]:
        # 构建输入的 pair 列表
        pairs = [[query, text] for text in texts]

        # 使用 tokenizer 对 pairs 进行编码，生成模型输入
        inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=8192)

        # 前向传播计算相似度分数
        with torch.no_grad():
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()

        # 将分数和对应的文本对组合在一起并排序
        combined = list(zip(scores, texts))
        sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)

        # 返回相似度最高的 top_k 个句子
        sorted_texts = [item[1] for item in sorted_combined]
        return sorted_texts[:top_k]


"""
对话模型
"""


class ChatModel():
    MODEL_PATH = 'D:/A_MyCodingWorkSpace/project/PyCharmProject/Qwen2-vl/Qwen2-VL-2B-Instruct'

    def __init__(self) -> None:
        print(f"初始化chat模型：{self.MODEL_PATH}")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(self.MODEL_PATH, torch_dtype="auto",
                                                                     device_map="auto")
        self.processor = AutoProcessor.from_pretrained(self.MODEL_PATH)

    def generate_answer(self, conversation, image=None):
        # 使用处理器应用聊天模板，生成文本提示
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        # 预处理输入数据
        inputs = self.processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")  # 将输入数据移至GPU（如果可用）

        # 使用模型生成输出
        output_ids = self.model.generate(**inputs, max_new_tokens=128)

        # 解码生成的token为可读文本
        output_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)
        return output_text[0]
