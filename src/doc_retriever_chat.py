from PIL import Image
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Any
from vectordbs import MilvusDB
from models import SentenceEmbeddingModel, RerankModel,ChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

"""
文档处理
"""
class DocumentHandler():
    DIM = 768  # 维度固定

    def __init__(self, vectordb: MilvusDB, embeddingModel: SentenceEmbeddingModel) -> None:
        self.vectordb = vectordb
        self.embeddingModel = embeddingModel
        self.COLLECTION_NAME = None  # 当前集合名称

    def set_collection(self, collection_name: str):
        self.COLLECTION_NAME = collection_name

    def create_collection(self, collection_name: str):
        self.vectordb.create_collection(collection_name, self.DIM)
        self.set_collection(collection_name)
        print(f"已创建并切换到集合: {collection_name}")

    """上传文件"""
    def upload_file(self, read_file_path: str, write_file_path: str) -> str:
        with open(read_file_path, 'rb') as r_file:
            content = r_file.read()
            with open(write_file_path, 'wb') as w_file:
                w_file.write(content)
        return write_file_path

    """装载和切分文档"""
    """目前仅支持PDF、TXT"""
    def load_and_split(self, file_path: str) -> List[Document]:
        doc_list: List[Document]

        if file_path.lower().endswith('.pdf'):
            # 处理PDF文件
            doc_list = PyPDFLoader(file_path).load()
        elif file_path.lower().endswith('.txt'):
            # 处理TXT文件
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            doc_list = [Document(page_content=text, metadata={"source": file_path})]
        else:
            raise ValueError("This type of file can NOT be supported.")

        # 使用文本分割器将文档分块
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        documents = text_splitter.split_documents(doc_list)

        return documents

    """embedding并存储到向量数据库中"""

    def embed_and_store_vector(self, documents: List[Document]) -> int:
        # 建表和索引
        self.vectordb.create_collection(self.COLLECTION_NAME, self.DIM)

        # 生成向量
        source = [d.metadata['source'] for d in documents]

        # 如果没有 'page' 字段，则设置为默认值 1
        page = [d.metadata.get('page', 1) for d in documents]

        texts = [d.page_content for d in documents]
        vectors = self.embeddingModel.embed_documents(texts=texts)

        # 存入句子和向量
        insert_list = [source, page, texts, vectors]
        nums = self.vectordb.insert_data(insert_list, self.COLLECTION_NAME)

        return nums

    """获取支持的doc列表"""
    def get_doc_file_list(self) -> List[str]:
        return self.vectordb.search_source(self.COLLECTION_NAME, self.DIM)


class MyMilvusRerankRetriever(BaseRetriever):
    vectordb: Any
    embeddingModel: Any
    rerankModel: Any
    document_handler: Any
    top_k: int = 20
    top_n: int = 3

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    # 设置模型相关
    def set(self, vectordb: MilvusDB, embeddingModel: SentenceEmbeddingModel, rerankModel: RerankModel,
            document_handler: DocumentHandler) -> None:
        self.vectordb = vectordb
        self.embeddingModel = embeddingModel
        self.rerankModel = rerankModel
        self.document_handler = document_handler  # 设置 DocumentHandler 实例

    # 设置从db取的相似条数
    def set_search_result_num(self, top_k: int) -> None:
        self.top_k = top_k

    # 设置rerank后取的条数
    def set_rerank_result_num(self, top_n: int) -> None:
        self.top_n = top_n

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        collection_name = self.document_handler.COLLECTION_NAME  # 从实例获取集合名称
        if self.embeddingModel is None:
            raise ValueError("embedding model error")
        key_feature = self.embeddingModel.embed_query(query)
        if self.vectordb is None:
            raise ValueError("vectordb error")
        print(f"Collection Name: {collection_name}")
        text_search_result: List[str] = self.vectordb.search_data(collection_name, key_feature, topk=self.top_k)
        print("---------Embedding检索出的结果>>>>", text_search_result)
        if len(text_search_result) == 0:
            return []
        if self.rerankModel is None:
            raise ValueError("rerank model error")
        rerank_result: List[str] = self.rerankModel.rank(query, text_search_result)
        print("=========Rerank出的最终结果>>>>", rerank_result)
        if rerank_result:
            # return [Document(page_content=doc[1]) for doc in rerank_result]
            return [Document(page_content=doc) for doc in rerank_result]
        return []


class Chat:
    QA_PROMPT = """根据给定的上下文和聊天历史记录来回答最后的问题。如果你不知道答案，就说不知道，一定不能试图编造答案.
    上下文: {context}

    聊天历史记录: {chat_history}

    问题: {question}
    有帮助的答案:"""

    def __init__(self, chatModel: ChatModel):
        # 初始化模型和处理器
        print(f"初始化chat模型完毕")
        self.model = chatModel.model
        self.processor = chatModel.processor

    def chat_with_history(self, message: str, history: List[dict], retriever: MyMilvusRerankRetriever,
                          image_path=None) -> str:
        try:
            # 从Milvus数据库检索相关文档，作为对话上下文
            context_docs = retriever.get_relevant_documents(message)
            context_text = "\n".join([doc.page_content for doc in context_docs])
        except Exception as e:
            # 如果出现异常，比如没有文档可以检索，则上下文为空
            print(f"文档检索失败或没有文档可供检索，错误信息: {e}")
            context_text = ""

        # 使用自定义QA_PROMPT构建提示
        chat_history_text = "\n".join([f"用户: {item['content'][0]['text']}" for item in history])
        prompt = self.QA_PROMPT.format(context=context_text, chat_history=chat_history_text, question=message)
        # 使用 apply_chat_template 生成提示词
        conversation = history + [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        print("聊天记录：{}".format(conversation))

        if context_text:
            conversation.insert(0, {"role": "system", "content": context_text})

        prompt_chat_template = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        # print("-" * 50 + '模型输入提示词prompt' + "-" * 50)
        # print(prompt_chat_template)
        # print("-" * 150)

        # 如果有图像路径，加载图像
        if image_path:
            image = Image.open(image_path)
            inputs = self.processor(
                text=[prompt_chat_template], images=[image], padding=True, return_tensors="pt"
            )
        else:
            # 仅处理文本
            inputs = self.processor(text=[prompt_chat_template], padding=True, return_tensors="pt")

        inputs = inputs.to("cuda")  # 将输入数据移至GPU（如果可用）

        # 使用模型生成输出
        output_ids = self.model.generate(**inputs, max_new_tokens=1024)

        # 提取生成的新token（去除输入部分）
        output_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]

        # 解码生成的token为可读文本
        output_text = self.processor.batch_decode(output_ids, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=True)
        # print("-" * 50 + '模型回复output' + "-" * 50)
        # print(output_text[0])
        # print("-" * 150)
        return output_text[0]
