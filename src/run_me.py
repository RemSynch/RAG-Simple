# rag.py
import os
from business import DocumentHandler, MyMilvusRerankRetriever, Chat
from vectordbs import MilvusDB
from models import SentenceEmbeddingModel, RerankModel, ChatModel

# 初始化模型
embeddingModel: SentenceEmbeddingModel = SentenceEmbeddingModel()
rerankModel: RerankModel = RerankModel()
chatModel: ChatModel = ChatModel()
vectordb: MilvusDB = MilvusDB()

# 初始化业务逻辑
document_handler = DocumentHandler(vectordb=vectordb, embeddingModel=embeddingModel)
myRetriever: MyMilvusRerankRetriever = MyMilvusRerankRetriever()
myRetriever.set(vectordb, embeddingModel, rerankModel, document_handler)
myRetriever.set_search_result_num(top_k=20)
myRetriever.set_rerank_result_num(top_n=3)
chat: Chat = Chat(chatModel)


def upload_document(file_path):
    # 装载并切分文档
    list_doc = document_handler.load_and_split(file_path)
    # 提取特征embedding并存入向量数据库
    vector_num = document_handler.embed_and_store_vector(list_doc)
    return f"{file_path} uploaded successfully with {vector_num} vectors."


def chat_with_model(message, history):
    return chat.chat_with_history(message, history, myRetriever)


if __name__ == '__main__':
    current_collection = None

    while True:
        print("=" * 50)
        # 输出当前选中的集合名
        if current_collection:
            print(f"当前选中的集合: {current_collection}")
        else:
            print("未选中集合，请先选择一个集合。")
        print("选择操作:")
        print("1. 创建新集合")
        print("2. 选择集合")
        print("3. 上传文件")
        print("4. 问答")
        print("5. 查询已创建的向量集合")
        print("6. 退出")

        choice = input("输入选项(1/2/3/4/5/6): ").strip()

        if choice == '1':
            collection_name = input("输入新集合名称: ").strip()
            document_handler.create_collection(collection_name)

        elif choice == '2':
            collections = vectordb.list_collections()
            if collections:
                print("当前已创建的向量集合:")
                for idx, collection in enumerate(collections):
                    print(f"{idx + 1}. {collection}")
                collection_choice = int(input("选择集合编号: ").strip())
                if 1 <= collection_choice <= len(collections):
                    current_collection = collections[collection_choice - 1]
                    document_handler.set_collection(current_collection)
                    print(f"已切换到集合: {current_collection}")
                else:
                    print("无效的选择。")
            else:
                print("没有找到已创建的向量集合。")

        elif choice == '3':
            if not current_collection:
                print("请先选择一个集合！")
                continue
            file_path = input("输入文件路径: ").strip()
            if os.path.exists(file_path):
                print(upload_document(file_path))
            else:
                print("文件不存在，请检查路径。")

        elif choice == '4':
            if not current_collection:
                print("请先选择一个集合！")
                continue
            history = []
            while True:
                message = input("输入问题(输入'退出'结束问答): ").strip()
                if message.lower() == '退出':
                    break
                response = chat_with_model(message, history)
                print(f"模型回复: {response}")
                history.append({"role": "user", "content": [{"type": "text", "text": message}]})
                history.append({"role": "assistant", "content": [{"type": "text", "text": response}]})

        elif choice == '5':
            collections = vectordb.list_collections()
            if collections:
                print("当前已创建的向量集合:")
                for collection in collections:
                    print(f"- {collection}")
            else:
                print("没有找到已创建的向量集合。")

        elif choice == '6':
            break

        else:
            print("无效的选项，请重新输入。")
