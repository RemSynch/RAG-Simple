from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from pymilvus import MilvusClient
from typing import List


class MilvusDB:
    INDEX_TYPE = 'HNSW'  # IVF_FLAT, HNSW
    METRIC_TYPE = 'COSINE'  # L2, IP, COSINE
    HOST = '192.168.22.128'  # 改为你的向量数据库所在的ip地址
    PORT = 19530

    def __init__(self, host=HOST, port=PORT) -> None:
        self.host = host
        self.port = port
        print(f"连接数据库 {host}:{port}")
        connections.connect(host=self.host, port=self.port)

    # 查询当前创建了哪些向量集合
    def list_collections(self) -> List[str]:
        try:
            collections = utility.list_collections()
            return collections
        except Exception as e:
            print(f"查询集合列表时出错: {e}")
            return []

    # 查询数据库中某个文件的向量数据

    def search_data_by_source(self, collection_name: str, source: str) -> List[str]:
        collection = Collection(collection_name)
        collection.load()
        res = collection.query(
            expr=f'source == "{source}"',
            output_fields=['source', 'vector']
        )
        return res

        # 根据文件路径删除向量数据

    def delete_data_by_source(self, collection_name: str, source: str) -> None:
        collection = Collection(collection_name)
        collection.load()
        collection.delete(expr=f'source == "{source}"')
        collection.flush()

    """创建表（如果存在会直接返回），并创建索引"""

    def create_collection(self, collection_name: str, dim: int) -> None:
        if utility.has_collection(collection_name):
            return

        fields = [
            FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name='source', dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name='page', dtype=DataType.INT64),
            FieldSchema(name='document', dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name='vector', dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]
        schema = CollectionSchema(fields=fields)
        collection = Collection(name=collection_name, schema=schema)

        # 创建索引，注意使用的类型
        index_params = {
            'metric_type': MilvusDB.METRIC_TYPE,
            'index_type': MilvusDB.INDEX_TYPE,
            "params": {"M": 8, "efConstruction": 64}
        }
        collection.create_index(field_name='vector', index_params=index_params)
        return

    """连接表，返回表内数据条数"""

    def connect_collection(self, collection_name: str) -> int:
        collection = Collection(collection_name)
        return self.collection.num_entities

    """插入数据（可以批量多条），返回表内数据条数"""

    def insert_data(self, user_data, collection_name: str) -> int:
        collection = Collection(collection_name)
        collection.insert(data=user_data, partition_name='_default')
        collection.flush()
        return collection.num_entities

    """查询向量字段"""

    def search_data(self, collection_name: str, key_feature: List[float], topk: int = 10) -> List[str]:
        collection = Collection(collection_name)
        collection.load()
        res = collection.search(
            data=[key_feature],
            limit=topk,  # 返回记录数
            anns_field='vector',  # 查询字段
            param={'nprobe': 10, 'metric_type': MilvusDB.METRIC_TYPE},
            output_fields=['document']
        )
        docs_result = []
        for hits in res:
            for hit in hits:
                docs_result.append(hit.entity.document)
        return docs_result

    """查询非向量字段"""

    def search_source(self, collection_name: str, dim: int) -> List[str]:
        self.create_collection(collection_name, dim)
        client = MilvusClient(
            uri=f"http://{self.host}:{self.port}"
        )
        try:
            res = client.query(
                collection_name=collection_name,
                filter='source like "uploads%"',  # 目前版本不支持distinct, group by等
                output_fields=['source']
            )
            source_set = list(set(item['source'] for item in res))
            return source_set
        except Exception as e:
            return []
