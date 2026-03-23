"""
ChromaDB 向量数据库封装
- 本地持久化
- 支持粗粒度和细粒度两个 Collection
- CRUD + 检索操作
- 支持增量更新
"""

import json
import chromadb
from typing import Optional


class VectorStore:
    """ChromaDB 向量数据库封装"""

    def __init__(self, persist_dir: str = "./data/chroma_db",
                 collection_coarse: str = "o_award_coarse",
                 collection_fine: str = "o_award_fine"):
        """
        初始化 ChromaDB

        Args:
            persist_dir: 持久化目录
            collection_coarse: 粗粒度集合名
            collection_fine: 细粒度集合名
        """
        self.client = chromadb.PersistentClient(path=persist_dir)

        self.coarse_collection = self.client.get_or_create_collection(
            name=collection_coarse,
            metadata={"hnsw:space": "cosine"},
        )
        self.fine_collection = self.client.get_or_create_collection(
            name=collection_fine,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"ChromaDB 已初始化: coarse={self.coarse_collection.count()} 条, "
              f"fine={self.fine_collection.count()} 条")

    def add_chunks(self, chunks: list[dict], embeddings: list, granularity: str = "coarse"):
        """
        将 chunks 和对应的 embeddings 添加到向量库

        Args:
            chunks: chunk 列表
            embeddings: 对应的 embedding 向量
            granularity: "coarse" 或 "fine"
        """
        collection = self.coarse_collection if granularity == "coarse" else self.fine_collection

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = []
        for c in chunks:
            meta = {
                "paper_id": c["paper_id"],
                "section_type": c["section_type"],
                "year": c["year"],
                "problem_type": c["problem_type"],
                "techniques": json.dumps(c["techniques"]),
                "quality_tags": json.dumps(c["quality_tags"]),
                "page_range": json.dumps(c["page_range"]),
            }
            metadatas.append(meta)

        # 分批添加 (ChromaDB 单次限制)
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            emb_batch = embeddings[i:end]
            if hasattr(emb_batch, 'tolist'):
                emb_batch = emb_batch.tolist()

            collection.upsert(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                embeddings=emb_batch,
            )

        print(f"已添加 {len(ids)} 条 {granularity} chunks 到向量库")

    def search(self, query_embedding, granularity: str = "fine",
               top_k: int = 5, where: Optional[dict] = None,
               where_document: Optional[dict] = None) -> list[dict]:
        """
        向量检索

        Args:
            query_embedding: 查询向量
            granularity: 检索粒度
            top_k: 返回数量
            where: 元数据过滤条件
            where_document: 文档内容过滤

        Returns:
            检索结果列表
        """
        collection = self.coarse_collection if granularity == "coarse" else self.fine_collection

        if hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document

        results = collection.query(**kwargs)

        # 整理结果
        output = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                meta = results["metadatas"][0][i]
                output.append({
                    "chunk_id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "distance": results["distances"][0][i],
                    "paper_id": meta.get("paper_id", ""),
                    "section_type": meta.get("section_type", ""),
                    "year": meta.get("year", 0),
                    "problem_type": meta.get("problem_type", ""),
                    "techniques": json.loads(meta.get("techniques", "[]")),
                    "quality_tags": json.loads(meta.get("quality_tags", "[]")),
                    "page_range": json.loads(meta.get("page_range", "[0,0]")),
                })

        return output

    def get_all_by_section(self, section_type: str, granularity: str = "coarse",
                           limit: int = 50) -> list[dict]:
        """按 section_type 获取所有 chunk（用于基因提炼）"""
        collection = self.coarse_collection if granularity == "coarse" else self.fine_collection

        results = collection.get(
            where={"section_type": section_type},
            limit=limit,
            include=["documents", "metadatas"],
        )

        output = []
        if results["ids"]:
            for i in range(len(results["ids"])):
                meta = results["metadatas"][i]
                output.append({
                    "chunk_id": results["ids"][i],
                    "text": results["documents"][i],
                    "paper_id": meta.get("paper_id", ""),
                    "section_type": meta.get("section_type", ""),
                    "year": meta.get("year", 0),
                })
        return output

    def get_paper_ids(self) -> list[str]:
        """获取所有已入库的论文 ID"""
        results = self.coarse_collection.get(include=["metadatas"])
        paper_ids = set()
        if results["metadatas"]:
            for meta in results["metadatas"]:
                paper_ids.add(meta.get("paper_id", ""))
        return list(paper_ids)

    def delete_paper(self, paper_id: str):
        """删除指定论文的所有 chunks"""
        for collection in [self.coarse_collection, self.fine_collection]:
            results = collection.get(
                where={"paper_id": paper_id},
                include=[],
            )
            if results["ids"]:
                collection.delete(ids=results["ids"])
        print(f"已删除论文: {paper_id}")

    def get_stats(self) -> dict:
        """获取数据库统计信息"""
        return {
            "coarse_count": self.coarse_collection.count(),
            "fine_count": self.fine_collection.count(),
            "paper_ids": self.get_paper_ids(),
        }
