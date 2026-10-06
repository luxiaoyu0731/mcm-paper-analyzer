"""
多路召回 + 后过滤检索模块 v4
- 三路召回（语义 + 章节 + 题型）
- 后过滤：同题号论文优先
- paper_letter_map.json 提供题号映射
"""

import re
import json
import os
from typing import Optional


class Retriever:
    """多路召回检索器"""

    def __init__(self, vectorstore, embedder, use_reranker: bool = False,
                 reranker_model: str = "BAAI/bge-reranker-base"):
        self.vectorstore = vectorstore
        self.embedder = embedder
        self.use_reranker = use_reranker
        self.reranker = None

        # 加载题号映射
        map_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'paper_letter_map.json')
        self.letter_map = {}
        if os.path.exists(map_path):
            with open(map_path) as f:
                self.letter_map = json.load(f)
            print(f"已加载题号映射: {len(self.letter_map)} 篇论文")

        if use_reranker:
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(reranker_model)
                print(f"Re-ranker 已加载: {reranker_model}")
            except Exception as e:
                print(f"Re-ranker 加载失败: {e}")
                self.use_reranker = False

    def _get_paper_letter(self, paper_id: str) -> str:
        """从 paper_id 提取题号字母"""
        # 先检查 paper_id 自带的题号
        m = re.search(r'_([A-F])_', paper_id)
        if m:
            return m.group(1)
        # 用7位编号查映射表
        num = re.search(r'(\d{7})', paper_id)
        if num and num.group(1) in self.letter_map:
            return self.letter_map[num.group(1)]['letter']
        return ""

    def retrieve_for_section(self, query_text: str, section_type: Optional[str] = None,
                             problem_type: Optional[str] = None,
                             problem_letter: Optional[str] = None,
                             top_k: int = 5, granularity: str = "fine") -> list[dict]:
        """
        多路召回 + 后过滤

        策略：
        1. 语义检索（广撒网）
        2. 章节类型过滤
        3. 题型过滤
        4. 合并去重
        5. 后过滤：同题号论文得分加成
        """
        query_embedding = self.embedder.encode_single(query_text)

        all_results = []
        seen_ids = set()

        def _merge(results):
            for r in results:
                if r["chunk_id"] not in seen_ids:
                    seen_ids.add(r["chunk_id"])
                    all_results.append(dict(r))

        # 路径 1: 语义检索（兜底，大范围）
        r1 = self.vectorstore.search(
            query_embedding, granularity=granularity, top_k=top_k * 3,
        )
        _merge(r1)

        # 路径 2: 同章节类型
        if section_type:
            try:
                r2 = self.vectorstore.search(
                    query_embedding, granularity=granularity, top_k=top_k,
                    where={"section_type": section_type},
                )
                _merge(r2)
            except Exception:
                pass

        # 路径 3: 同题型
        if problem_type:
            try:
                r3 = self.vectorstore.search(
                    query_embedding, granularity=granularity, top_k=top_k,
                    where={"problem_type": problem_type},
                )
                _merge(r3)
            except Exception:
                pass

        # 后过滤：同题号论文加成
        if problem_letter:
            for r in all_results:
                r_letter = self._get_paper_letter(r["paper_id"])
                if r_letter == problem_letter:
                    # 同题号：距离减半（更优先）
                    r["distance"] = r.get("distance", 1.0) * 0.5
                    r["_same_letter"] = True
                elif r_letter:
                    # 不同题号：轻微惩罚
                    r["distance"] = r.get("distance", 1.0) * 1.2
                    r["_same_letter"] = False

        # 排序
        if self.use_reranker and self.reranker and all_results:
            pairs = [(query_text, r["text"]) for r in all_results]
            scores = self.reranker.predict(pairs)
            for i, score in enumerate(scores):
                all_results[i]["rerank_score"] = float(score)
            all_results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        else:
            all_results.sort(key=lambda x: x.get("distance", 1.0))

        return all_results[:top_k]

    def retrieve_global(self, section_type: str, top_k: int = 10) -> list[dict]:
        """按 section_type 获取 O 奖论文的代表性片段"""
        return self.vectorstore.get_all_by_section(
            section_type=section_type,
            granularity="coarse",
            limit=top_k,
        )

    def retrieve_by_technique(self, technique: str, top_k: int = 5) -> list[dict]:
        """按技术/算法名精确匹配"""
        query_embedding = self.embedder.encode_single(
            f"We use {technique} method to solve the problem"
        )
        try:
            results = self.vectorstore.search(
                query_embedding, granularity="fine", top_k=top_k * 3,
                where_document={"$contains": technique},
            )
        except Exception:
            results = self.vectorstore.search(
                query_embedding, granularity="fine", top_k=top_k,
            )
        return results[:top_k]

    @staticmethod
    def _clean_paper_id(paper_id: str) -> str:
        """清理 paper_id 中的垃圾后缀"""
        cleaned = re.sub(r'_*公众号.*', '', paper_id)
        cleaned = re.sub(r'_*竞赛资料网.*', '', cleaned)
        cleaned = re.sub(r'_+$', '', cleaned)
        return cleaned

    def format_retrieval_context(self, results: list[dict]) -> str:
        """将检索结果格式化为 LLM 可用的上下文文本"""
        if not results:
            return "（无相关检索结果）"

        parts = []
        for i, r in enumerate(results, 1):
            pid = self._clean_paper_id(r['paper_id'])
            letter = self._get_paper_letter(r['paper_id']) or '?'
            ptype = r.get('problem_type', 'unknown')
            same = " ⭐同题号" if r.get("_same_letter") else ""
            source = f"[{pid}, 题{letter}, {r['section_type']}, 题型={ptype}, p.{r['page_range'][0]}-{r['page_range'][1]}]{same}"
            parts.append(f"### 参考片段 {i} {source}\n{r['text']}\n")

        return "\n".join(parts)
