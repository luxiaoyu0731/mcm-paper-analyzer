"""
O 奖基因图谱提炼模块 v2
- 跨论文元数据统计（技术、质量标签、章节长度等）
- 多轮 LLM 分析（不截断原文）
- 多样性采样（覆盖更多论文而非集中少量）
- 结构化统计 + LLM 归纳双管齐下
"""

import json
import os
import re
from collections import Counter, defaultdict
from typing import Optional
from openai import OpenAI


# 需要提炼的章节类型
SECTION_TYPES = [
    "abstract", "intro", "assumption", "modeling",
    "validation", "sensitivity", "visualization", "conclusion",
]

GENE_EXTRACTION_PROMPT = """你是 MCM/ICM 资深评委。以下是从 {n} 篇 O 奖论文的 [{section_type}] 章节中检索到的代表性片段。

请分析这些片段的共性模式，输出 JSON 格式的"获奖基因"。

## 跨论文统计数据（基于全部入库论文）
{cross_paper_stats}

## 检索到的 O 奖片段（已按论文多样性采样）
{retrieved_chunks}

## 输出要求
请严格输出以下 JSON 格式（不要包含 markdown 代码块标记）：
{{
  "section_type": "{section_type}",
  "must_have": ["所有 O 奖论文都具备的特征，列 3-5 条"],
  "frequent": ["多数 O 奖论文具备的特征，列 3-5 条"],
  "rare_but_impressive": ["少数论文有但非常出彩的特征"],
  "anti_patterns": ["O 奖论文中从不出现的反面模式"],
  "visualization_standards": {{
    "common_chart_types": ["该章节常见的图表类型"],
    "avg_figures_per_section": 0,
    "best_practices": ["图表最佳实践"]
  }},
  "statistical": {{
    "avg_length_words": 0,
    "common_techniques": ["技术名称及使用频率"],
    "common_structures": ["结构模式"]
  }},
  "exemplars": [
    {{"paper_id": "...", "why": "为什么这篇的该章节特别优秀", "key_quote": "..."}}
  ]
}}
"""


class GeneExtractor:
    """O 奖基因图谱提炼器 v2"""

    def __init__(self, retriever, llm_client: OpenAI, model: str = "openai/gpt-4o"):
        self.retriever = retriever
        self.llm_client = llm_client
        self.model = model

    def _compute_cross_paper_stats(self, section_type: str) -> dict:
        """
        从向量库元数据中计算跨论文统计数据
        不依赖 LLM，直接从已标注的 metadata 中聚合
        """
        collection = self.retriever.vectorstore.coarse_collection

        # 获取该章节类型的所有 chunks 元数据
        results = collection.get(
            where={"section_type": section_type},
            include=["metadatas", "documents"],
        )

        if not results["ids"]:
            return {"note": "无数据"}

        # 统计
        technique_counter = Counter()
        quality_tag_counter = Counter()
        problem_type_counter = Counter()
        paper_ids = set()
        total_words = 0
        figure_count = 0
        table_count = 0
        chunk_count = len(results["ids"])

        for i in range(len(results["ids"])):
            meta = results["metadatas"][i]
            doc = results["documents"][i]

            paper_ids.add(meta.get("paper_id", ""))
            problem_type_counter[meta.get("problem_type", "unknown")] += 1

            # 解析 JSON 列表字段
            techniques = json.loads(meta.get("techniques", "[]"))
            for t in techniques:
                technique_counter[t] += 1

            quality_tags = json.loads(meta.get("quality_tags", "[]"))
            for tag in quality_tags:
                quality_tag_counter[tag] += 1

            # 文本统计
            total_words += len(doc.split())

            # 图表引用统计
            figure_count += len(re.findall(r'\[IMAGE ON PAGE', doc))
            figure_count += len(re.findall(r'\[VISION ANALYSIS:', doc))
            table_count += len(re.findall(r'\[TABLE ON PAGE', doc))

        avg_words = total_words / max(chunk_count, 1)

        return {
            "total_papers": len(paper_ids),
            "total_chunks": chunk_count,
            "avg_chunk_words": round(avg_words),
            "technique_frequency": dict(technique_counter.most_common(15)),
            "quality_tag_frequency": dict(quality_tag_counter.most_common(10)),
            "problem_type_distribution": dict(problem_type_counter),
            "figure_references": figure_count,
            "table_references": table_count,
        }

    def _diverse_sample_chunks(self, section_type: str,
                                target_count: int = 30) -> list[dict]:
        """
        多样性采样：确保覆盖尽可能多的不同论文
        而非从 ChromaDB 的 get() 中随机返回（可能集中在少数论文）
        """
        collection = self.retriever.vectorstore.coarse_collection

        # 获取该章节所有 chunks
        results = collection.get(
            where={"section_type": section_type},
            include=["metadatas", "documents"],
            limit=500,  # 获取足够多
        )

        if not results["ids"]:
            return []

        # 按论文分组
        paper_chunks = defaultdict(list)
        for i in range(len(results["ids"])):
            pid = results["metadatas"][i].get("paper_id", "")
            paper_chunks[pid].append({
                "chunk_id": results["ids"][i],
                "text": results["documents"][i],
                "paper_id": pid,
                "section_type": section_type,
            })

        # 轮询采样：每篇论文取一个 chunk，循环直到达到目标数
        sampled = []
        paper_ids = list(paper_chunks.keys())
        round_idx = 0
        while len(sampled) < target_count and round_idx < 5:
            for pid in paper_ids:
                chunks = paper_chunks[pid]
                if round_idx < len(chunks):
                    sampled.append(chunks[round_idx])
                if len(sampled) >= target_count:
                    break
            round_idx += 1

        return sampled

    def extract_section_gene(self, section_type: str) -> dict:
        """提炼某个章节类型的获奖基因"""
        # 1. 计算跨论文统计数据
        cross_stats = self._compute_cross_paper_stats(section_type)

        # 2. 多样性采样 chunks
        chunks = self._diverse_sample_chunks(section_type, target_count=30)

        if not chunks:
            return {"section_type": section_type, "must_have": [], "note": "无数据"}

        # 3. 格式化检索结果 —— 不截断
        chunks_text = ""
        for i, c in enumerate(chunks, 1):
            chunks_text += f"\n### 片段 {i} [{c['paper_id']}]\n{c['text']}\n"

        # 4. 如果总文本过长，分批送入 LLM（GPT-4o 128K，一般不会超）
        stats_text = json.dumps(cross_stats, ensure_ascii=False, indent=2)

        prompt = GENE_EXTRACTION_PROMPT.format(
            n=cross_stats.get("total_papers", len(chunks)),
            section_type=section_type,
            cross_paper_stats=stats_text,
            retrieved_chunks=chunks_text,
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=4096,
            )
            content = response.choices[0].message.content.strip()

            # 清理 markdown 代码块标记
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]
            content = content.strip()

            gene = json.loads(content)
            # 附加统计数据到基因中
            gene["_cross_paper_stats"] = cross_stats
            return gene
        except json.JSONDecodeError:
            return {
                "section_type": section_type,
                "raw_analysis": content,
                "parse_error": True,
                "_cross_paper_stats": cross_stats,
            }
        except Exception as e:
            return {
                "section_type": section_type,
                "error": str(e),
                "_cross_paper_stats": cross_stats,
            }

    def extract_full_profile(self, output_path: Optional[str] = None) -> dict:
        """
        提炼完整的 O 奖基因图谱

        Args:
            output_path: 输出 JSON 文件路径

        Returns:
            完整的基因图谱字典
        """
        print("开始提炼 O 奖基因图谱 (v2 增强版)...")

        gene_profile = {
            "version": "2.0",
            "sections": {},
            "statistical_profile": {},
            "global_statistics": {},
        }

        for section_type in SECTION_TYPES:
            print(f"  提炼 [{section_type}] 基因...")
            gene = self.extract_section_gene(section_type)
            gene_profile["sections"][section_type] = gene

        # 全局统计
        gene_profile["statistical_profile"] = self._compute_global_statistics()
        gene_profile["global_statistics"] = self._compute_technique_rankings()

        # 持久化
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(gene_profile, f, ensure_ascii=False, indent=2)
            print(f"基因图谱已保存到: {output_path}")

        return gene_profile

    def _compute_global_statistics(self) -> dict:
        """从向量库中计算全局统计数据"""
        stats = self.retriever.vectorstore.get_stats()
        paper_ids = stats.get("paper_ids", [])

        return {
            "total_papers": len(paper_ids),
            "total_coarse_chunks": stats.get("coarse_count", 0),
            "total_fine_chunks": stats.get("fine_count", 0),
            "paper_ids": paper_ids,
        }

    def _compute_technique_rankings(self) -> dict:
        """跨所有章节统计技术使用频率排行"""
        collection = self.retriever.vectorstore.coarse_collection
        results = collection.get(include=["metadatas"], limit=10000)

        if not results["metadatas"]:
            return {}

        technique_counter = Counter()
        quality_counter = Counter()
        section_counter = Counter()

        for meta in results["metadatas"]:
            section_counter[meta.get("section_type", "unknown")] += 1
            techniques = json.loads(meta.get("techniques", "[]"))
            for t in techniques:
                technique_counter[t] += 1
            tags = json.loads(meta.get("quality_tags", "[]"))
            for tag in tags:
                quality_counter[tag] += 1

        return {
            "technique_rankings": dict(technique_counter.most_common(20)),
            "quality_tag_rankings": dict(quality_counter.most_common(10)),
            "section_distribution": dict(section_counter),
        }
