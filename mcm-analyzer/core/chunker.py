"""
智能分块模块 - 双层分块策略
- 粗粒度块（~1500 tokens）：按自然章节切分
- 细粒度块（~300 tokens）：按段落/子小节切分
- 每个 chunk 附加元数据标签
"""

import re
from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 问题类型关键词
PROBLEM_TYPE_KEYWORDS = {
    "continuous": ["differential equation", "PDE", "ODE", "continuous", "heat", "flow", "diffusion", "wave"],
    "discrete": ["graph", "network", "combinatorial", "integer", "scheduling", "assignment", "discrete"],
    "optimization": ["optimize", "minimize", "maximize", "linear programming", "objective function", "constraint"],
    "data_driven": ["machine learning", "regression", "classification", "neural network", "deep learning",
                     "random forest", "clustering", "prediction", "training", "dataset"],
    "stochastic": ["Monte Carlo", "simulation", "probability", "stochastic", "random", "Markov"],
}

# 技术/算法关键词
TECHNIQUE_KEYWORDS = [
    "PDE", "ODE", "FEM", "finite element", "finite difference",
    "Monte Carlo", "Markov chain", "MCMC", "Bayesian",
    "linear programming", "integer programming", "dynamic programming",
    "genetic algorithm", "simulated annealing", "particle swarm",
    "neural network", "CNN", "LSTM", "RNN", "transformer",
    "regression", "SVM", "random forest", "XGBoost", "gradient boosting",
    "sensitivity analysis", "principal component", "PCA",
    "clustering", "k-means", "DBSCAN",
    "graph theory", "shortest path", "Dijkstra", "network flow",
    "time series", "ARIMA", "exponential smoothing",
    "cellular automata", "agent-based", "game theory",
    "AHP", "TOPSIS", "entropy weight", "grey relational",
    "fuzzy", "wavelet", "Fourier",
]

# O奖特征标签关键词
QUALITY_TAG_KEYWORDS = {
    "novel_metric": ["novel", "innovative", "new metric", "propose", "we define", "we introduce"],
    "strong_validation": ["validate", "verification", "cross-validation", "ground truth", "benchmark"],
    "comprehensive_sensitivity": ["sensitivity", "robustness", "perturbation", "stress test"],
    "clear_framework": ["framework", "flowchart", "pipeline", "architecture", "workflow"],
    "strong_visualization": ["figure", "plot", "chart", "diagram", "heatmap", "visualization"],
    "rigorous_math": ["theorem", "proof", "lemma", "corollary", "convergence"],
}


def detect_problem_type(text: str) -> str:
    """从文本中推断问题类型"""
    text_lower = text.lower()
    scores = {}
    for ptype, keywords in PROBLEM_TYPE_KEYWORDS.items():
        scores[ptype] = sum(1 for kw in keywords if kw.lower() in text_lower)
    if max(scores.values()) == 0:
        return "general"
    return max(scores, key=scores.get)


def extract_techniques(text: str) -> list[str]:
    """从文本中提取使用的技术/算法"""
    found = []
    text_lower = text.lower()
    for tech in TECHNIQUE_KEYWORDS:
        if tech.lower() in text_lower:
            found.append(tech)
    return list(set(found))


def extract_quality_tags(text: str) -> list[str]:
    """提取 O 奖特征标签"""
    tags = []
    text_lower = text.lower()
    for tag, keywords in QUALITY_TAG_KEYWORDS.items():
        if any(kw.lower() in text_lower for kw in keywords):
            tags.append(tag)
    return tags


def create_chunks(paper: dict, coarse_size: int = 1500, coarse_overlap: int = 150,
                  fine_size: int = 300, fine_overlap: int = 30) -> list[dict]:
    """
    对解析后的论文进行双层分块

    Args:
        paper: parse_pdf() 的输出
        coarse_size: 粗粒度块大小 (字符数，约等于 token 数)
        coarse_overlap: 粗粒度块重叠
        fine_size: 细粒度块大小
        fine_overlap: 细粒度块重叠

    Returns:
        chunk 列表，每个 chunk 包含文本和元数据
    """
    paper_id = paper["paper_id"]
    year = paper["year"]

    # 从全文推断 problem_type
    problem_type = detect_problem_type(paper["full_text"])

    coarse_splitter = RecursiveCharacterTextSplitter(
        chunk_size=coarse_size,
        chunk_overlap=coarse_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )

    fine_splitter = RecursiveCharacterTextSplitter(
        chunk_size=fine_size,
        chunk_overlap=fine_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )

    all_chunks = []
    chunk_counter = {"coarse": 0, "fine": 0}

    for section in paper["sections"]:
        section_type = section["type"]
        section_text = section["text"]
        page_range = section.get("page_range", [0, 0])

        if not section_text.strip():
            continue

        # 粗粒度分块
        coarse_texts = coarse_splitter.split_text(section_text)
        for ct in coarse_texts:
            chunk_counter["coarse"] += 1
            all_chunks.append({
                "chunk_id": f"{paper_id}_{section_type}_coarse_{chunk_counter['coarse']:03d}",
                "paper_id": paper_id,
                "section_type": section_type,
                "year": year,
                "problem_type": problem_type,
                "techniques": extract_techniques(ct),
                "quality_tags": extract_quality_tags(ct),
                "granularity": "coarse",
                "page_range": page_range,
                "text": ct,
            })

        # 细粒度分块
        fine_texts = fine_splitter.split_text(section_text)
        for ft in fine_texts:
            chunk_counter["fine"] += 1
            all_chunks.append({
                "chunk_id": f"{paper_id}_{section_type}_fine_{chunk_counter['fine']:03d}",
                "paper_id": paper_id,
                "section_type": section_type,
                "year": year,
                "problem_type": problem_type,
                "techniques": extract_techniques(ft),
                "quality_tags": extract_quality_tags(ft),
                "granularity": "fine",
                "page_range": page_range,
                "text": ft,
            })

    return all_chunks


def chunk_papers(papers: list[dict], **kwargs) -> tuple[list[dict], list[dict]]:
    """
    批量分块，返回粗粒度和细粒度两组 chunks

    Returns:
        (coarse_chunks, fine_chunks)
    """
    coarse_chunks = []
    fine_chunks = []

    for paper in papers:
        chunks = create_chunks(paper, **kwargs)
        for chunk in chunks:
            if chunk["granularity"] == "coarse":
                coarse_chunks.append(chunk)
            else:
                fine_chunks.append(chunk)

    print(f"分块完成: 粗粒度 {len(coarse_chunks)} 块, 细粒度 {len(fine_chunks)} 块")
    return coarse_chunks, fine_chunks
