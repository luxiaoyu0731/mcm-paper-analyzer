"""
Embedding 封装模块
- 支持 sentence-transformers 模型
- 支持中英双语 (bge-large-zh-v1.5)
- 批量向量化
"""

import numpy as np
from typing import Optional


class Embedder:
    """Embedding 模型封装"""

    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5",
                 device: str = "cpu", batch_size: int = 32):
        """
        初始化 Embedding 模型

        Args:
            model_name: sentence-transformers 模型名
            device: 运行设备 (cpu/cuda/mps)
            batch_size: 批量编码大小
        """
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device

        print(f"加载 Embedding 模型: {model_name} (device={device})")
        self.model = SentenceTransformer(model_name, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Embedding 维度: {self.dimension}")

    def encode(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """
        将文本列表编码为向量

        Args:
            texts: 文本列表
            show_progress: 是否显示进度条

        Returns:
            numpy 数组，shape = (len(texts), dimension)
        """
        if not texts:
            return np.array([])

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2 归一化，便于余弦相似度计算
        )
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """编码单个文本"""
        return self.encode([text], show_progress=False)[0]
