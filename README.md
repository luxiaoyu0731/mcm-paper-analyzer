# MCM Paper Analyzer

用你提供的参考论文，检索相关方法与章节，生成可对照原文修改的论文诊断报告。

![论文检索概念插画](docs/media/project-hero.png)

Python · PyMuPDF · ChromaDB · RAG · Apache-2.0

## 核心功能

- 提取 PDF 文本、章节与图片，建立粗细两层检索索引。
- 按语义、章节、题型与题号召回参考内容，并去重排序。
- 结合待诊断论文生成章节分析、对标与修改建议，输出 Markdown。

## 快速开始

```sh
git clone https://github.com/luxiaoyu0731/mcm-paper-analyzer.git
cd mcm-paper-analyzer/mcm-analyzer
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp config.example.yaml config.yaml
```

在本地 `config.yaml` 填写模型接口，自备有权使用的参考 PDF：

```sh
.venv/bin/python main.py ingest --dir ./data/o_award_papers/ --no-vision
.venv/bin/python main.py analyze --paper ./my_paper.pdf --compare
```

参考论文不随仓库分发。首次使用可能下载向量模型；导入阶段提炼论文模式和诊断阶段都会调用配置的模型服务并可能收费。`--no-vision` 只跳过图片分析，不关闭文本模型调用。论文片段可能外发，配置密钥不要提交到仓库。

<details>
<summary>开发与使用边界</summary>

`mcm-analyzer/core/` 负责解析与检索，`analysis/` 负责诊断和报告。

```sh
python3 -B -m unittest discover -s tests -v
```

模型评分不是获奖概率；参考相似性也不能证明数学正确。当前测试覆盖检索一致性，完整 PDF→模型→报告流程仍需验收。详见 [代码审查](docs/code-review.md)。

[贡献指南](CONTRIBUTING.md) · [安全反馈](SECURITY.md)

</details>

[Apache-2.0](LICENSE) · [素材说明](docs/media/README.md)
