# 贡献指南

先说明问题与可复现输入，提交小范围修改，附对应测试结果。不要上传用户文件、论文、密钥、模型权重或环境目录。文档修正与代码修正分清范围。对外部模型行为请使用离线 mock；需要收费实验时先与维护者确认。

提交 PR 前运行 README 中的相关检查，并列出未运行项目。

## 从一个小任务开始 / Starter tasks

先在 Issue 中说明选择的任务和复现方法，再提交最小修改。测试只使用合成资料；不把外部模型收费作为贡献前提。

- **无检索结果时的报告提示** — `mcm-analyzer/analysis/report_builder.py`。验收：没有参考片段时明确提示，不编造引用；使用固定替身测试，不调用模型。
- **检索重复项的边界测试** — `tests/test_retriever.py`。验收：同一 chunk 重复召回只出现一次，存储行不变；无需向量模型下载。

[报告问题 / Report a bug](https://github.com/luxiaoyu0731/mcm-paper-analyzer/issues/new?template=bug_report.yml) · [首次使用反馈 / First-use feedback](https://github.com/luxiaoyu0731/mcm-paper-analyzer/issues/new?template=first_use.yml)
