# 🏆 MCM/ICM O-Award Paper Analyzer

基于 RAG（检索增强生成）的美赛论文智能诊断系统。通过提取近 5 年 O 奖论文的"获奖基因"，对你的论文进行逐章节深度诊断，给出评级、量化对标和结构性改写建议。

## ✨ 核心功能

- **O 奖基因提取** — 自动解析 161 篇 O 奖论文，提炼出 8 个章节维度的评判标准
- **RAG 智能检索** — 双层向量化（粗/细粒度）+ 多路召回 + 同题号优先
- **逐章节诊断** — 事实核查 → 评级(✅/⚠️/❌) → 量化对标 → Before/After 改写
- **GPT-4o 视觉分析** — 提取论文图表并用视觉模型分析图表质量
- **Claude CLI 集成** — 支持通过 Claude Code Max 套餐零成本调用
- **多模型支持** — OpenRouter / OpenAI / DeepSeek / Claude 任选

## 📐 系统架构

```
PDF 论文 ──→ 解析器 (PyMuPDF) ──→ 分块器 (粗1500/细300 tokens)
                                         │
                                         ▼
                                   Embedding (bge-large-zh-v1.5)
                                         │
                                         ▼
                                   ChromaDB 向量库 ←── O奖论文 ×161
                                         │
                                         ▼
                              多路检索 (语义+章节+题型+题号)
                                         │
                                         ▼
                              LLM 诊断 (Claude/GPT-4o)
                                         │
                                         ▼
                              📊 诊断报告 (Markdown)
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd mcm-analyzer
pip install -r requirements.txt
```

### 2. 配置

```bash
# 复制配置模板
cp config.example.yaml config.yaml

# 编辑 config.yaml，填入你的 API Key
# 支持 OpenRouter / OpenAI / DeepSeek 等
```

**两种 LLM 调用方式：**

| 方式 | 配置 | 费用 |
|------|------|------|
| API 调用 | 在 `config.yaml` 填入 API Key | 按 token 计费 |
| Claude CLI | 安装 Claude Code + Max 套餐 | 包含在订阅中 |

### 3. 导入 O 奖论文

```bash
# 将 O 奖论文 PDF 放入 data/o_award_papers/ 目录
# 按年份/题号组织: data/o_award_papers/2025/C/2503389.pdf

# 导入（含 GPT-4o 图片分析）
python main.py ingest --dir ./data/o_award_papers/

# 跳过图片分析（更快，免费）
python main.py ingest --dir ./data/o_award_papers/ --no-vision
```

### 4. 分析你的论文

```bash
# 完整诊断（推荐）
python main.py analyze --paper ./my_paper.pdf --compare

# 快速体检
python main.py analyze --paper ./my_paper.pdf --mode quick

# 单章节深入分析
python main.py analyze --paper ./my_paper.pdf --mode section --section modeling
```

## 📂 项目结构

```
mcm-analyzer/
├── main.py                 # CLI 入口
├── config.yaml             # 配置文件 (需自行创建)
├── config.example.yaml     # 配置模板
├── requirements.txt        # Python 依赖
├── core/
│   ├── parser.py           # PDF 解析 + 图片提取 + GPT-4o 视觉分析
│   ├── chunker.py          # 双层分块 (粗1500/细300 tokens)
│   ├── embedder.py         # 向量编码 (bge-large-zh-v1.5)
│   ├── vectorstore.py      # ChromaDB 向量数据库
│   ├── retriever.py        # 多路召回检索器
│   └── gene_extractor.py   # O奖基因提取 (跨论文统计+LLM提炼)
├── analysis/
│   ├── analyzer.py         # 诊断主流程 (API + CLI 双模式)
│   ├── prompts.py          # Prompt 模板库 v6
│   └── report_builder.py   # 报告生成
└── data/                   # 数据目录 (不含在仓库中)
    ├── chroma_db/          # 向量数据库
    ├── o_award_papers/     # O奖论文 PDF
    ├── parsed_papers/      # 解析缓存
    └── gene_profile.json   # 提取的O奖基因图谱
```

## 🧬 O 奖基因图谱

系统从 161 篇 O 奖论文中提取 8 个维度的评判标准：

| 维度 | must_have (必备) | frequent (高频) | 示例 |
|------|-----------------|-----------------|------|
| Abstract | 问题陈述+方法+数值结果 | 多模型比较、敏感性分析 | "80.3% cost reduction" |
| Modeling | 清晰框架+数学推导+公式 | ODE、回归、时间序列 | 29 个方程 |
| Sensitivity | 参数变化分析+图表展示 | ≥3参数、Monte Carlo | tornado diagram |
| Visualization | 标签清晰+数据驱动+图文结合 | 多种图表类型 | 雷达图+热力图 |

## 📊 诊断报告示例

```
# 📊 MCM/ICM 论文诊断报告

## 总评概览
✅×3 + ⚠️×3 + ❌×0 → B+

### 五维雷达图
| 维度 | 得分 | O奖基准 | 依据 |
|------|------|---------|------|
| 创新性 | 6.5 | 8.5 | TGC框架有亮点... |
| 严谨性 | 5.0 | 9.0 | 推导跳步严重... |

## 逐章节诊断
### [modeling] ⚠️
- 问题：优化方法过于简化 (grid search)
- Before: "The solution can be obtained through direct enumeration..."
- After: "We employ a multi-stage hybrid optimization approach..."
```

## ⚙️ 高级配置

### Claude CLI 模式（推荐，Max 套餐用户）

```python
# 在代码中使用
analyzer = PaperAnalyzer(retriever=retriever, gene_profile=gene, use_cli=True)
```

### 切换模型

在 `config.yaml` 中修改 `model` 字段：

```yaml
model: "anthropic/claude-sonnet-4"   # Claude Sonnet (推荐)
model: "openai/gpt-4o"               # GPT-4o
model: "deepseek/deepseek-chat"      # DeepSeek (最便宜)
```

## 📋 依赖项

| 依赖 | 用途 | 是否必需 |
|------|------|---------|
| PyMuPDF | PDF 解析 | ✅ |
| sentence-transformers | 本地向量编码 | ✅ |
| chromadb | 向量数据库 | ✅ |
| openai | API 调用 (OpenAI 兼容) | API 模式需要 |
| pyyaml | 配置文件 | ✅ |
| click | CLI 框架 | ✅ |
| rich | 终端美化 | ✅ |
| langchain-text-splitters | 文本分块 | ✅ |

## 🔒 隐私说明

- 所有论文解析和向量化在**本地**完成
- 仅诊断阶段调用外部 LLM API
- API Key 存储在本地 `config.yaml`（已加入 .gitignore）
- 使用 Claude CLI 模式可完全避免 API Key 泄露风险

## 📄 License

MIT
