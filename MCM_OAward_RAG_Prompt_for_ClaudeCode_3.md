# MCM/ICM O奖论文分析系统 — Claude Code 工程 Prompt

> **用途**：交给 Claude Code，让它构建一个基于 RAG（检索增强生成）的本地分析系统，能吃进近十年美赛 O 奖论文、提炼获奖基因、并对用户作品给出精准修改意见。

---

## 第一步：角色定义（Role）

```
你是一位具备 20 年 MCM/ICM 评审经验的资深评委兼学术优化教练，同时也是一位精通 RAG 系统架构的全栈工程师。

你的双重身份：
1. **学术侧**：你深谙 O 奖（Outstanding Winner）论文的评判标准，能从数十篇获奖论文中提炼共性"获奖基因"（Success DNA），并能对学生草稿进行逐章节的手术刀式诊断。
2. **工程侧**：你精通用 Python 构建本地 RAG pipeline——包括文档解析、文本分块、向量化、向量数据库、检索策略、以及基于检索结果的增强生成。

你的核心原则：
- 所有分析结论必须"有据可查"——即能追溯到具体某篇 O 奖论文的具体章节
- 修改建议必须"可操作"——不说空话，给出具体的改写示例、图表建议、结构调整方案
- 系统必须"可增量"——新论文加入后无需重建整个知识库
```

---

## 第二步：任务拆解（Task Decomposition）

整个系统分为两个阶段、六个子任务：

### 阶段 A：知识库构建（离线，只做一次）

```
任务 A1 — 文档解析与清洗
- 输入：O 奖论文 PDF 文件夹（路径由用户指定）
- 处理：
  · 使用 PyMuPDF (fitz) 提取文本，对扫描件 fallback 到 pytesseract OCR
  · 按论文的自然结构分段：Abstract / Introduction / Model / Analysis / Sensitivity / Conclusion
  · 清洗：去页眉页脚、去乱码、合并跨页段落
- 输出：每篇论文一个结构化 JSON，字段包括：
  {
    "paper_id": "2024_A_Stanford",
    "year": 2024,
    "problem": "A",
    "sections": [
      {"type": "abstract", "text": "...", "page_range": [1,1]},
      {"type": "modeling", "text": "...", "page_range": [3,8]},
      ...
    ],
    "metadata": {"team": "...", "word_count": 8500}
  }

任务 A2 — 智能分块（Chunking）
- 策略：采用"双层分块"
  · 粗粒度块（~1500 tokens）：按自然章节切分，保留完整语义单元
  · 细粒度块（~300 tokens）：在粗粒度块内按段落/子小节切分，用于精准检索
- 每个 chunk 附加元数据标签：
  {
    "chunk_id": "2024_A_Stanford_modeling_03",
    "paper_id": "2024_A_Stanford",
    "section_type": "modeling",     # enum: abstract/intro/assumption/modeling/validation/sensitivity/visualization/conclusion
    "year": 2024,
    "problem_type": "continuous",   # enum: continuous/discrete/optimization/network/data_driven
    "techniques": ["PDE", "FEM", "sensitivity_analysis"],  # 从内容中自动提取
    "quality_tags": ["novel_metric", "strong_validation"],  # O 奖特征标签
    "granularity": "coarse"         # coarse / fine
  }

任务 A3 — 向量化与入库
- Embedding 模型：使用 sentence-transformers（推荐 all-MiniLM-L6-v2 或 bge-large-zh-v1.5，视中英文比例定）
- 向量数据库：使用 ChromaDB（本地持久化，零外部依赖）
- 建立两个 Collection：
  · `o_award_coarse`：粗粒度块，用于全局模式总结
  · `o_award_fine`：细粒度块，用于精准对标检索

任务 A4 — O 奖获奖基因提炼（离线总结）
- 对知识库做一次全局扫描，生成"O 奖基因图谱"并持久化为 JSON：
  {
    "abstract_patterns": {
      "must_have": ["明确模型名称", "引用关键数值结果", "Problem→Method→Result→Conclusion 四段式"],
      "frequent": ["一句话概括创新点", "提及验证方法"],
      "examples": [{"paper": "2023_C_MIT", "quote": "...", "analysis": "..."}]
    },
    "modeling_patterns": { ... },
    "visualization_patterns": {
      "type_distribution": {"framework_flowchart": 95%, "mechanism_diagram": 87%, "EDA_plot": 78%, ...},
      "avg_figure_count": 12.3,
      "best_practices": [...]
    },
    "validation_patterns": { ... },
    "writing_patterns": { ... },
    "statistical_profile": {
      "avg_page_count": 24.5,
      "avg_reference_count": 18,
      "avg_assumption_count": 5.2,
      "model_type_distribution": {...}
    }
  }
- 这份基因图谱是后续所有分析的"黄金标准"参照物
```

### 阶段 B：在线分析（每次用户提交时执行）

```
任务 B1 — 用户论文解析与对标检索
- 解析用户论文（同 A1 流程）
- 对用户论文的每个章节，在向量库中做"多路召回"：
  · 语义检索：用章节文本 embedding 检索相似的 O 奖论文片段（top-5）
  · 元数据过滤：优先召回同 problem_type、同 section_type 的 chunk
  · 关键词增强：提取用户使用的模型名/算法名，做精确匹配增强
- 召回结果去重、按相关度排序，附带来源追溯信息

任务 B2 — 增强生成（RAG 核心推理）
- 将以下内容组装为最终 prompt 发送给 LLM：
  ┌─────────────────────────────────────────────┐
  │  System: O奖评委角色设定 + 评分框架          │
  │  Context 1: O奖基因图谱（A4 产出的 JSON）    │
  │  Context 2: 检索到的 O奖论文相关片段          │
  │  Context 3: 用户论文的完整结构化文本          │
  │  Instruction: 具体分析指令（见下方模板）       │
  └─────────────────────────────────────────────┘
```

---

## 第三步：知识架构与检索策略（RAG Core）

```
### 检索策略设计

采用"分层检索 + 多路融合"架构：

1. **全局层（Global）**：
   - 触发条件：用户要求"总结 O 奖共性" / "获奖基因是什么"
   - 检索方式：从 o_award_coarse 中按 section_type 分组检索
   - 后处理：Map-Reduce 摘要——先对每个 section_type 的 top-K 做小结，再汇总

2. **章节层（Section）**：
   - 触发条件：逐章节对比诊断
   - 检索方式：用用户论文的某个章节文本检索 o_award_fine 中同 section_type 的 chunk
   - 后处理：挑选最相关的 3-5 个 chunk 作为"对标样本"

3. **技术层（Technique）**：
   - 触发条件：用户使用了特定模型/算法（如 LSTM、蒙特卡洛）
   - 检索方式：元数据过滤 techniques 字段 + 语义检索
   - 后处理：展示 O 奖论文中如何使用同类技术

### 检索质量保障

- Chunk overlap：相邻块重叠 10%，防止语义断裂
- Re-ranking：对初步召回结果用 Cross-Encoder 重排序（如 bge-reranker-base）
- 来源追溯：每条检索结果必须携带 paper_id + section_type + page_range，
  最终输出中以脚注形式呈现，如 [参见 2023_C_MIT, Modeling, p.5-7]
```

---

## 第四步：输入输出规范（I/O Specification）

### 输入格式

```
系统接受两种输入模式：

模式 1 — 知识库构建（批量导入）
  命令：`python main.py ingest --dir ./o_award_papers/`
  输入：包含 O 奖论文 PDF 的文件夹
  预期：解析 → 分块 → 向量化 → 入库 → 生成基因图谱

模式 2 — 论文诊断（单篇分析）
  命令：`python main.py analyze --paper ./my_draft.pdf [--mode full|section|quick]`
  输入：用户论文 PDF
  可选参数：
    --mode full      完整诊断（默认，最详细）
    --mode section   仅分析指定章节（配合 --section modeling）
    --mode quick     快速体检（只出 checklist 和总评）
    --compare        同时输出与最相似 O 奖论文的逐项对比
```

### 输出格式

```
所有输出采用以下标准化结构（Markdown 格式）：

# 📊 MCM/ICM 论文诊断报告

## 一、总评概览
- **综合评级**：A / B+ / B / C+（对标 O 奖标准）
- **核心优势**：（1-3 句话）
- **致命短板**：（1-3 句话）
- **与 O 奖差距**：整体距离评估 + 雷达图数据（5 维度打分）

## 二、逐章节手术刀诊断

### 2.1 摘要（Abstract）
- **现状评估**：✅ 达标 / ⚠️ 需改进 / ❌ 严重不足
- **O 奖对标**：
  > "最相似的 O 奖论文 [2023_C_MIT] 的摘要这样写：……"（检索结果）
- **具体问题**：
  1. [问题描述] → [改写建议，给出具体文字示例]
- **优先级**：🔴 高 / 🟡 中 / 🟢 低

### 2.2 建模策略（Modeling）
（同上结构，额外包含：）
- **模型选择合理性**：与 O 奖中同类问题的模型选择对比
- **创新性评估**：是否有"Spark Point"，O 奖论文中类似创新的案例

### 2.3 可视化系统（Visualization）
- **图表清单审计**：逐一评估现有图表
- **缺失图表建议**：基于 O 奖统计数据，建议补充的图表类型
  · "O 奖论文平均含 12.3 张图，你目前只有 6 张"
  · "95% 的 O 奖论文含框架流程图（Type A），你缺少此类图"

### 2.4 验证与敏感性分析（Validation）
### 2.5 写作与排版（Writing）

## 三、O 奖基因对比矩阵
（表格：O 奖平均值 vs 你的论文，逐项对比）

## 四、优先级行动清单（Actionable To-Do）
- 🔴 [24h 内必须完成] ……
- 🟡 [建议完成] ……
- 🟢 [锦上添花] ……

## 五、参考来源
- [1] 2023_C_MIT, Modeling Section, p.5-7 — 关于 PDE 建模范式
- [2] 2024_A_Stanford, Abstract, p.1 — 摘要写作样板
（所有引用均可追溯到知识库中的具体论文和页码）
```

---

## 第五步：约束条件（Constraints）

```
### 工程约束

1. **纯本地运行**：除 LLM API 调用外，所有数据处理和向量检索在本地完成，
   不依赖任何云端 RAG 服务（Pinecone/Weaviate 等）
2. **依赖最小化**：核心依赖控制在以下范围：
   - 文档解析：PyMuPDF + pytesseract（OCR fallback）
   - 分块/嵌入：langchain-text-splitters + sentence-transformers
   - 向量库：chromadb（持久化到本地 ./chroma_db/）
   - Re-ranking：可选（FlagReranker 或 Cross-Encoder）
   - LLM 调用：openai SDK（兼容任意 OpenAI-compatible API）
3. **增量更新**：新论文入库不需重建索引，支持 `ingest --append`
4. **Context Window 管理**：
   - 单次 LLM 调用的 context 控制在 30K tokens 以内
   - 超长论文采用"章节轮询"策略：逐章节分析 → 最后汇总
5. **中英文混合**：Embedding 模型需支持中英双语（论文为英文，输出为中文）

### 分析约束

1. **所有评价必须有对标依据**：不允许出现"建议改进建模"这种空话，
   必须说明"O 奖论文 [X] 在类似场景下使用了 [Y] 方法，建议参考其 p.Z 的处理方式"
2. **改写建议必须给出 Before/After 示例**：
   - Before: "We built a model to solve the problem."
   - After: "We developed a two-stage PDE-constrained optimization model, achieving a 15% improvement in prediction accuracy over baseline."
3. **不过度赞美**：O 奖评委不会说"写得很好"，而是直接指出"Abstract 缺少数值结果引用，
   这在 O 奖论文中出现率为 98%，你需要补上"
4. **统计数据必须基于实际知识库**：引用的"O 奖平均值"必须来自 A4 步骤生成的基因图谱，
   不可凭空编造
```

---

## 第六步：迭代与评估机制（Iteration）

```
### 自检流程

每次生成诊断报告后，系统自动执行以下自检：

1. **来源覆盖检查**：报告中的每条建议是否都有 [参见 ...] 标注？
   - 如果有裸建议（无来源），重新检索补充
2. **平衡性检查**：优势和不足的比例是否合理？
   - 不应全是批评（打击信心）或全是表扬（无价值）
3. **可操作性检查**：To-Do 清单中的每一项是否足够具体，能直接执行？
   - "改进摘要" ❌ → "在摘要第二句后插入最终模型预测的数值结果" ✅
4. **幻觉检查**：引用的 O 奖论文片段是否真实存在于向量库中？
   - 对每条引用做回查验证，删除无法验证的引用

### 用户反馈循环

- 用户可以对报告的某个章节追问："展开说说建模部分的问题"
- 系统此时触发更精细的检索（技术层检索），并给出更详细的对标分析
- 支持多轮对话，每轮对话的检索结果叠加到 context 中
```

---

## 附录：项目文件结构

```
mcm-analyzer/
├── main.py                  # CLI 入口（ingest / analyze / chat）
├── config.yaml              # 配置文件（API key、模型选择、路径等）
├── requirements.txt
│
├── core/
│   ├── parser.py            # PDF 解析 + OCR + 结构化
│   ├── chunker.py           # 双层分块策略
│   ├── embedder.py          # Embedding 封装（支持切换模型）
│   ├── vectorstore.py       # ChromaDB 封装（CRUD + 检索）
│   ├── retriever.py         # 多路召回 + Re-ranking
│   └── gene_extractor.py    # O 奖基因图谱提炼
│
├── analysis/
│   ├── analyzer.py          # 论文诊断主流程
│   ├── prompts.py           # 所有 Prompt 模板（见下方）
│   └── report_builder.py    # 报告格式化输出
│
├── data/
│   ├── o_award_papers/      # O 奖论文 PDF 存放目录
│   ├── chroma_db/           # ChromaDB 持久化目录
│   └── gene_profile.json    # O 奖基因图谱
│
└── tests/
    ├── test_parser.py
    ├── test_retriever.py
    └── test_analyzer.py
```

---

## 附录：prompts.py 中的 Prompt 模板

```python
# ===== 模板 1：基因提炼（用于 A4 阶段） =====
GENE_EXTRACTION_PROMPT = """
你是 MCM/ICM 资深评委。以下是从 {n} 篇 O 奖论文的 [{section_type}] 章节中检索到的代表性片段。

请分析这些片段的共性模式，输出 JSON 格式的"获奖基因"：

## 检索到的 O 奖片段
{retrieved_chunks}

## 输出要求
{{
  "must_have": ["所有 O 奖论文都具备的特征，列 3-5 条"],
  "frequent": ["多数 O 奖论文具备的特征，列 3-5 条"],
  "rare_but_impressive": ["少数论文有但非常出彩的特征"],
  "anti_patterns": ["O 奖论文中从不出现的反面模式"],
  "statistical": {{
    "avg_length_words": 数字,
    "common_techniques": ["技术名称"],
    "common_structures": ["结构模式"]
  }},
  "exemplars": [
    {{"paper_id": "...", "why": "为什么这篇的该章节特别优秀", "key_quote": "..."}}
  ]
}}
"""

# ===== 模板 2：章节诊断（用于 B2 阶段） =====
SECTION_DIAGNOSIS_PROMPT = """
你是 MCM/ICM 资深评委，正在对一篇参赛论文的 [{section_type}] 章节进行诊断。

## O 奖基因标准（该章节）
{gene_profile_for_section}

## 最相关的 O 奖论文片段（作为对标参照）
{retrieved_similar_chunks}

## 待诊断的用户论文章节
{user_section_text}

## 诊断要求
请按以下结构输出：

1. **评级**：✅ 达标 / ⚠️ 需改进 / ❌ 严重不足（对标 O 奖水准）

2. **优势**（至少找 1 个真实优势，不要凑数）

3. **问题诊断**（每个问题按此格式）：
   - 问题：[具体描述]
   - O 奖对标：[引用检索到的 O 奖片段，标明来源]
   - 改写建议：
     · Before: [用户原文]
     · After: [建议改写]
   - 优先级：🔴高 / 🟡中 / 🟢低

4. **缺失项**：O 奖基因中要求但用户论文完全没有的内容

注意：
- 每条建议必须引用至少一个 O 奖对标来源
- 不要说"建议改进"这种空话，给出具体到可以直接 copy-paste 的改写文字
- 用中文输出
"""

# ===== 模板 3：全局汇总（最终报告） =====
FINAL_REPORT_PROMPT = """
你是 MCM/ICM 资深评委。以下是对一篇参赛论文各章节的逐一诊断结果。
请将它们整合为一份完整的诊断报告。

## 各章节诊断结果
{all_section_diagnoses}

## O 奖统计画像
{statistical_profile}

## 输出要求
1. 总评概览：综合评级 + 一句话总评
2. 五维雷达图打分（1-10）：摘要质量 / 建模深度 / 可视化 / 验证严谨性 / 写作水平
3. 与 O 奖差距：对比矩阵（O 奖平均 vs 本文，逐项）
4. 优先级行动清单：合并各章节的建议，按紧急度排序
5. 参考来源汇总

用中文输出，语气严谨但有建设性（像导师而非法官）。
"""

# ===== 模板 4：O 奖共性总结（独立功能） =====
PATTERN_SUMMARY_PROMPT = """
你是 MCM/ICM 研究专家。以下是从近十年 O 奖论文知识库中检索到的片段。

## 检索结果
{retrieved_chunks}

## O 奖基因图谱
{gene_profile}

请输出一份"O 奖获奖基因报告"，包含：
1. 共性 DNA：所有 O 奖论文的共同特征（分章节总结）
2. 技术偏好：最常用的建模方法 Top 10 + 使用频率
3. 可视化标准：图表类型分布、平均数量、最佳实践
4. 写作范式：摘要结构模板、段落开头常用句式、学术表达典型用法
5. 避坑指南：O 奖论文绝不会犯的错误
6. 年度趋势：近 3 年与 5 年前的变化（如果数据足够）

用中文输出。每个结论后标注数据来源（基于 N 篇论文统计）。
"""
```

---

