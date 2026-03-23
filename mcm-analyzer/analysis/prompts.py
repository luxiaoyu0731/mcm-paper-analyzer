"""
Prompt 模板库 v6 — 精简版
核心策略：system message 放硬约束，user prompt 放内容
"""

# ===== 模板 1：章节诊断 =====
SECTION_DIAGNOSIS_PROMPT = """对 [{section_type}] 章节进行诊断。

## 论文信息
{paper_meta}

## O 奖必备项（must_have）
{must_have_items}

## O 奖参考片段
{retrieved_similar_chunks}

## 用户论文 [{section_type}] 全文
{user_section_text}

---

请按以下格式输出：

### 0. 事实核查
逐项列出本章节实际包含的内容：
- 公式/方程：（逐一列出，如"Eq.1: Tsiolkovsky rocket equation"）
- 图表：（逐一列出，如"Figure 3: Unit cost decline"）
- 方法/技术：（逐一列出）
- 数值结果：（逐一列出）
- 篇幅：约 X 字

### 1. 评级
评级标准（已校准，参考片段是多篇O奖精华合集，不要求单篇论文达到合集最高水平）：
- ✅ 达标 = 完成核心任务，有实质内容（公式/数据/图表/论证），must_have 基本满足
  例：abstract有问题+方法+数值→✅；modeling有框架+公式+结果→✅；sensitivity测3+参数有图表→✅
- ⚠️ 需改进 = 有框架但存在明显缺陷（关键推导缺失/论证断裂/内容过薄）
  例：modeling有框架但关键公式无推导→⚠️；sensitivity只测1参数→⚠️
- ❌ 不合格 = 名存实亡或严重不合格
  例：只有标题没内容→❌；假设无任何解释→❌

校准锚点：合格MCM论文通常3-5个✅+1-3个⚠️。全⚠️说明你评级过严。
格式：**评级：[符号]** + 逐项评估（必备项✓/✗ + 高频项✓/✗）

### 2. 量化对标
| 维度 | O 奖做法 | 用户论文（引用原文） | 差距 |
（≥5行）

### 3. 优势（2-3条，引用原文）

### 4. 问题（≥3个）
每个问题：
- 严重程度：🔴/🟡/🟢
- 用户原文：> "..."
- O 奖原文：> "..." [论文ID, 章节]
- Before → After（结构性重写，不是追加）

### 5. 缺失的 must_have 项
"""

# system message 放在 analyzer.py 的 _call_llm 里

# ===== 模板 2：全局汇总 =====
FINAL_REPORT_PROMPT = """将以下逐章诊断忠实汇总为报告。

## 论文信息
{paper_meta}

## 各章节诊断
{all_section_diagnoses}

## 统计画像
{statistical_profile}

---

输出格式：

# 📊 MCM/ICM 论文诊断报告

## 一、总评概览
统计各章评级：✅×? + ⚠️×? + ❌×?
按规则映射：
- ≥2个❌ → C
- 1个❌ → C+
- 全⚠️无✅ → B-
- ✅⚠️混合 → B（✅>⚠️则B+）
- ✅≥5且⚠️≤1 → A
- 全✅ → A+

### 五维雷达图（✅=7-9分，⚠️=5-6分，❌=2-4分）
| 维度 | 得分 | O奖基准 | 依据 |

## 二、逐章节（评级+核心问题+改进）

## 三、优先级行动清单（≥5条，🔴🟡🟢）

## 四、最佳 5 个 Before → After
从逐章中选结构性重写最好的5个。
"""


# ===== 模板 3：快速体检 =====
QUICK_CHECK_PROMPT = """快速体检。

## 论文信息
{paper_meta}

## O 奖基因
{gene_profile}

## 论文全文
{paper_text}

---

# 📋 快速体检

## Must-Have 检查（逐项 ✅/❌ + 证据）
- 摘要含数值结果：
- 摘要 ≤1页：
- 假设 ≥5个且有解释：
- Notation表：
- 模型Flowchart：
- 灵敏度分析独立章节：
- 灵敏度 ≥3参数：
- 图表有编号+标题+轴标签：
- 参考文献 ≥10篇：

## 评级 + 总评

## Top 5 改进
"""


# ===== 模板 4：O 奖共性总结 =====
PATTERN_SUMMARY_PROMPT = """基于知识库生成获奖基因报告。

## 检索结果
{retrieved_chunks}

## 基因图谱
{gene_profile}

输出：
1. 共性DNA（分章节，附数据）
2. 技术Top15
3. 可视化标准
4. 写作范式
5. 避坑指南（10个）
6. 年度趋势
"""


# ===== 模板 5：对比分析 =====
COMPARISON_PROMPT = """全局对标分析。

## 论文信息
{paper_meta}

## 用户论文全文
{user_paper}

## O 奖片段（⭐=同题号）
{similar_o_award_chunks}

## O 奖基因标准
{gene_profile}

---

# 📊 全局对标

## 1. 架构对比（结构、篇幅、逻辑链）

## 2. 最值得借鉴的 3 个 O 奖写法
（引用 O 奖原文 [论文ID, 章节]）

## 3. 用户论文独有优势

## 4. 3 个 Before → After（跨章节，结构性重写）
"""
