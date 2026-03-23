"""
论文诊断主流程 v4
- 题号后过滤（同C题优先）
- 论文元信息注入
- 反幻觉预理解
"""

import json
from typing import Optional
from openai import OpenAI
from core.chunker import detect_problem_type

from analysis.prompts import (
    SECTION_DIAGNOSIS_PROMPT,
    FINAL_REPORT_PROMPT,
    QUICK_CHECK_PROMPT,
    COMPARISON_PROMPT,
    PATTERN_SUMMARY_PROMPT,
)


class PaperAnalyzer:

    def __init__(self, retriever, gene_profile: dict,
                 llm_client: OpenAI = None, model: str = "openai/gpt-4o",
                 max_context_tokens: int = 60000, use_cli: bool = False):
        self.retriever = retriever
        self.gene_profile = gene_profile
        self.llm_client = llm_client
        self.model = model
        self.use_cli = use_cli
        self.max_context_tokens = max_context_tokens

    def _build_paper_meta(self, paper: dict) -> str:
        """构建论文元信息"""
        meta = paper.get("metadata", {})
        problem = meta.get("problem", "Unknown")
        year = meta.get("year", 0)
        word_count = meta.get("word_count", 0)
        total_images = meta.get("total_images", 0)
        figure_captions = meta.get("figure_captions", [])
        table_captions = meta.get("table_captions", [])

        text_sample = paper.get("full_text", "")[:500]
        chinese_chars = sum(1 for c in text_sample if '\u4e00' <= c <= '\u9fff')
        language = "中文" if chinese_chars > len(text_sample) * 0.1 else "英文"

        problem_type = detect_problem_type(paper.get("full_text", ""))

        self._current_problem_type = problem_type
        self._current_problem_letter = problem
        self._current_language = language

        sections_found = [s["type"] for s in paper.get("sections", [])]
        section_types = list(dict.fromkeys(sections_found))

        lines = [
            f"- **题号**: Problem {problem}" if problem != "Unknown" else "- **题号**: 未识别",
            f"- **年份**: {year if year else '未知'}",
            f"- **论文语言**: {language}（Before/After改写必须用{language}）",
            f"- **题型**: {problem_type}",
            f"- **总词数**: {word_count:,}",
            f"- **图片数**: {total_images}",
            f"- **Figure标题**: {'; '.join(figure_captions[:8])}{'...' if len(figure_captions) > 8 else ''}",
            f"- **Table标题**: {'; '.join(table_captions[:8])}{'...' if len(table_captions) > 8 else ''}",
            f"- **检测到的章节**: {', '.join(section_types)}",
        ]
        return "\n".join(lines)

    def analyze(self, paper_path: str, mode: str = "full",
                section_filter: Optional[str] = None,
                compare: bool = False) -> str:
        print(f"解析用户论文: {paper_path}")
        from core.parser import parse_pdf
        paper = parse_pdf(paper_path, llm_client=self.llm_client,
                          vision_model=self.model)

        self._paper_meta = self._build_paper_meta(paper)
        print(f"论文元信息:\n{self._paper_meta}")

        if mode == "quick":
            return self._quick_check(paper)
        elif mode == "section":
            return self._analyze_section(paper, section_filter or "modeling")
        else:
            report = self._full_analysis(paper)
            if compare:
                report += "\n\n---\n\n" + self._comparison_analysis(paper)
            return report

    def _full_analysis(self, paper: dict) -> str:
        print("开始完整诊断...")
        section_diagnoses = {}
        analyzable_types = ["abstract", "intro", "assumption", "modeling",
                           "validation", "sensitivity", "visualization", "conclusion"]

        merged_sections = {}
        for section in paper["sections"]:
            stype = section["type"]
            if stype not in analyzable_types or not section["text"].strip():
                continue
            if stype not in merged_sections:
                merged_sections[stype] = section["text"]
            else:
                merged_sections[stype] += "\n\n" + section["text"]

        for stype in analyzable_types:
            if stype not in merged_sections:
                continue
            merged_text = merged_sections[stype]
            print(f"  诊断 [{stype}] (共 {len(merged_text)} 字符) ...", flush=True)
            diagnosis = self._diagnose_section(merged_text, stype, paper)
            section_diagnoses[stype] = diagnosis

        print("  汇总报告...", flush=True)
        return self._generate_final_report(section_diagnoses)

    def _multi_query_retrieve(self, section_text: str, section_type: str,
                              top_k: int = 5) -> list[dict]:
        QUERY_CHUNK_SIZE = 2000
        MAX_QUERY_SEGMENTS = 6
        all_results = []
        seen_ids = set()

        all_segments = []
        for i in range(0, len(section_text), QUERY_CHUNK_SIZE):
            seg = section_text[i:i + QUERY_CHUNK_SIZE]
            if len(seg.strip()) > 100:
                all_segments.append(seg)

        if len(all_segments) > MAX_QUERY_SEGMENTS:
            step = len(all_segments) / MAX_QUERY_SEGMENTS
            segments = [all_segments[int(i * step)] for i in range(MAX_QUERY_SEGMENTS)]
        else:
            segments = all_segments

        problem_type = getattr(self, '_current_problem_type', None)
        problem_letter = getattr(self, '_current_problem_letter', None)

        for seg in segments:
            results = self.retriever.retrieve_for_section(
                query_text=seg,
                section_type=section_type,
                problem_type=problem_type,
                problem_letter=problem_letter,
                top_k=top_k,
                granularity="fine",
            )
            for r in results:
                if r["chunk_id"] not in seen_ids:
                    seen_ids.add(r["chunk_id"])
                    all_results.append(r)

        all_results.sort(key=lambda x: x.get("distance", 1.0))
        return all_results[:top_k]

    def _format_must_have(self, section_type: str) -> str:
        """只提取 must_have 列表，精简传入"""
        gene_section = self.gene_profile.get("sections", {}).get(section_type, {})
        if not gene_section:
            return "（无基因数据）"
        must_have = gene_section.get("must_have", [])
        frequent = gene_section.get("frequent", [])
        lines = []
        for i, item in enumerate(must_have, 1):
            lines.append(f"{i}. [必备] {item}")
        for i, item in enumerate(frequent, len(must_have) + 1):
            lines.append(f"{i}. [高频但非必备] {item}")
        return "\n".join(lines) if lines else "（无基因数据）"

    def _diagnose_section(self, section_text: str, section_type: str,
                         paper: dict) -> str:
        must_have_text = self._format_must_have(section_type)

        results = self._multi_query_retrieve(section_text, section_type, top_k=5)
        retrieved_text = self.retriever.format_retrieval_context(results)

        # 超长章节 (>20000字符) 分段诊断
        LONG_THRESHOLD = 20000
        if len(section_text) > LONG_THRESHOLD:
            midpoint = len(section_text) // 2
            # 在中点附近找段落分界
            split_pos = section_text.rfind('\n\n', midpoint - 2000, midpoint + 2000)
            if split_pos == -1:
                split_pos = midpoint
            part1 = section_text[:split_pos]
            part2 = section_text[split_pos:]

            print(f"    → 超长章节分段: Part1={len(part1)}字, Part2={len(part2)}字", flush=True)

            prompt1 = SECTION_DIAGNOSIS_PROMPT.format(
                section_type=section_type,
                paper_meta=self._paper_meta,
                must_have_items=must_have_text,
                retrieved_similar_chunks=retrieved_text,
                user_section_text=f"【前半部分，共{len(part1)}字】\n{part1}",
            )
            diag1 = self._call_llm(prompt1)

            prompt2 = SECTION_DIAGNOSIS_PROMPT.format(
                section_type=section_type,
                paper_meta=self._paper_meta,
                must_have_items=must_have_text,
                retrieved_similar_chunks=retrieved_text,
                user_section_text=f"【后半部分，共{len(part2)}字】\n{part2}",
            )
            diag2 = self._call_llm(prompt2)

            # 合并两段诊断
            merge_prompt = (
                f"以下是 [{section_type}] 章节的两段诊断结果。请合并成一份统一诊断，"
                f"保留所有不重复的问题和优势，统一评级（取两段中较低的），"
                f"合并量化对标表，Before/After 从两段中各取最佳的。"
                f"输出格式与原始诊断完全一致。\n\n"
                f"=== 前半部分诊断 ===\n{diag1}\n\n"
                f"=== 后半部分诊断 ===\n{diag2}"
            )
            return self._call_llm(merge_prompt, temperature=0.2)
        else:
            prompt = SECTION_DIAGNOSIS_PROMPT.format(
                section_type=section_type,
                paper_meta=self._paper_meta,
                must_have_items=must_have_text,
                retrieved_similar_chunks=retrieved_text,
                user_section_text=section_text,
            )
            return self._call_llm(prompt)

    def _generate_final_report(self, section_diagnoses: dict) -> str:
        all_diagnoses = ""
        for stype, diagnosis in section_diagnoses.items():
            all_diagnoses += f"\n\n### [{stype}] 章节诊断\n{diagnosis}\n"

        stats = json.dumps(
            self.gene_profile.get("statistical_profile", {}),
            ensure_ascii=False, indent=2
        )

        prompt = FINAL_REPORT_PROMPT.format(
            paper_meta=self._paper_meta,
            all_section_diagnoses=all_diagnoses,
            statistical_profile=stats,
        )
        overview = self._call_llm(prompt, temperature=0.2)

        full_report = overview
        full_report += "\n\n---\n\n## 六、逐章节详细诊断（附录）\n"
        full_report += "> 以下为各章节的完整诊断原文。\n"
        for stype, diagnosis in section_diagnoses.items():
            full_report += f"\n### 📌 [{stype}] 章节\n\n{diagnosis}\n"

        return full_report

    def _quick_check(self, paper: dict) -> str:
        print("执行快速体检...")
        gene_text = json.dumps(self.gene_profile, ensure_ascii=False, indent=2)
        prompt = QUICK_CHECK_PROMPT.format(
            paper_meta=self._paper_meta,
            gene_profile=gene_text,
            paper_text=paper["full_text"],
        )
        return self._call_llm(prompt)

    def _analyze_section(self, paper: dict, section_type: str) -> str:
        merged_text = ""
        for section in paper["sections"]:
            if section["type"] == section_type:
                if merged_text:
                    merged_text += "\n\n"
                merged_text += section["text"]
        if not merged_text:
            return f"未找到 [{section_type}] 章节。检测到: {[s['type'] for s in paper['sections']]}"
        return self._diagnose_section(merged_text, section_type, paper)

    def _comparison_analysis(self, paper: dict) -> str:
        print("  生成对比分析...", flush=True)
        results = self._multi_query_retrieve(paper["full_text"], section_type=None, top_k=5)
        similar_text = self.retriever.format_retrieval_context(results)
        gene_text = json.dumps(self.gene_profile, ensure_ascii=False, indent=2)

        prompt = COMPARISON_PROMPT.format(
            paper_meta=self._paper_meta,
            user_paper=paper["full_text"],
            similar_o_award_chunks=similar_text,
            gene_profile=gene_text,
        )
        return self._call_llm(prompt)

    def summarize_patterns(self) -> str:
        print("生成 O 奖获奖基因报告...")
        all_chunks_text = ""
        for stype in ["abstract", "modeling", "validation", "visualization"]:
            chunks = self.retriever.retrieve_global(stype, top_k=10)
            for c in chunks:
                all_chunks_text += f"\n[{c['paper_id']}, {stype}]\n{c['text']}\n"

        gene_text = json.dumps(self.gene_profile, ensure_ascii=False, indent=2)
        prompt = PATTERN_SUMMARY_PROMPT.format(
            retrieved_chunks=all_chunks_text,
            gene_profile=gene_text,
        )
        return self._call_llm(prompt)

    SYSTEM_MSG = (
        '你是 MCM/ICM 资深评委（15年评审经验）。三条铁律：\n\n'
        '【铁律1：事实优先+校准评级】先做事实核查，列出论文实际有什么。'
        '事实核查确认存在的内容，后续绝不能说「缺乏」。\n'
        '评级校准（极重要）：参考片段是多篇O奖精华合集，单篇论文不可能处处达到合集最高水平。\n'
        '✅ 达标 = 完成了核心任务，有实质内容（公式/数据/图表/论证），质量达到合格MCM论文水平。'
        '例：abstract有问题陈述+方法+数值结果→✅；modeling有框架+公式+结果→✅；sensitivity测了3+参数有图表→✅\n'
        '⚠️ 需改进 = 有框架但存在明显缺陷（关键内容缺失/论证断裂/重大跳步）。'
        '例：modeling有框架但关键公式无推导→⚠️；sensitivity只测1个参数→⚠️\n'
        '❌ 不合格 = 名存实亡或严重不合格。例：只有标题没内容→❌\n'
        '校准锚点：一篇合格MCM论文通常3-5个✅和1-3个⚠️。全⚠️=评级过严，全✅=评级过松。\n\n'
        '【铁律2：Before/After 必须是结构性重写】'
        '❌ 禁止：在原句末尾追加一句话。'
        '❌ 禁止：只加 This demonstrates 或 Furthermore 之类的空话。'
        '✅ 正确：重组段落结构，补充具体技术细节/数据/论证链，字数增加50%+。'
        'Before 取2-3句原文，After 重写为5-8句，加入具体内容。\n\n'
        '【铁律3：O 奖引用必须来自参考片段】'
        '引用格式：[论文ID, 章节]。禁止写 [基因标准] 或 [O 奖基因标准]。'
        '只引用 prompt 中「参考片段」里的实际文字。\n\n'
        '诊断用中文。Before/After 用论文原文语言。'
    )

    def _call_llm(self, prompt: str, temperature: float = 0.3,
                  max_retries: int = 2) -> str:
        # 优先使用 claude CLI（Max 套餐免费），失败时回退到 API
        if self.use_cli:
            return self._call_llm_cli(prompt)
        return self._call_llm_api(prompt, temperature, max_retries)

    def _call_llm_cli(self, prompt: str) -> str:
        """通过 claude CLI 调用，使用 Max 套餐额度"""
        import subprocess, tempfile, os

        # prompt 写入临时文件，通过 stdin 传入
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False,
                                          encoding='utf-8') as f:
            f.write(prompt)
            tmp_path = f.name

        # 继承当前环境变量 + 添加 SSL bypass（解决代理证书问题）
        env = os.environ.copy()
        env['NODE_TLS_REJECT_UNAUTHORIZED'] = '0'

        try:
            with open(tmp_path, 'r', encoding='utf-8') as stdin_f:
                result = subprocess.run(
                    [
                        'claude', '-p',
                        '--model', 'sonnet',
                        '--system-prompt', self.SYSTEM_MSG,
                        '--output-format', 'text',
                    ],
                    stdin=stdin_f,
                    capture_output=True, text=True,
                    timeout=600, env=env,
                )
            os.unlink(tmp_path)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            else:
                err = result.stderr[:500] if result.stderr else "无输出"
                print(f"    ⚠️ CLI 调用失败: {err}", flush=True)
                return f"❌ CLI 调用失败: {err}"
        except subprocess.TimeoutExpired:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return "❌ CLI 调用超时 (600s)"
        except FileNotFoundError:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            print("    ⚠️ claude CLI 未找到，回退到 API", flush=True)
            self.use_cli = False
            return self._call_llm_api(prompt)
        except Exception as e:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return f"❌ CLI 调用异常: {e}"

    def _call_llm_api(self, prompt: str, temperature: float = 0.3,
                      max_retries: int = 2) -> str:
        """通过 OpenRouter API 调用（需要余额）"""
        import time
        for attempt in range(max_retries + 1):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_MSG},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=16384,
                    timeout=300,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < max_retries:
                    wait = 10 * (attempt + 1)
                    print(f"    ⚠️ API 调用失败 (尝试 {attempt+1}/{max_retries+1}): {e}, {wait}s 后重试...", flush=True)
                    time.sleep(wait)
                else:
                    return f"❌ API 调用失败 ({max_retries+1}次尝试后): {e}"
