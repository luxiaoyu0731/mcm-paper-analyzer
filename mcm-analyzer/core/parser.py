"""
PDF 解析与清洗模块
- 使用 PyMuPDF (fitz) 提取文本
- 检测并注入图片/图表占位信息
- 过滤目录页避免重复章节
- 对扫描件 fallback 到 pytesseract OCR
- 按论文自然结构分段
- 清洗：去页眉页脚、去乱码、合并跨页段落
"""

import os
import re
import json
import base64
import fitz  # PyMuPDF
from typing import Optional
from pathlib import Path

# OCR fallback
try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False


def extract_page_images(page, doc, max_images: int = 5,
                        min_size: int = 5000) -> list[dict]:
    """
    从 PDF 页面提取嵌入图片的原始字节

    Args:
        page: fitz.Page 对象
        doc: fitz.Document 对象
        max_images: 每页最多提取图片数
        min_size: 最小图片字节数（过滤小图标/装饰图）

    Returns:
        [{"image_bytes": bytes, "ext": "png", "width": int, "height": int}, ...]
    """
    images = page.get_images(full=True)
    extracted = []

    for img_info in images[:max_images]:
        xref = img_info[0]
        try:
            pix = fitz.Pixmap(doc, xref)
            # 转换 CMYK 到 RGB
            if pix.n > 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)

            img_bytes = pix.tobytes("png")
            if len(img_bytes) < min_size:
                continue

            extracted.append({
                "image_bytes": img_bytes,
                "ext": "png",
                "width": pix.width,
                "height": pix.height,
            })
            pix = None  # 释放内存
        except Exception:
            continue

    return extracted


def describe_images_with_vision(images: list[dict], captions: list[str],
                                llm_client, model: str = "openai/gpt-4o",
                                page_num: int = 0) -> list[str]:
    """
    调用 GPT-4o Vision API 分析图片内容

    Args:
        images: extract_page_images() 的输出
        captions: 对应页面的 figure/table captions
        llm_client: OpenAI 兼容客户端
        model: 视觉模型名称
        page_num: 页码（用于上下文）

    Returns:
        每张图的文字描述列表
    """
    if not images or not llm_client:
        return []

    descriptions = []
    caption_context = "\n".join(captions) if captions else "No caption found"

    for i, img in enumerate(images):
        b64 = base64.b64encode(img["image_bytes"]).decode("utf-8")

        try:
            response = llm_client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"This is an image from page {page_num} of an MCM/ICM "
                                f"(Mathematical Contest in Modeling) competition paper. "
                                f"Nearby captions: {caption_context}\n\n"
                                f"Please describe this figure/table in detail for academic review:\n"
                                f"1. What type of visualization is this? (chart, diagram, flowchart, map, table, etc.)\n"
                                f"2. What data or concept does it present?\n"
                                f"3. Key observations (trends, patterns, notable values)\n"
                                f"4. Quality assessment: color scheme, labeling, readability, professionalism\n"
                                f"Answer in English, be concise but thorough (100-200 words)."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                                "detail": "low",  # 用 low 节省 token
                            },
                        },
                    ],
                }],
                max_tokens=300,
                temperature=0.3,
            )
            desc = response.choices[0].message.content.strip()
            descriptions.append(desc)
        except Exception as e:
            descriptions.append(f"[Vision analysis failed: {e}]")

    return descriptions


# 章节类型关键词映射
SECTION_PATTERNS = {
    "abstract": [
        r"(?i)^#{0,3}\s*abstract",
        r"(?i)^#{0,3}\s*summary",
    ],
    "intro": [
        r"(?i)^#{0,3}\s*\d*\.?\s*introduction",
        r"(?i)^#{0,3}\s*\d*\.?\s*background",
        r"(?i)^#{0,3}\s*\d*\.?\s*problem\s*(re)?statement",
    ],
    "assumption": [
        r"(?i)^#{0,3}\s*\d*\.?\s*assumptions?",
        r"(?i)^#{0,3}\s*\d*\.?\s*notation",
        r"(?i)^#{0,3}\s*\d*\.?\s*nomenclature",
    ],
    "modeling": [
        r"(?i)^#{0,3}\s*\d*\.?\s*model",
        r"(?i)^#{0,3}\s*\d*\.?\s*method",
        r"(?i)^#{0,3}\s*\d*\.?\s*approach",
        r"(?i)^#{0,3}\s*\d*\.?\s*formulation",
        r"(?i)^#{0,3}\s*\d*\.?\s*algorithm",
        r"(?i)^#{0,3}\s*\d*\.?\s*framework",
    ],
    "validation": [
        r"(?i)^#{0,3}\s*\d*\.?\s*valid",
        r"(?i)^#{0,3}\s*\d*\.?\s*verification",
        r"(?i)^#{0,3}\s*\d*\.?\s*testing",
        r"(?i)^#{0,3}\s*\d*\.?\s*evaluation",
    ],
    "sensitivity": [
        r"(?i)^#{0,3}\s*\d*\.?\s*sensitiv",
        r"(?i)^#{0,3}\s*\d*\.?\s*robustness",
        r"(?i)^#{0,3}\s*\d*\.?\s*error\s*analysis",
        r"(?i)^#{0,3}\s*\d*\.?\s*stability",
    ],
    "visualization": [
        r"(?i)^#{0,3}\s*\d*\.?\s*result",
        r"(?i)^#{0,3}\s*\d*\.?\s*simulation",
        r"(?i)^#{0,3}\s*\d*\.?\s*numerical",
        r"(?i)^#{0,3}\s*\d*\.?\s*experiment",
    ],
    "conclusion": [
        r"(?i)^#{0,3}\s*\d*\.?\s*conclusion",
        r"(?i)^#{0,3}\s*\d*\.?\s*discussion",
        r"(?i)^#{0,3}\s*\d*\.?\s*future\s*work",
        r"(?i)^#{0,3}\s*\d*\.?\s*strengths?\s*(and|&)\s*weakness",
    ],
    "references": [
        r"(?i)^#{0,3}\s*\d*\.?\s*reference",
        r"(?i)^#{0,3}\s*\d*\.?\s*bibliograph",
    ],
}


def clean_text(text: str) -> str:
    """清洗文本：去乱码、合并断行、规范空白"""
    # 去除常见乱码字符
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # 去除页眉页脚常见模式 (页码、Team #xxxxx 等)
    text = re.sub(r'(?m)^Team\s*#?\s*\d+\s*$', '', text)
    text = re.sub(r'(?m)^Page\s*\d+\s*(of\s*\d+)?\s*$', '', text)
    text = re.sub(r'(?m)^\d+\s*$', '', text)  # 单独的页码行
    # 合并被分页断开的段落（行尾非句号的换行）
    text = re.sub(r'(?<=[a-z,;])\n(?=[a-z])', ' ', text)
    # 多余空行合并
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def is_toc_page(text: str) -> bool:
    """检测页面是否为目录页（Table of Contents）"""
    lines = text.strip().split('\n')
    if not lines:
        return False
    # 目录页特征：大量 ". . ." 点线 + 页码
    dot_leader_count = 0
    for line in lines:
        # 匹配 "Section Title . . . . . . . . 12" 这类目录行
        if re.search(r'\.(\s*\.){3,}', line) or re.search(r'\.\s{0,2}\.{2,}', line):
            dot_leader_count += 1
    # 如果超过30%的行都是目录行，判定为目录页
    if len(lines) > 3 and dot_leader_count / len(lines) > 0.3:
        return True
    return False


def extract_figure_info(page) -> list[dict]:
    """从页面中提取图片信息和 figure caption"""
    figures = []
    # 获取页面中的嵌入图片
    images = page.get_images(full=True)
    text = page.get_text("text")

    # 提取 figure caption（如 "Figure 1: xxx" 或 "Fig. 2. xxx"）
    captions = re.findall(
        r'(?i)((?:Figure|Fig\.?)\s*\d+[\s:.]*[^\n]{0,150})',
        text
    )

    # 提取 table caption
    table_captions = re.findall(
        r'(?i)((?:Table)\s*\d+[\s:.]*[^\n]{0,150})',
        text
    )

    if images or captions or table_captions:
        figures.append({
            "image_count": len(images),
            "figure_captions": captions,
            "table_captions": table_captions,
        })

    return figures


def inject_figure_placeholders(text: str, figure_info: list[dict], page_num: int,
                               vision_descriptions: list[str] = None) -> str:
    """在页面文本中注入图表占位信息 + 视觉分析描述"""
    if not figure_info:
        return text

    info = figure_info[0]
    placeholders = []

    # 注入 figure caption 占位
    for j, caption in enumerate(info["figure_captions"]):
        placeholder = f"[IMAGE ON PAGE {page_num}: {caption.strip()}]"
        # 如果有对应的视觉分析描述，附加到占位符后
        if vision_descriptions and j < len(vision_descriptions):
            placeholder += f"\n[VISION ANALYSIS: {vision_descriptions[j]}]"
        placeholders.append(placeholder)

    # 注入 table caption 占位
    for caption in info["table_captions"]:
        placeholders.append(f"[TABLE ON PAGE {page_num}: {caption.strip()}]")

    # 如果有图片但没有匹配到 caption，添加通用占位
    if info["image_count"] > 0 and not info["figure_captions"] and not info["table_captions"]:
        placeholder = f"[{info['image_count']} IMAGE(S) ON PAGE {page_num}]"
        # 对无 caption 的图片也附加视觉描述
        if vision_descriptions:
            for desc in vision_descriptions:
                placeholder += f"\n[VISION ANALYSIS: {desc}]"
        placeholders.append(placeholder)

    if placeholders:
        placeholder_block = "\n".join(placeholders)
        text = text + "\n" + placeholder_block + "\n"

    return text


def extract_text_fitz(pdf_path: str, llm_client=None,
                      vision_model: str = "openai/gpt-4o") -> list[dict]:
    """
    使用 PyMuPDF 提取每页文本，包含图表检测和可选的视觉分析

    Args:
        pdf_path: PDF 文件路径
        llm_client: OpenAI 兼容客户端（提供后启用视觉分析）
        vision_model: 视觉模型名称
    """
    doc = fitz.open(pdf_path)
    pages = []
    total_figures = 0
    all_figure_captions = []
    all_table_captions = []
    all_vision_descriptions = []

    for i, page in enumerate(doc):
        text = page.get_text("text")
        page_num = i + 1

        # 检测目录页
        toc_flag = is_toc_page(text)

        # 提取图表信息
        fig_info = extract_figure_info(page)

        # 视觉分析：提取实际图片并调用 GPT-4o
        vision_descriptions = []
        if not toc_flag and llm_client and fig_info:
            img_count = fig_info[0]["image_count"]
            if img_count > 0:
                extracted_images = extract_page_images(page, doc)
                if extracted_images:
                    all_captions = (fig_info[0]["figure_captions"] +
                                    fig_info[0]["table_captions"])
                    vision_descriptions = describe_images_with_vision(
                        extracted_images, all_captions,
                        llm_client, vision_model, page_num,
                    )
                    all_vision_descriptions.extend(vision_descriptions)

        # 注入图表占位信息 + 视觉描述到文本流
        if not toc_flag:
            text = inject_figure_placeholders(
                text, fig_info, page_num, vision_descriptions
            )

        # 统计图表
        if fig_info:
            total_figures += fig_info[0]["image_count"]
            all_figure_captions.extend(fig_info[0]["figure_captions"])
            all_table_captions.extend(fig_info[0]["table_captions"])

        pages.append({
            "page_num": page_num,
            "text": text if not toc_flag else "",
            "char_count": len(text.strip()) if not toc_flag else 0,
            "is_toc": toc_flag,
            "image_count": fig_info[0]["image_count"] if fig_info else 0,
        })

    doc.close()

    # 附加汇总信息
    if pages:
        pages[0]["_figure_summary"] = {
            "total_images": total_figures,
            "figure_captions": all_figure_captions,
            "table_captions": all_table_captions,
            "vision_descriptions": all_vision_descriptions,
        }

    return pages


def extract_text_ocr(pdf_path: str) -> list[dict]:
    """OCR fallback：对扫描件使用 pytesseract"""
    if not HAS_OCR:
        raise RuntimeError("pytesseract 未安装，无法对扫描件进行 OCR。请运行: pip install pytesseract Pillow")

    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        # 渲染为图片
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img, lang='eng+chi_sim')
        pages.append({
            "page_num": i + 1,
            "text": text,
            "char_count": len(text.strip()),
        })
    doc.close()
    return pages


def detect_section_type(text_line: str) -> Optional[str]:
    """检测一行文本是否是章节标题，返回章节类型"""
    line = text_line.strip()
    if len(line) > 100:  # 标题一般不会太长
        return None
    for section_type, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, line):
                return section_type
    return None


def is_toc_section(text: str) -> bool:
    """检测一个章节段落是否实际上是目录内容（而非正文）"""
    lines = [l for l in text.strip().split('\n') if l.strip()]
    if not lines:
        return False
    # 统计含 ". . ." 点线的行
    dot_leader_lines = sum(
        1 for l in lines if re.search(r'\.(\s*\.){3,}', l)
    )
    # 如果内容很短（<500字符）且大量点线行，就是目录
    if len(text.strip()) < 600 and dot_leader_lines >= 2:
        return True
    # 如果超过一半的行都是点线，也是目录
    if len(lines) > 2 and dot_leader_lines / len(lines) > 0.4:
        return True
    return False


def segment_into_sections(full_text: str, pages: list[dict]) -> list[dict]:
    """将全文按章节标题分段"""
    lines = full_text.split('\n')
    sections = []
    current_section = {
        "type": "preamble",
        "text": "",
        "start_line": 0,
    }

    for i, line in enumerate(lines):
        section_type = detect_section_type(line)
        if section_type:
            # 保存当前段
            if current_section["text"].strip():
                sections.append(current_section)
            current_section = {
                "type": section_type,
                "text": line + "\n",
                "start_line": i,
            }
        else:
            current_section["text"] += line + "\n"

    # 保存最后一段
    if current_section["text"].strip():
        sections.append(current_section)

    # 过滤掉目录段落
    sections = [s for s in sections if not is_toc_section(s["text"])]

    # 过滤掉过短的章节残片（<80字符且不含实质内容，仅为标题残留）
    # 将其合并到下一个同类型章节，或直接丢弃
    filtered = []
    for s in sections:
        text_stripped = s["text"].strip()
        # 仅保留有实质内容的段落（标题行 + 至少一些正文）
        if len(text_stripped) < 80 and s["type"] != "preamble":
            # 跳过纯标题残片
            continue
        filtered.append(s)
    sections = filtered

    # 估算页码范围
    total_chars = sum(p["char_count"] for p in pages)
    char_per_page = total_chars / max(len(pages), 1)

    char_offset = 0
    for section in sections:
        section_len = len(section["text"])
        start_page = int(char_offset / char_per_page) + 1
        end_page = int((char_offset + section_len) / char_per_page) + 1
        section["page_range"] = [
            min(start_page, len(pages)),
            min(end_page, len(pages))
        ]
        char_offset += section_len

        # 统计该章节内的图表引用
        fig_refs = re.findall(r'\[IMAGE ON PAGE \d+: ([^\]]+)\]', section["text"])
        table_refs = re.findall(r'\[TABLE ON PAGE \d+: ([^\]]+)\]', section["text"])
        generic_imgs = re.findall(r'\[\d+ IMAGE\(S\) ON PAGE \d+\]', section["text"])
        section["figure_count"] = len(fig_refs) + len(generic_imgs)
        section["table_count"] = len(table_refs)

        del section["start_line"]

    return sections


def infer_paper_metadata(filename: str, full_text: str, pages: list[dict] = None,
                         override_year: int = 0) -> dict:
    """从文件名和文本推断论文元数据"""
    year = override_year
    if not year:
        year_match = re.search(r'(20\d{2})', filename)
        year = int(year_match.group(1)) if year_match else 0
    # 二次校验：文件名中的数字可能是队伍编号而非年份
    if year and (year < 2000 or year > 2030):
        year = 0

    problem_match = re.search(r'(?i)\b([A-F])\d*\.pdf', filename)
    if not problem_match:
        problem_match = re.search(r'(?i)problem\s*([A-F])', full_text[:2000])
    problem = problem_match.group(1).upper() if problem_match else "Unknown"

    word_count = len(full_text.split())

    # 生成 paper_id — 清理垃圾后缀
    safe_name = re.sub(r'[^\w]', '_', Path(filename).stem)
    # 去除 _公众号_竞赛资料网_ 等来源标记
    safe_name = re.sub(r'_*公众号.*', '', safe_name)
    safe_name = re.sub(r'_*竞赛资料网.*', '', safe_name)
    safe_name = re.sub(r'_+$', '', safe_name)  # 去尾部下划线
    # 如果清理后只剩纯数字（队伍编号），直接用
    paper_id = f"{year}_{problem}_{safe_name}" if year else safe_name

    # 图表统计
    figure_summary = {}
    if pages and pages[0].get("_figure_summary"):
        figure_summary = pages[0]["_figure_summary"]

    return {
        "paper_id": paper_id,
        "year": year,
        "problem": problem,
        "word_count": word_count,
        "filename": filename,
        "total_images": figure_summary.get("total_images", 0),
        "figure_captions": figure_summary.get("figure_captions", []),
        "table_captions": figure_summary.get("table_captions", []),
    }


def parse_pdf(pdf_path: str, llm_client=None,
              vision_model: str = "openai/gpt-4o",
              override_year: int = 0) -> dict:
    """
    解析单个 PDF 文件，输出结构化 JSON

    Args:
        pdf_path: PDF 文件路径
        llm_client: OpenAI 兼容客户端（提供后启用图片视觉分析）
        vision_model: 视觉模型名称
        override_year: 外部传入的年份（优先级高于从文件名提取）

    返回:
    {
        "paper_id": "2024_A_Stanford",
        "year": 2024,
        "problem": "A",
        "sections": [...],
        "metadata": {...},
        "full_text": "..."
    }
    """
    filename = os.path.basename(pdf_path)

    # 1. 提取文本（含图表检测 + 可选视觉分析）
    pages = extract_text_fitz(pdf_path, llm_client=llm_client,
                               vision_model=vision_model)
    total_chars = sum(p["char_count"] for p in pages)

    # 如果文本层提取太少，尝试 OCR
    if total_chars < 100 and len(pages) > 0:
        try:
            pages = extract_text_ocr(pdf_path)
        except Exception as e:
            print(f"  OCR 失败: {e}")

    # 2. 合并全文并清洗（跳过目录页）
    full_text = "\n\n".join(p["text"] for p in pages if p.get("text"))
    full_text = clean_text(full_text)

    # 3. 按章节分段
    sections = segment_into_sections(full_text, pages)

    # 4. 推断元数据（含图表统计）
    metadata = infer_paper_metadata(filename, full_text, pages,
                                     override_year=override_year)

    return {
        "paper_id": metadata["paper_id"],
        "year": metadata["year"],
        "problem": metadata["problem"],
        "sections": sections,
        "metadata": metadata,
        "full_text": full_text,
    }


def parse_directory(dir_path: str, output_dir: Optional[str] = None,
                    llm_client=None, vision_model: str = "openai/gpt-4o") -> list[dict]:
    """
    批量解析目录下所有 PDF 文件

    Args:
        dir_path: PDF 文件夹路径
        output_dir: 解析结果输出目录（可选，保存为 JSON）
        llm_client: OpenAI 兼容客户端（提供后启用图片视觉分析）
        vision_model: 视觉模型名称

    Returns:
        解析后的论文列表
    """
    pdf_files = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, f))

    if not pdf_files:
        print(f"未在 {dir_path} 中找到 PDF 文件")
        return []

    print(f"找到 {len(pdf_files)} 个 PDF 文件，开始解析...", flush=True)

    papers = []
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"  [{i}/{len(pdf_files)}] 解析: {os.path.basename(pdf_path)}", flush=True)
        try:
            # 从父目录名提取年份（如 "2024年美赛O奖论文" → 2024）
            parent_dir = os.path.basename(os.path.dirname(pdf_path))
            year_from_dir = 0
            dir_year_match = re.search(r'(20\d{2})', parent_dir)
            if dir_year_match:
                y = int(dir_year_match.group(1))
                if 2000 <= y <= 2030:
                    year_from_dir = y

            paper = parse_pdf(pdf_path, llm_client=llm_client,
                              vision_model=vision_model,
                              override_year=year_from_dir)
            papers.append(paper)

            # 保存单篇解析结果
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                out_path = os.path.join(output_dir, f"{paper['paper_id']}.json")
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(paper, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  ❌ 解析失败: {e}", flush=True)

    print(f"解析完成: {len(papers)}/{len(pdf_files)} 成功", flush=True)
    return papers
