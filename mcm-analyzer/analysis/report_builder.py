"""
报告格式化输出模块
- 将分析结果输出为 Markdown
- 保存到文件
- 终端美化输出
"""

import os
from datetime import datetime


def save_report(report: str, output_path: str = None, paper_name: str = "unnamed"):
    """
    保存报告到文件

    Args:
        report: Markdown 格式的报告文本
        output_path: 输出路径（可选，默认生成带时间戳的文件名）
        paper_name: 论文名称
    """
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in paper_name)
        output_path = f"./report_{safe_name}_{timestamp}.md"

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    return output_path


def print_report_rich(report: str):
    """使用 rich 库美化终端输出"""
    try:
        from rich.console import Console
        from rich.markdown import Markdown

        console = Console()
        md = Markdown(report)
        console.print(md)
    except ImportError:
        # fallback 到普通打印
        print(report)
