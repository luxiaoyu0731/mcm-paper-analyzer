#!/usr/bin/env python3
"""
MCM/ICM O奖论文分析系统 - CLI 入口

用法:
  # 1. 导入 O 奖论文建库
  python main.py ingest --dir ./data/o_award_papers/

  # 2. 增量添加论文
  python main.py ingest --dir ./new_papers/ --append

  # 3. 完整诊断
  python main.py analyze --paper ./my_draft.pdf

  # 4. 快速体检
  python main.py analyze --paper ./my_draft.pdf --mode quick

  # 5. 单章节分析
  python main.py analyze --paper ./my_draft.pdf --mode section --section modeling

  # 6. 带 O 奖对比
  python main.py analyze --paper ./my_draft.pdf --compare

  # 7. 查看 O 奖获奖基因报告
  python main.py patterns

  # 8. 查看知识库状态
  python main.py status

  # 9. 交互式对话
  python main.py chat
"""

import os
import sys
import json
import yaml
import click
from pathlib import Path

# 将项目根目录添加到 Python 路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def load_config(config_path: str = None) -> dict:
    """加载配置文件"""
    if not config_path:
        config_path = os.path.join(PROJECT_ROOT, "config.yaml")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 环境变量覆盖 API Key
    env_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
    if env_key:
        config["llm"]["api_key"] = env_key

    return config


def get_llm_client(config: dict):
    """创建 LLM 客户端 (OpenAI 兼容，支持 OpenRouter/DeepSeek 等)"""
    from openai import OpenAI

    api_key = config["llm"]["api_key"]
    if not api_key:
        click.echo("❌ 未设置 API Key！请在 config.yaml 中填入 api_key，或设置环境变量 OPENROUTER_API_KEY")
        sys.exit(1)

    return OpenAI(
        api_key=api_key,
        base_url=config["llm"]["base_url"],
    )


def init_components(config: dict, need_embedder: bool = True):
    """初始化所有组件"""
    from core.embedder import Embedder
    from core.vectorstore import VectorStore
    from core.retriever import Retriever

    # 解析相对路径
    persist_dir = os.path.join(PROJECT_ROOT, config["vectorstore"]["persist_directory"])

    vectorstore = VectorStore(
        persist_dir=persist_dir,
        collection_coarse=config["vectorstore"]["collection_coarse"],
        collection_fine=config["vectorstore"]["collection_fine"],
    )

    embedder = None
    if need_embedder:
        embedder = Embedder(
            model_name=config["embedding"]["model_name"],
            device=config["embedding"]["device"],
            batch_size=config["embedding"]["batch_size"],
        )

    retriever = None
    if embedder:
        retriever = Retriever(
            vectorstore=vectorstore,
            embedder=embedder,
            use_reranker=config["retrieval"].get("use_reranker", False),
            reranker_model=config["retrieval"].get("reranker_model", ""),
        )

    return vectorstore, embedder, retriever


def load_gene_profile(config: dict) -> dict:
    """加载基因图谱"""
    gene_path = os.path.join(PROJECT_ROOT, config["paths"]["gene_profile"])
    if os.path.exists(gene_path):
        with open(gene_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


@click.group()
@click.option('--config', '-c', default=None, help='配置文件路径')
@click.pass_context
def cli(ctx, config):
    """MCM/ICM O奖论文分析系统"""
    ctx.ensure_object(dict)
    ctx.obj['config_path'] = config


@cli.command()
@click.option('--dir', '-d', 'paper_dir', required=True, help='O 奖论文 PDF 文件夹路径')
@click.option('--append', is_flag=True, help='增量添加（不清空现有数据）')
@click.option('--no-vision', is_flag=True, help='跳过 GPT-4o 图片视觉分析（节省费用）')
@click.pass_context
def ingest(ctx, paper_dir, append, no_vision):
    """导入 O 奖论文到知识库"""
    config = load_config(ctx.obj['config_path'])

    click.echo("=" * 60)
    click.echo("📚 MCM/ICM O奖论文导入系统")
    click.echo("=" * 60)

    # 视觉分析客户端（可选）
    vision_client = None
    vision_model = config["llm"]["model"]
    if not no_vision:
        vision_client = get_llm_client(config)
        click.echo(f"🔍 已启用 GPT-4o 图片视觉分析 (model: {vision_model})")
        click.echo("   使用 --no-vision 可跳过图片分析以节省费用")
    else:
        click.echo("⏭️ 已跳过图片视觉分析")

    # 1. 解析 PDF（含可选视觉分析）
    from core.parser import parse_directory
    parsed_dir = os.path.join(PROJECT_ROOT, config["paths"]["parsed_papers"])
    papers = parse_directory(paper_dir, output_dir=parsed_dir,
                             llm_client=vision_client, vision_model=vision_model)

    if not papers:
        click.echo("❌ 未找到可解析的 PDF 文件")
        return

    # 2. 分块
    from core.chunker import chunk_papers
    coarse_chunks, fine_chunks = chunk_papers(
        papers,
        coarse_size=config["chunking"]["coarse_size"],
        coarse_overlap=config["chunking"]["coarse_overlap"],
        fine_size=config["chunking"]["fine_size"],
        fine_overlap=config["chunking"]["fine_overlap"],
    )

    # 3. 向量化与入库
    vectorstore, embedder, retriever = init_components(config)

    click.echo("向量化粗粒度块...")
    coarse_texts = [c["text"] for c in coarse_chunks]
    coarse_embeddings = embedder.encode(coarse_texts)
    vectorstore.add_chunks(coarse_chunks, coarse_embeddings, granularity="coarse")

    click.echo("向量化细粒度块...")
    fine_texts = [c["text"] for c in fine_chunks]
    fine_embeddings = embedder.encode(fine_texts)
    vectorstore.add_chunks(fine_chunks, fine_embeddings, granularity="fine")

    # 4. 提炼基因图谱
    click.echo("\n🧬 提炼 O 奖基因图谱...")
    llm_client = get_llm_client(config)

    from core.gene_extractor import GeneExtractor
    gene_extractor = GeneExtractor(
        retriever=retriever,
        llm_client=llm_client,
        model=config["llm"]["model"],
    )

    gene_path = os.path.join(PROJECT_ROOT, config["paths"]["gene_profile"])
    gene_profile = gene_extractor.extract_full_profile(output_path=gene_path)

    # 完成
    stats = vectorstore.get_stats()
    click.echo("\n" + "=" * 60)
    click.echo("✅ 导入完成！")
    click.echo(f"   论文数量: {len(papers)}")
    click.echo(f"   粗粒度块: {stats['coarse_count']}")
    click.echo(f"   细粒度块: {stats['fine_count']}")
    click.echo(f"   基因图谱: {gene_path}")
    click.echo("=" * 60)


@cli.command()
@click.option('--paper', '-p', required=True, help='用户论文 PDF 路径')
@click.option('--mode', '-m', default='full', type=click.Choice(['full', 'section', 'quick']),
              help='分析模式: full(完整) / section(单章节) / quick(快速体检)')
@click.option('--section', '-s', default=None, help='指定章节 (section 模式下使用)')
@click.option('--compare', is_flag=True, help='输出与最相似 O 奖论文的对比')
@click.option('--output', '-o', default=None, help='报告输出路径')
@click.pass_context
def analyze(ctx, paper, mode, section, compare, output):
    """分析用户论文"""
    config = load_config(ctx.obj['config_path'])

    click.echo("=" * 60)
    click.echo("📊 MCM/ICM 论文诊断系统")
    click.echo("=" * 60)

    # 检查知识库
    gene_profile = load_gene_profile(config)
    if not gene_profile:
        click.echo("⚠️ 未找到 O 奖基因图谱。请先运行 `python main.py ingest` 导入论文。")
        click.echo("  系统将在无对标数据的情况下进行基础分析。")
        gene_profile = {"sections": {}, "statistical_profile": {}}

    # 初始化组件
    vectorstore, embedder, retriever = init_components(config)
    llm_client = get_llm_client(config)

    from analysis.analyzer import PaperAnalyzer
    analyzer = PaperAnalyzer(
        retriever=retriever,
        gene_profile=gene_profile,
        llm_client=llm_client,
        model=config["llm"]["model"],
        max_context_tokens=config["context"]["max_context_tokens"],
    )

    # 执行分析
    report = analyzer.analyze(
        paper_path=paper,
        mode=mode,
        section_filter=section,
        compare=compare,
    )

    # 输出
    from analysis.report_builder import print_report_rich, save_report

    print_report_rich(report)

    # 保存报告
    paper_name = Path(paper).stem
    saved_path = save_report(report, output_path=output, paper_name=paper_name)
    click.echo(f"\n📁 报告已保存到: {saved_path}")


@cli.command()
@click.pass_context
def patterns(ctx):
    """查看 O 奖获奖基因报告"""
    config = load_config(ctx.obj['config_path'])

    gene_profile = load_gene_profile(config)
    if not gene_profile:
        click.echo("❌ 未找到基因图谱。请先运行 `python main.py ingest` 导入论文。")
        return

    vectorstore, embedder, retriever = init_components(config)
    llm_client = get_llm_client(config)

    from analysis.analyzer import PaperAnalyzer
    analyzer = PaperAnalyzer(
        retriever=retriever,
        gene_profile=gene_profile,
        llm_client=llm_client,
        model=config["llm"]["model"],
    )

    report = analyzer.summarize_patterns()

    from analysis.report_builder import print_report_rich, save_report
    print_report_rich(report)
    saved_path = save_report(report, paper_name="o_award_patterns")
    click.echo(f"\n📁 报告已保存到: {saved_path}")


@cli.command()
@click.pass_context
def status(ctx):
    """查看知识库状态"""
    config = load_config(ctx.obj['config_path'])

    vectorstore, _, _ = init_components(config, need_embedder=False)
    stats = vectorstore.get_stats()

    click.echo("=" * 60)
    click.echo("📦 知识库状态")
    click.echo("=" * 60)
    click.echo(f"  粗粒度块: {stats['coarse_count']}")
    click.echo(f"  细粒度块: {stats['fine_count']}")
    click.echo(f"  论文数量: {len(stats['paper_ids'])}")
    if stats['paper_ids']:
        click.echo("  论文列表:")
        for pid in sorted(stats['paper_ids']):
            click.echo(f"    - {pid}")

    gene_path = os.path.join(PROJECT_ROOT, config["paths"]["gene_profile"])
    if os.path.exists(gene_path):
        click.echo(f"\n  基因图谱: ✅ 已生成 ({gene_path})")
    else:
        click.echo(f"\n  基因图谱: ❌ 未生成")

    click.echo(f"\n  LLM 提供商: {config['llm']['provider']}")
    click.echo(f"  LLM 模型: {config['llm']['model']}")
    click.echo(f"  Embedding: {config['embedding']['model_name']}")


@cli.command()
@click.pass_context
def chat(ctx):
    """交互式对话模式"""
    config = load_config(ctx.obj['config_path'])

    click.echo("=" * 60)
    click.echo("💬 MCM/ICM 评审对话模式")
    click.echo("   输入问题与 O 奖评委对话")
    click.echo("   输入 'quit' 或 'exit' 退出")
    click.echo("=" * 60)

    gene_profile = load_gene_profile(config)
    vectorstore, embedder, retriever = init_components(config)
    llm_client = get_llm_client(config)

    conversation_history = []

    # 系统角色设定
    system_msg = """你是一位具备 20 年 MCM/ICM 评审经验的资深评委兼学术优化教练。
你深谙 O 奖（Outstanding Winner）论文的评判标准，能从获奖论文中提炼共性"获奖基因"。
你的回答必须：
1. 有据可查——引用具体 O 奖论文的具体章节
2. 可操作——给出具体的改写示例、图表建议、结构调整方案
3. 用中文回答

当前知识库状态：{stats}
""".format(stats=json.dumps(vectorstore.get_stats(), ensure_ascii=False))

    conversation_history.append({"role": "system", "content": system_msg})

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ('quit', 'exit', 'q'):
            click.echo("再见！")
            break

        if not user_input:
            continue

        # 检索相关上下文
        if retriever and embedder:
            results = retriever.retrieve_for_section(
                query_text=user_input, top_k=3, granularity="fine",
            )
            if results:
                context = retriever.format_retrieval_context(results)
                user_input_with_context = f"""用户问题: {user_input}

## 相关 O 奖论文参考（从知识库检索）
{context}

请结合以上参考资料回答用户问题。"""
            else:
                user_input_with_context = user_input
        else:
            user_input_with_context = user_input

        conversation_history.append({"role": "user", "content": user_input_with_context})

        try:
            response = llm_client.chat.completions.create(
                model=config["llm"]["model"],
                messages=conversation_history,
                temperature=0.7,
                max_tokens=4096,
            )
            reply = response.choices[0].message.content.strip()
            conversation_history.append({"role": "assistant", "content": reply})

            click.echo(f"\n评委: {reply}")
        except Exception as e:
            click.echo(f"\n❌ 调用失败: {e}")


if __name__ == "__main__":
    cli()
