#!/usr/bin/env python3
"""拼装 docs/html/src/ 下的模块化部件，导出离线自包含的单文件 HTML。

用法:
    python3 build.py            # 输出到 docs/html/query-cleaning-pipeline.html
    python3 build.py out.html   # 输出到指定路径

部件组织:
    src/shell/head.html   文档头（doctype / html / head 开标签 / meta / title）
    src/shell/tail.html   文档尾（</body></html>）
    src/styles/*.css      样式（base / header / pipeline / taxonomy / prompt / responsive）
    src/sections/*.html   页面内容区块（header / 流程与步骤 / 分类体系 / 组装说明）
    src/scripts/copy.js   交互脚本（提示词复制按钮）

提示词面板不手写：构建时直接导入运行时 PROMPTS（惰性构建、读 templates/*.md
并做 fail-loud 校验），保证页面展示的就是生产提示词——全部 10 个面板
（segment / value_gate / complexity_gate / classify_complex / classify_normal /
verify_complex / verify_recheck / template_family / dedup_pair / translate），模板
提取规则不再在 build.py 里二次实现（曾与 assemble.py 双实现必然漂移）。

构建产物无任何外部依赖（样式与脚本全部内联），任意环境双击即可离线打开。
"""

import html
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "src"
PKG = ROOT.parent.parent / "src" / "query_pipeline"
DEFAULT_OUT = ROOT / "query-cleaning-pipeline.html"


def read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------- 提示词提取（与运行时 PROMPTS 单源同步） ----------
# 提示词面板不手写、也不在 build.py 里二次实现提取规则：直接导入运行时的
# query_pipeline.prompts（惰性构建，读 templates/*.md），页面展示的就是生产
# 提示词。


def _prompts() -> "dict[str, str]":
    sys.path.insert(0, str(PKG.parent))  # src/
    from query_pipeline.prompts import PROMPTS, resolve_prompt  # noqa: PLC0415

    # 触发惰性构建（模板解析 + fail-loud 校验）：构建文档时模板坏会在此报错，
    # 与生产行为一致，而不是默默展示一份过时的提示词。
    for prompt_id in (
        "segment",
        "value_gate",
        "complexity_gate",
        "classify_complex",
        "classify_normal",
        "verify_complex",
        "verify_recheck",
        "template_family",
        "dedup_pair",
        "translate",
    ):
        resolve_prompt(prompt_id)
    return PROMPTS


def prompt_section(label: str, title: str, text: str) -> str:
    return f"""      <section class="prompt-panel" aria-label="{label}">
        <details>
          <summary>
            <span class="summary-title">
              <strong>{title}</strong>
            </span>
            <span class="prompt-actions">
              <button class="copy-button" type="button">复制提示词</button>
              <span class="copy-status" aria-live="polite"></span>
            </span>
          </summary>
          <pre class="prompt-code">{html.escape(text.strip())}</pre>
        </details>
      </section>"""


def build_prompt_sections() -> str:
    prompts = _prompts()
    panels = [
        ("会话分段提示词", "会话分段提示词", "segment"),
        ("价值门提示词", "价值门提示词", "value_gate"),
        ("复杂度门提示词", "复杂度门提示词", "complexity_gate"),
        ("复杂分类提示词", "复杂分类提示词（9 类，含 few-shot 定义与示例）", "classify_complex"),
        ("普通分类提示词", "普通分类提示词（16 类，含适用/排除/边界/易混）", "classify_normal"),
        ("独立复判提示词", "独立复判提示词（含已确认负例）", "verify_complex"),
        ("复判从严提示词", "复判从严提示词（第 2 轮起，逐轮从严）", "verify_recheck"),
        ("模板族裁决提示词", "模板族裁决提示词", "template_family"),
        ("语义去重提示词", "语义去重提示词", "dedup_pair"),
        ("翻译提示词", "翻译提示词", "translate"),
    ]
    sections = [
        prompt_section(label, title, prompts[prompt_id])
        for label, title, prompt_id in panels
    ]
    return "\n\n".join(sections)


# ---------- 拼装 ----------

def build() -> str:
    head = read(SRC / "shell" / "head.html").rstrip("\n")
    tail = read(SRC / "shell" / "tail.html").rstrip("\n")

    css_files = sorted((SRC / "styles").glob("*.css"))
    styles = "\n\n".join(read(p).rstrip("\n") for p in css_files)

    section_files = sorted((SRC / "sections").glob("*.html"))
    sections = "\n\n".join(read(p).rstrip("\n") for p in section_files)
    sections += "\n\n" + build_prompt_sections()

    scripts = "\n\n".join(read(p).rstrip("\n") for p in sorted((SRC / "scripts").glob("*.js")))

    return "\n".join([
        head,
        "    <style>",
        styles,
        "    </style>",
        "  </head>",
        "  <body>",
        "    <main>",
        sections,
        "    </main>",
        "    <script>",
        scripts,
        "    </script>",
        tail,
        "",
    ])


def main() -> None:
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out.write_text(build(), encoding="utf-8")
    print(f"built {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
