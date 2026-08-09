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

提示词面板不手写：构建时从 src/query_pipeline/prompts/*.py 提取常量自动生成，
保证页面展示的就是生产提示词；verify 提示词按代码逻辑追加已确认负例。

构建产物无任何外部依赖（样式与脚本全部内联），任意环境双击即可离线打开。
"""

import ast
import html
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "src"
PKG = ROOT.parent.parent / "src" / "query_pipeline"
PROMPTS_DIR = PKG / "prompts"
TPL_DIR = PKG / "templates"
DEFAULT_OUT = ROOT / "query-cleaning-pipeline.html"


def read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------- 提示词提取（与 src/query_pipeline/prompts/ 保持同步） ----------

def extract_prompt_constant(py_file: pathlib.Path, name: str) -> str:
    """提取模块级字符串常量；支持 NAME = dedent(...) 形式。"""
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target = node.targets[0] if isinstance(node, ast.Assign) else node.target
        if not (isinstance(target, ast.Name) and target.id == name):
            continue
        value = node.value
        # 三种常见形式：NAME = "..." / NAME = dedent("...") / NAME = "...".strip()
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
        if isinstance(value, ast.Call):
            if value.args:
                arg = value.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    return arg.value
            if isinstance(value.func, ast.Constant) and isinstance(value.func.value, str):
                return value.func.value
            if (
                isinstance(value.func, ast.Attribute)
                and isinstance(value.func.value, ast.Constant)
                and isinstance(value.func.value.value, str)
            ):
                return value.func.value.value
    raise ValueError(f"{name} not found in {py_file}")


_ANNOTATION_RE = re.compile(r"^(?:以及|看下来|\d+、)")


def parse_bad_cases() -> tuple[str, ...]:
    """与 prompts/assemble.py::parse_bad_cases 同规则：负例行，去掉 `-- reason`。"""
    text = read(TPL_DIR / "bad_cases_for_complex.md")
    cases: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or _ANNOTATION_RE.match(line):
            continue
        case = re.split(r"\s*--\s*", line)[0].strip()
        if case:
            cases.append(case)
    return tuple(cases)


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
    seg = extract_prompt_constant(PROMPTS_DIR / "segment.py", "SEGMENT")
    value = extract_prompt_constant(PROMPTS_DIR / "value_gate.py", "VALUE_GATE")
    complexity = extract_prompt_constant(PROMPTS_DIR / "complexity_gate.py", "COMPLEXITY_GATE")
    verify = extract_prompt_constant(PROMPTS_DIR / "verify.py", "VERIFY_COMPLEX").strip()
    cases = parse_bad_cases()
    if cases:
        verify = (
            verify
            + "\n\n以下问句已被确认为不复杂（负例，判定时参照，不要与之冲突）：\n"
            + "\n".join(f"- {c}" for c in cases)
        )
    translate = extract_prompt_constant(PROMPTS_DIR / "translate.py", "TRANSLATE")
    sections = [
        prompt_section("会话分段提示词", "会话分段提示词 <code>segment</code> · LLM", seg),
        prompt_section("价值门提示词", "价值门提示词 <code>value_gate</code> · LLM", value),
        prompt_section("复杂度门提示词", "复杂度门提示词 <code>complexity_gate</code> · LLM", complexity),
        prompt_section("独立复判提示词", "独立复判提示词 <code>verify_complex</code> · LLM（含已确认负例）", verify),
        prompt_section("翻译提示词", "翻译提示词 <code>translate</code> · LLM", translate),
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
