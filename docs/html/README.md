# docs/html — 架构说明页（模块化拆分版）

`query-cleaning-pipeline.html` 是 query_pipeline 清洗标注管线的单页架构说明：
8 阶段流程（preclean → segment → rule_gate → judge → verify → simple_gate → answer_gate → post）、
输出去向、真实运行数据漏斗、双分类体系（复杂 9 类 + 普通 16 类）与全部生产提示词（9 个面板）。
本目录把单文件拆成模块化部件，由构建脚本拼装回离线自包含的单页。

## 目录结构

```
docs/html/
├── build.py                      # 拼装脚本（Python 标准库，无第三方依赖）
├── query-cleaning-pipeline.html  # 构建产物：离线自包含单页（双击即可打开）
├── README.md
└── src/                          # 源部件（修改只动这里，改完跑 build.py）
    ├── shell/
    │   ├── head.html             # 文档头：doctype / html / head / meta / title
    │   └── tail.html             # 文档尾：</body></html>
    ├── styles/                   # 样式，按职责分 6 个文件
    │   ├── 01-base.css           #   设计变量(:root) / 全局重置 / 基础排版
    │   ├── 02-header.css         #   标题区与输出去向图例
    │   ├── 03-pipeline.css       #   流程图 / 8 阶段决策表 / 产物栏 / 数据漏斗
    │   ├── 04-taxonomy.css       #   双分类体系表（复杂 9 + 普通 16）
    │   ├── 05-prompt.css         #   提示词面板 / 复制按钮
    │   └── 06-responsive.css     #   响应式(≤860px) 与打印样式
    ├── sections/                 # 页面内容区块（每个 section 一个文件）
    │   ├── 00-header.html        #   标题 + 输出去向图例
    │   ├── 10-pipeline.html      #   流程 / 8 阶段步骤表 / 产物 / 数据漏斗
    │   ├── 30-taxonomy.html      #   双分类体系（构建时从 templates/ 数据生成）
    │   └── 40-classify-note.html #   分类提示词组装说明
    └── scripts/
        └── copy.js               # 提示词复制按钮交互
```

提示词面板**不手写**：`build.py` 构建时直接导入运行时
`query_pipeline.prompts`（惰性构建，读 `templates/*.md` 并做 fail-loud 校验），
生成全部 9 个面板（segment / value_gate / complexity_gate / complex_judge /
classify_complex / classify_normal / verify_complex / verify_recheck /
translate）——页面展示的提示词与代码永远同步。

## 构建

```bash
python3 build.py                  # 输出 docs/html/query-cleaning-pipeline.html
python3 build.py out.html         # 或输出到指定路径
```

产物样式与脚本全部内联，无任何外部依赖，任意环境离线打开。

## 拆分约定

- **样式归样式**：所有视觉规则在 `src/styles/`，按页面区块分文件；文件拼接顺序即样式声明顺序。
- **内容归内容**：页面结构在 `src/sections/`，不掺样式。
- **交互归交互**：复制按钮一段 JS，在 `src/scripts/copy.js`。
- **提示词归代码**：提示词面板由构建脚本从 `src/query_pipeline/prompts/` 自动生成，改提示词只改代码，页面重建即同步。
- **分类归模板**：分类表由构建脚本从 `src/query_pipeline/templates/`（categories.md / complex_few_shot.md / normal_few_shot.md）解析生成，改分类只改模板。
- **无死代码**：所有 CSS 选择器都有对应元素；旧版遗留样式与 SVG 渲染脚本已在拆分时删除。

## 修改指引

| 想改什么 | 改哪里 |
|---|---|
| 页面配色 / 字体 / 间距 | `styles/01-base.css` |
| 流程图 / 步骤表 / 阈值文案 | `sections/10-pipeline.html` |
| 数据漏斗示例数字 | `sections/10-pipeline.html`（换一次真实 run 的 summary.json 即可） |
| 分类体系 | `src/query_pipeline/templates/categories.md` 等模板，重建页面 |
| 提示词正文 | `src/query_pipeline/prompts/*.py`，重建页面 |
| 复制按钮行为 | `scripts/copy.js` |

> 注意：`30-taxonomy.html` 由模板生成，改分类请改模板后重新构建；
> 手动编辑该文件会在下次构建时被覆盖。
