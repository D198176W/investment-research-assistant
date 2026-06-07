# 智能投研助手系统

> 基于 **LangGraph + LangChain** 的五阶段 AI 投研分析平台

---

## 项目简介

智能投研助手是一款面向投资分析师的 AI 驱动的投研分析平台，通过 **LangGraph StateGraph** 构建五阶段 Agent 工作流（感知→建模→推理→决策→报告），实现从市场数据感知到投资决策生成的全流程自动化。

### 核心亮点

- **LangGraph 状态机架构**：基于 `StateGraph` + 条件路由边，支持阶段回退与自动重试
- **Pydantic 结构化输出**：4 个 Pydantic 数据模型保障 LLM 输出格式一致性
- **自动重试机制**：每阶段独立重试计数器（MAX_RETRIES=3），失败自动回退至上一阶段
- **完整 Web 界面**：Streamlit 构建，支持 API 配置、模型切换、历史持久化
- **历史记录管理**：JSON 文件持久化，支持最多 20 条分析记录 + 单条删除

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit Web UI                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────┐ │
│  │ 感知阶段 │→│ 建模阶段 │→│ 推理阶段 │→│ 决策阶段 │→│ 报告  │ │
│  └────┬─────┘ └─────────┘ └─────────┘ └─────────┘ └───┬───┘ │
│       │ (retry)    │ (retry)    │ (retry)    │ (retry)   │     │
│       └────────────┴────────────────────────┴───────────┘     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
              LangGraph StateGraph (条件路由 + 重试)
                              ↓
              ┌──────────────────────────────┐
              │   DashScope API (通义千问)    │
              │   qwen-plus / qwen-max / ... │
              └──────────────────────────────┘
```

---

## 技术栈

| 类别 | 技术 |
|------|------|
| **核心框架** | Python 3.11+, LangChain, LangGraph |
| **LLM** | DashScope API (通义千问) |
| **前端** | Streamlit |
| **数据验证** | Pydantic BaseModel |
| **输出解析** | JsonOutputParser, StrOutputParser |

---

## 功能特性

### 五阶段分析流程

| 阶段 | 功能 | 输出模型 |
|------|------|----------|
| **感知** | 收集市场概况、关键指标、近期新闻、行业趋势 | `PerceptionOutput` |
| **建模** | 构建市场状态、经济周期、风险因素、机会领域 | `ModelingOutput` |
| **推理** | 生成 3 个差异化投资方案，含置信度评估 | `ReasoningPlan[]` |
| **决策** | 选择最优方案，给出投资论点与风险评估 | `DecisionOutput` |
| **报告** | 生成结构完整的投资研究报告 | 纯文本 |

### 错误处理机制

- **每阶段独立重试**：最多 3 次，失败后自动回退至上一阶段重新执行
- **条件路由**：`router()` 函数根据 `current_phase` 动态决定下一步走向
- **结构化保障**：`JsonOutputParser` + Pydantic 模型双重校验

---

## 快速开始

### 1. 安装依赖

```bash
cd investment-research-assistant
pip install -r requirements.txt
```

### 2. 配置 API 密钥

```bash
# 复制示例配置
cp .env.example .env

# 编辑 .env 填入你的 API 密钥
```

### 3. 启动应用

```bash
streamlit run main.py
```

浏览器自动打开 `http://localhost:8501` 即可使用。

---

## 项目结构

```
investment-research-assistant/
├── main.py              # 完整应用（LangGraph + Streamlit 合并版）
├── .env.example         # 环境变量模板
├── .gitignore           # Git 忽略配置
├── requirements.txt     # Python 依赖
└── README.md            # 项目说明
```

---

## 安全提示

️ **请勿将 `.env` 文件提交到版本控制系统**，该文件包含 API 密钥等敏感信息。项目已配置 `.gitignore` 自动忽略。

---

## License

MIT
