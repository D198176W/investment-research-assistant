# 智能投研助手 - 完整版 (LangGraph + Streamlit)
# 融合: LangGraph StateGraph 工作流 + 重试机制 + Pydantic 结构化输出 + Streamlit UI + 历史持久化
import json
import streamlit as st
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from io import BytesIO
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Tongyi
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# 页面配置
st.set_page_config(page_title="智能投研助手", page_icon="📊", layout="wide")

# 自定义 CSS
st.markdown("""
<style>
    .main-header { background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%); padding: 2rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem; }
    .phase-card { background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; border-left: 5px solid #007bff; }
    .success-card { background-color: #d4edda; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; border-left: 5px solid #28a745; }
    .error-card { background-color: #f8d7da; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; border-left: 5px solid #dc3545; }
    .report-content { background-color: #ffffff; padding: 1.5rem; border-radius: 8px; border: 1px solid #dee2e6; max-height: 500px; overflow-y: auto; white-space: pre-wrap; line-height: 1.6; }
    .metric-box { background-color: #e3f2fd; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# 主标题
st.markdown('<div class="main-header"><h1> 智能投研助手</h1><p>基于 LangGraph 的五阶段 AI 投资研究系统</p></div>', unsafe_allow_html=True)

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
MAX_RETRIES = 3

# ==================== Pydantic 数据模型 ====================

class PerceptionOutput(BaseModel):
    """感知阶段输出"""
    market_overview: str = Field(..., description="市场概况")
    key_indicators: Dict[str, str] = Field(..., description="关键指标")
    recent_news: List[str] = Field(..., description="近期新闻")
    industry_trends: Dict[str, str] = Field(..., description="行业趋势")

class ModelingOutput(BaseModel):
    """建模阶段输出"""
    market_state: str = Field(..., description="市场状态")
    economic_cycle: str = Field(..., description="经济周期")
    risk_factors: List[str] = Field(..., description="风险因素")
    opportunity_areas: List[str] = Field(..., description="机会领域")
    market_sentiment: str = Field(..., description="市场情绪")

class ReasoningPlan(BaseModel):
    """推理阶段方案"""
    plan_id: str = Field(..., description="方案ID")
    hypothesis: str = Field(..., description="投资假设")
    analysis_approach: str = Field(..., description="分析方法")
    expected_outcome: str = Field(..., description="预期结果")
    confidence_level: float = Field(..., description="置信度(0~1)")
    pros: List[str] = Field(..., description="优势")
    cons: List[str] = Field(..., description="劣势")

class DecisionOutput(BaseModel):
    """决策阶段输出"""
    selected_plan_id: str = Field(..., description="选中方案")
    investment_thesis: str = Field(..., description="投资论点")
    supporting_evidence: List[str] = Field(..., description="支持证据")
    risk_assessment: str = Field(..., description="风险评估")
    recommendation: str = Field(..., description="投资建议")
    timeframe: str = Field(..., description="时间框架")

# ==================== Prompt 模板 ====================

PERCEPTION_PROMPT = """你是一个专业的投资研究分析师，请收集市场数据:
研究主题: {research_topic}
行业焦点: {industry_focus}
时间范围: {time_horizon}

请从以下方面分析：
1.市场概况和最新动态
2.关键经济和市场指标
3.近期重要新闻（至少3条）
4.行业趋势分析（至少三个细分领域）

请严格按以下JSON格式输出:
{{
    "market_overview": "市场概况",
    "key_indicators": {{"指标名": "指标值"}},
    "recent_news": ["新闻1", "新闻2", "新闻3"],
    "industry_trends": {{"细分领域": "趋势分析"}}
}}"""

MODELING_PROMPT = """你是资深投资策略师，请根据市场数据构建内部模型:
研究主题: {research_topic}
行业焦点: {industry_focus}
时间范围: {time_horizon}
市场数据: {perception_data}

请构建全面的市场模型，包括：
1.当前市场状态评估
2.经济周期判断
3.主要风险因素（至少三个）
4.潜在机会领域（至少三个）
5.市场情绪分析

输出JSON格式:
{{
    "market_state": "市场状态",
    "economic_cycle": "经济周期",
    "risk_factors": ["风险1", "风险2", "风险3"],
    "opportunity_areas": ["机会1", "机会2", "机会3"],
    "market_sentiment": "市场情绪"
}}"""

REASONING_PROMPT = """你是战略投资顾问，请生成3个不同的投资方案:
研究主题: {research_topic}
行业焦点: {industry_focus}
时间范围: {time_horizon}
市场模型: {world_model}

请为每个方案提供：方案ID、投资假设、分析方法、预期结果、置信度(0~1)、优势(至少3点)、劣势(至少3点)
方案应有明显差异，代表不同投资思路。

输出JSON数组:
[
    {{"plan_id": "方案1", "hypothesis": "假设", "analysis_approach": "方法", "expected_outcome": "结果", "confidence_level": 0.85, "pros": ["优势1"], "cons": ["劣势1"]}},
    ...
]"""

DECISION_PROMPT = """你是投资决策委员会主席，请评估候选方案并选择最优:
研究主题: {research_topic}
行业焦点: {industry_focus}
时间范围: {time_horizon}
市场模型: {world_model}
候选方案: {reasoning_plans}

请基于假设、预期结果、置信度和优缺点选择最优方案，给出详细理由。

输出JSON格式:
{{
    "selected_plan_id": "选中方案",
    "investment_thesis": "投资论点",
    "supporting_evidence": ["证据1", "证据2"],
    "risk_assessment": "风险评估",
    "recommendation": "投资建议",
    "timeframe": "时间框架"
}}"""

REPORT_PROMPT = """你是专业投研报告撰写人，请生成完整投资报告:
研究主题: {research_topic}
行业焦点: {industry_focus}
时间范围: {time_horizon}
市场数据: {perception_data}
市场模型: {world_model}
选定策略: {selected_plan}

请生成结构完整、逻辑清晰的专业报告，包括：
1.标题和摘要 2.市场背景 3.核心观点 4.分析论证 5.风险因素 6.投资建议 7.时间框架

报告应专业、客观、有深度。"""

# ==================== LangGraph 节点函数 ====================

def perception_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """感知阶段 - 带重试"""
    logging.info("阶段1: 感知 - 收集市场数据")
    retry_count = state.get("perception_retry", 0)
    if retry_count >= MAX_RETRIES:
        return {**state, "error": f"感知阶段重试超过{MAX_RETRIES}次", "current_phase": "error"}
    try:
        llm = Tongyi(model_name=state["model_name"], dashscope_api_key=state["api_key"], base_url=state["api_url"], model_kwargs={})
        prompt = ChatPromptTemplate.from_template(PERCEPTION_PROMPT)
        chain = prompt | llm | JsonOutputParser()
        result = chain.invoke({"research_topic": state["research_topic"], "industry_focus": state["industry_focus"], "time_horizon": state["time_horizon"]})
        return {**state, "perception_data": result, "current_phase": "modeling", "perception_retry": 0}
    except Exception as e:
        logging.error(f"感知阶段错误: {e}")
        return {**state, "error": f"感知阶段出错: {e}", "current_phase": "perception", "perception_retry": retry_count + 1}

def modeling_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """建模阶段 - 带重试"""
    logging.info("阶段2: 建模 - 构建内部模型")
    retry_count = state.get("modeling_retry", 0)
    if retry_count >= MAX_RETRIES:
        return {**state, "error": f"建模阶段重试超过{MAX_RETRIES}次", "current_phase": "error"}
    try:
        if not state.get("perception_data"):
            return {**state, "error": "缺少感知数据", "current_phase": "perception", "modeling_retry": retry_count + 1}
        llm = Tongyi(model_name=state["model_name"], dashscope_api_key=state["api_key"], base_url=state["api_url"], model_kwargs={})
        prompt = ChatPromptTemplate.from_template(MODELING_PROMPT)
        chain = prompt | llm | JsonOutputParser()
        result = chain.invoke({"research_topic": state["research_topic"], "industry_focus": state["industry_focus"], "time_horizon": state["time_horizon"], "perception_data": json.dumps(state["perception_data"], ensure_ascii=False)})
        return {**state, "world_model": result, "current_phase": "reasoning", "modeling_retry": 0}
    except Exception as e:
        logging.error(f"建模阶段错误: {e}")
        return {**state, "error": f"建模阶段出错: {e}", "current_phase": "modeling", "modeling_retry": retry_count + 1}

def reasoning_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """推理阶段 - 带重试"""
    logging.info("阶段3: 推理 - 生成候选方案")
    retry_count = state.get("reasoning_retry", 0)
    if retry_count >= MAX_RETRIES:
        return {**state, "error": f"推理阶段重试超过{MAX_RETRIES}次", "current_phase": "error"}
    try:
        if not state.get("world_model"):
            return {**state, "error": "缺少世界模型", "current_phase": "modeling", "reasoning_retry": retry_count + 1}
        llm = Tongyi(model_name=state["model_name"], dashscope_api_key=state["api_key"], base_url=state["api_url"], model_kwargs={})
        prompt = ChatPromptTemplate.from_template(REASONING_PROMPT)
        chain = prompt | llm | JsonOutputParser()
        result = chain.invoke({"research_topic": state["research_topic"], "industry_focus": state["industry_focus"], "time_horizon": state["time_horizon"], "world_model": json.dumps(state["world_model"], ensure_ascii=False)})
        return {**state, "reasoning_plans": result, "current_phase": "decision", "reasoning_retry": 0}
    except Exception as e:
        logging.error(f"推理阶段错误: {e}")
        return {**state, "error": f"推理阶段出错: {e}", "current_phase": "reasoning", "reasoning_retry": retry_count + 1}

def decision_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """决策阶段 - 带重试"""
    logging.info("阶段4: 决策 - 选择最优方案")
    retry_count = state.get("decision_retry", 0)
    if retry_count >= MAX_RETRIES:
        return {**state, "error": f"决策阶段重试超过{MAX_RETRIES}次", "current_phase": "error"}
    try:
        if not state.get("reasoning_plans"):
            return {**state, "error": "缺少候选方案", "current_phase": "reasoning", "decision_retry": retry_count + 1}
        llm = Tongyi(model_name=state["model_name"], dashscope_api_key=state["api_key"], base_url=state["api_url"], model_kwargs={})
        prompt = ChatPromptTemplate.from_template(DECISION_PROMPT)
        chain = prompt | llm | JsonOutputParser()
        result = chain.invoke({"research_topic": state["research_topic"], "industry_focus": state["industry_focus"], "time_horizon": state["time_horizon"], "world_model": json.dumps(state["world_model"], ensure_ascii=False), "reasoning_plans": json.dumps(state["reasoning_plans"], ensure_ascii=False)})
        return {**state, "selected_plan": result, "current_phase": "report", "decision_retry": 0}
    except Exception as e:
        logging.error(f"决策阶段错误: {e}")
        return {**state, "error": f"决策阶段出错: {e}", "current_phase": "decision", "decision_retry": retry_count + 1}

def report_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """报告阶段 - 带重试"""
    logging.info("阶段5: 报告 - 生成完整报告")
    retry_count = state.get("report_retry", 0)
    if retry_count >= MAX_RETRIES:
        return {**state, "error": f"报告阶段重试超过{MAX_RETRIES}次", "current_phase": "error"}
    try:
        if not state.get("selected_plan"):
            return {**state, "error": "缺少选定方案", "current_phase": "decision", "report_retry": retry_count + 1}
        llm = Tongyi(model_name=state["model_name"], dashscope_api_key=state["api_key"], base_url=state["api_url"], model_kwargs={})
        prompt = ChatPromptTemplate.from_template(REPORT_PROMPT)
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"research_topic": state["research_topic"], "industry_focus": state["industry_focus"], "time_horizon": state["time_horizon"], "perception_data": json.dumps(state["perception_data"], ensure_ascii=False), "world_model": json.dumps(state["world_model"], ensure_ascii=False), "selected_plan": json.dumps(state["selected_plan"], ensure_ascii=False)})
        return {**state, "final_report": result, "current_phase": "completed", "report_retry": 0}
    except Exception as e:
        logging.error(f"报告阶段错误: {e}")
        return {**state, "error": f"报告阶段出错: {e}", "current_phase": "report", "report_retry": retry_count + 1}

# ==================== 条件路由 ====================

def router(state: Dict[str, Any]) -> str:
    """根据当前阶段决定下一步"""
    current = state.get("current_phase")
    
    # 错误处理
    if state.get("error"):
        return "error"
    
    # 完成状态
    if current == "completed":
        return "completed"
    
    # 正常流转: 返回current_phase的值(即节点函数设置的下一阶段)
    return current

def get_fallback_map():
    """获取各阶段的条件边映射"""
    return {
        "perception": {"modeling": "modeling", "perception": "perception", "error": "perception", "END": END},
        "modeling": {"reasoning": "reasoning", "modeling": "modeling", "perception": "perception", "error": "modeling", "END": END},
        "reasoning": {"decision": "decision", "reasoning": "reasoning", "modeling": "modeling", "error": "reasoning", "END": END},
        "decision": {"report": "report", "decision": "decision", "reasoning": "reasoning", "error": "decision", "END": END},
        "report": {"completed": END, "report": "report", "decision": "decision", "error": "report", "END": END}
    }

def create_research_workflow():
    """创建 LangGraph 工作流"""
    workflow = StateGraph(dict)
    workflow.add_node("perception", perception_node)
    workflow.add_node("modeling", modeling_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("decision", decision_node)
    workflow.add_node("report", report_node)
    workflow.set_entry_point("perception")
    fallback = get_fallback_map()
    for phase, mapping in fallback.items():
        workflow.add_conditional_edges(phase, router, mapping)
    return workflow.compile()

def run_research_agent(topic: str, industry: str, horizon: str, api_key: str, api_url: str, model_name: str) -> Dict[str, Any]:
    """运行完整分析流程"""
    agent = create_research_workflow()
    initial_state = {
        "research_topic": topic, "industry_focus": industry, "time_horizon": horizon,
        "api_key": api_key, "api_url": api_url, "model_name": model_name,
        "current_phase": "perception", "perception_data": None, "world_model": None,
        "reasoning_plans": None, "selected_plan": None, "final_report": None, "error": None,
        "perception_retry": 0, "modeling_retry": 0, "reasoning_retry": 0, "decision_retry": 0, "report_retry": 0
    }
    return agent.invoke(initial_state)

# ==================== Session State 初始化 ====================

if 'api_key' not in st.session_state: st.session_state.api_key = ""
if 'api_url' not in st.session_state: st.session_state.api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
if 'model_name' not in st.session_state: st.session_state.model_name = "qwen-plus"
if 'current_analysis' not in st.session_state: st.session_state.current_analysis = None
if 'analysis_complete' not in st.session_state: st.session_state.analysis_complete = False
if 'conversation_history' not in st.session_state: st.session_state.conversation_history = []
if 'show_history' not in st.session_state: st.session_state.show_history = False
if 'pdf_data' not in st.session_state: st.session_state.pdf_data = None
if 'pdf_ready' not in st.session_state: st.session_state.pdf_ready = False

HISTORY_FILE = "analysis_history.json"

def load_history():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f: return json.load(f)
    except: pass
    return []

def save_history(history):
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f: json.dump(history, f, ensure_ascii=False, indent=2)
    except: pass

if not st.session_state.conversation_history:
    st.session_state.conversation_history = load_history()

# ==================== PDF 导出功能 ====================

def generate_pdf_report(report_text: str, topic: str, industry: str, horizon: str) -> bytes:
    """生成PDF格式的投资报告"""
    buffer = BytesIO()
    
    # 创建PDF文档
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20*mm,
        leftMargin=20*mm,
        topMargin=20*mm,
        bottomMargin=20*mm
    )
    
    # 注册中文字体（支持 Windows 和 Linux）
    font_name = 'Helvetica'
    font_registered = False
    
    # Linux 环境（如 Streamlit Cloud）使用内置 CID 字体
    try:
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
        font_name = 'STSong-Light'
        font_registered = True
    except:
        pass
    
    # Windows 字体回退
    if not font_registered:
        try:
            font_path = os.path.join(os.environ.get("SYSTEMROOT", "C:\\Windows"), "Fonts", "simhei.ttf")
            pdfmetrics.registerFont(TTFont('SimHei', font_path))
            font_name = 'SimHei'
            font_registered = True
        except:
            try:
                font_path = os.path.join(os.environ.get("SYSTEMROOT", "C:\\Windows"), "Fonts", "msyh.ttc")
                pdfmetrics.registerFont(TTFont('MSYH', font_path))
                font_name = 'MSYH'
            except:
                pass
    
    # 定义样式
    styles = getSampleStyleSheet()
    
    # 标题样式
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontName=font_name,
        fontSize=18,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=10*mm,
        alignment=1  # 居中
    )
    
    # 副标题样式
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=12,
        textColor=colors.HexColor('#7f8c8d'),
        spaceAfter=8*mm,
        alignment=1  # 居中
    )
    
    # 正文样式
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=11,
        leading=18,
        spaceAfter=5*mm,
        alignment=0  # 左对齐
    )
    
    # 小标题样式
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontName=font_name,
        fontSize=14,
        textColor=colors.HexColor('#3498db'),
        spaceBefore=6*mm,
        spaceAfter=3*mm
    )
    
    # 构建PDF内容
    story = []
    
    # 标题
    story.append(Paragraph("智能投研助手 - 投资分析报告", title_style))
    
    # 副标题 - 研究信息
    meta_text = f"研究主题: {topic} | 行业焦点: {industry} | 时间范围: {horizon}"
    story.append(Paragraph(meta_text, subtitle_style))
    story.append(Spacer(5*mm, 5*mm))
    
    # 分割线
    story.append(Paragraph("_" * 70, body_style))
    story.append(Spacer(5*mm, 5*mm))
    
    # 处理报告内容
    # 按行分割并处理
    lines = report_text.split('\n')
    current_section = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 检测标题行（以数字开头的行，如"1.市场背景"）
        if line and line[0].isdigit() and '.' in line[:5]:
            story.append(Paragraph(line, heading_style))
        elif line.startswith('#'):
            # Markdown标题
            clean_line = line.lstrip('#').strip()
            story.append(Paragraph(clean_line, heading_style))
        else:
            story.append(Paragraph(line, body_style))
    
    # 添加页脚信息
    story.append(Spacer(10*mm, 10*mm))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=9,
        textColor=colors.HexColor('#95a5a6'),
        alignment=1  # 居中
    )
    story.append(Paragraph(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 智能投研助手", footer_style))
    
    # 构建PDF
    doc.build(story)
    
    # 获取PDF字节数据
    pdf_data = buffer.getvalue()
    buffer.close()
    
    return pdf_data

# ==================== 侧边栏 ====================

with st.sidebar:
    st.header("⚙️ 系统配置")
    api_key_input = st.text_input("DashScope API密钥", value=st.session_state.api_key, type="password")
    if api_key_input: st.session_state.api_key = api_key_input

    st.divider()
    st.header("📜 历史记录")
    if st.session_state.conversation_history:
        st.write(f"共 {len(st.session_state.conversation_history)} 条记录")
        if st.button("️ 清空历史", use_container_width=True, type="secondary"):
            st.session_state.conversation_history = []
            try: os.remove(HISTORY_FILE)
            except: pass
            st.rerun()
        st.divider()
        for i, record in enumerate(st.session_state.conversation_history):
            st.markdown(f"**{record.get('time', 'N/A')}**")
            st.markdown(f" {record.get('topic', 'N/A')}")
            st.caption(f"{record.get('industry', '')} | {record.get('horizon', '')}")
            if i < len(st.session_state.conversation_history) - 1: st.divider()
    else:
        st.info("暂无历史记录")

    st.divider()
    st.header(" 模型配置")
    api_url_input = st.text_input("API地址", value=st.session_state.api_url)
    if api_url_input: st.session_state.api_url = api_url_input
    model_name_input = st.text_input("模型名称", value=st.session_state.model_name, help="qwen-plus, qwen-max, qwen-turbo")
    if model_name_input: st.session_state.model_name = model_name_input

    st.divider()
    st.header("📋 分析配置")
    research_topic = st.text_input("研究主题", value="人工智能芯片市场")
    industry_focus = st.text_input("行业焦点", value="半导体")
    time_horizon = st.selectbox("时间范围", ["短期(1-3个月)", "中期(3-6个月)", "中期(6-12个月)", "长期(1年以上)"], index=2)
    start_analysis = st.button(" 开始智能分析", use_container_width=True, type="primary")

# ==================== 主界面 ====================

st.subheader(" 智能投资分析")

if st.session_state.current_analysis:
    current = st.session_state.current_analysis
    phase = current.get("current_phase", "感知阶段")
    phases = ["感知阶段", "建模阶段", "推理阶段", "决策阶段", "报告阶段", "完成"]
    progress_html = "<div style='display: flex; justify-content: space-between; gap: 5px;'>"
    for p in phases:
        if phase == p: color, text = "#007bff", f"{p} "
        elif phase in phases and phases.index(phase) > phases.index(p): color, text = "#28a745", f"{p} ✓"
        else: color, text = "#6c757d", p
        progress_html += f"<span style='background-color: {color}; color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem; font-weight: bold; flex: 1; text-align: center;'>{text}</span>"
    progress_html += "</div>"
    st.markdown(progress_html, unsafe_allow_html=True)

if st.session_state.current_analysis and st.session_state.current_analysis.get("error"):
    st.error(f" {st.session_state.current_analysis['error']}")

# 历史展开区
with st.expander(" 历史分析记录", expanded=st.session_state.show_history):
    if st.session_state.conversation_history:
        for i, record in enumerate(st.session_state.conversation_history):
            col_t, col_d = st.columns([4, 1])
            with col_t: st.markdown(f"**{record.get('time', 'N/A')}**")
            with col_d:
                if st.button("🗑️", key=f"del_{i}"):
                    st.session_state.conversation_history.pop(i)
                    save_history(st.session_state.conversation_history)
                    st.rerun()
            st.markdown(f" **主题**: {record.get('topic')}")
            st.caption(f"行业: {record.get('industry')} | 范围: {record.get('horizon')} | 模型: {record.get('model')}")
            if i < len(st.session_state.conversation_history) - 1: st.divider()
    else:
        st.info(" 暂无历史分析记录")

# 阶段结果展示
if st.session_state.current_analysis:
    s = st.session_state.current_analysis
    if s.get("perception_data"):
        with st.expander("🔍 感知阶段 - 市场数据", expanded=True):
            d = s["perception_data"]
            st.markdown(f"**市场概况:** {d.get('market_overview', 'N/A')}")
            st.markdown("**关键指标:**")
            cols = st.columns(2)
            for i, (k, v) in enumerate(d.get('key_indicators', {}).items()):
                with cols[i % 2]: st.markdown(f'<div class="metric-box">**{k}**: {v}</div>', unsafe_allow_html=True)
            st.markdown("**近期新闻:**")
            for n in d.get('recent_news', [])[:3]: st.markdown(f" {n}")
            st.markdown("**行业趋势:**")
            for k, v in d.get('industry_trends', {}).items(): st.markdown(f"**{k}**: {v}")

    if s.get("world_model"):
        with st.expander(" 建模阶段 - 内部模型", expanded=True):
            m = s["world_model"]
            st.markdown(f"**市场状态:** {m.get('market_state')}")
            st.markdown(f"**经济周期:** {m.get('economic_cycle')}")
            st.markdown(f"**市场情绪:** {m.get('market_sentiment')}")
            st.markdown("**风险因素:**")
            for r in m.get('risk_factors', [])[:3]: st.markdown(f" {r}")
            st.markdown("**机会领域:**")
            for o in m.get('opportunity_areas', [])[:3]: st.markdown(f" {o}")

    if s.get("reasoning_plans"):
        with st.expander("🤔 推理阶段 - 候选方案", expanded=True):
            plans = s["reasoning_plans"]
            if isinstance(plans, list):
                for i, plan in enumerate(plans):
                    st.markdown(f"**方案 {i+1}: {plan.get('plan_id')}**")
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"*假设:* {plan.get('hypothesis')}")
                        st.markdown(f"*方法:* {plan.get('analysis_approach')}")
                    with c2: st.metric("置信度", f"{plan.get('confidence_level', 0):.0%}")
                    st.markdown(f"*预期:* {plan.get('expected_outcome')}")
                    st.markdown(f"*优:* {', '.join(plan.get('pros', [])[:2])}")
                    st.markdown(f"*劣:* {', '.join(plan.get('cons', [])[:2])}")
                    st.divider()

    if s.get("selected_plan"):
        with st.expander("✅ 决策阶段 - 最优方案", expanded=True):
            dec = s["selected_plan"]
            st.markdown(f"**选中:** {dec.get('selected_plan_id')}")
            st.markdown(f"**论点:** {dec.get('investment_thesis')}")
            st.markdown(f"**风险:** {dec.get('risk_assessment')}")
            st.info(f"**建议:** {dec.get('recommendation')}")
            st.markdown(f"**时间:** {dec.get('timeframe')}")
            st.markdown("**证据:**")
            for e in dec.get('supporting_evidence', [])[:3]: st.markdown(f" {e}")

    if s.get("final_report"):
        st.markdown("### 📄 最终投资报告")
        st.markdown(f"**主题:** {s['research_topic']} ({s['industry_focus']}) | **范围:** {s['time_horizon']}")
        with st.container():
            st.markdown(f'<div class="report-content">{s["final_report"]}</div>', unsafe_allow_html=True)
        
        # PDF 导出按钮
        st.markdown("---")
        if st.session_state.pdf_ready and st.session_state.pdf_data:
            st.download_button(
                label="💾 下载PDF报告",
                data=st.session_state.pdf_data,
                file_name=f"投研报告_{s['research_topic']}_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=False,
                type="primary"
            )
            if st.button("🔄 重新生成PDF", use_container_width=False):
                st.session_state.pdf_data = None
                st.session_state.pdf_ready = False
                st.rerun()
        else:
            if st.button("📥 导出PDF报告", use_container_width=False, type="primary"):
                with st.spinner("正在生成PDF报告..."):
                    try:
                        pdf_data = generate_pdf_report(
                            s["final_report"], 
                            s['research_topic'], 
                            s['industry_focus'], 
                            s['time_horizon']
                        )
                        st.session_state.pdf_data = pdf_data
                        st.session_state.pdf_ready = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ PDF生成失败: {str(e)}")

# 处理分析请求
if start_analysis:
    if not st.session_state.api_key:
        st.error(" 请先在侧边栏输入API密钥")
    else:
        st.session_state.analysis_complete = False
        st.session_state.current_analysis = {"research_topic": research_topic, "industry_focus": industry_focus, "time_horizon": time_horizon, "current_phase": "初始化...", "perception_data": None, "world_model": None, "reasoning_plans": None, "selected_plan": None, "final_report": None, "error": None}
        with st.spinner(" 智能体正在进行多阶段分析（LangGraph工作流），请稍候..."):
            result = run_research_agent(research_topic, industry_focus, time_horizon, st.session_state.api_key, st.session_state.api_url, st.session_state.model_name)
            st.session_state.current_analysis = result
            if result.get("current_phase") == "completed":
                st.session_state.analysis_complete = True
                record = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "topic": research_topic, "industry": industry_focus, "horizon": time_horizon, "model": st.session_state.model_name, "report_preview": result.get("final_report", "")[:200] + "..." if result.get("final_report") else "无报告"}
                st.session_state.conversation_history.insert(0, record)
                if len(st.session_state.conversation_history) > 20: st.session_state.conversation_history = st.session_state.conversation_history[:20]
                save_history(st.session_state.conversation_history)
        st.success("✅ 分析完成！")
        st.rerun()

if st.button("️ 清空分析结果", use_container_width=True):
    st.session_state.current_analysis = None
    st.session_state.analysis_complete = False
    st.rerun()

# 系统信息
with st.expander("ℹ️ 系统架构信息"):
    st.write("**核心架构:** LangGraph StateGraph + 条件路由 + 自动重试机制")
    st.write("**五阶段流程:** 感知→建模→推理→决策→报告")
    st.write("**错误处理:** 每阶段独立重试(最多3次)，失败可回退至上一阶段")
    st.write("**输出保障:** Pydantic 结构化输出模型 + JsonOutputParser")
    st.write("**历史管理:** JSON 文件持久化，支持20条记录 + 单条删除")
    st.write("**技术栈:** Python, LangChain, LangGraph, Streamlit, DashScope, Pydantic")
