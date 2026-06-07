# 智能投研助手 - 稳定版 (顺序执行 + 异常兜底)
import json
import streamlit as st
import os
import logging
from datetime import datetime
from typing import Dict, Any
from io import BytesIO
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Tongyi
from langchain_core.output_parsers import StrOutputParser
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
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>📊 智能投研助手</h1><p>基于大模型的五阶段 AI 投资研究系统</p></div>', unsafe_allow_html=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
MAX_RETRIES = 3

# ==================== 提示词模板 ====================

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

市场数据:
{perception_data}

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

市场模型:
{world_model}

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

市场模型:
{world_model}

候选方案:
{reasoning_plans}

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

市场数据:
{perception_data}

市场模型:
{world_model}

选定策略:
{selected_plan}

请生成结构完整、逻辑清晰的专业报告，包括：
1.标题和摘要 2.市场背景 3.核心观点 4.分析论证 5.风险因素 6.投资建议 7.时间框架

报告应专业、客观、有深度。"""

# ==================== LLM调用函数 ====================

def call_llm(prompt_text: str, api_key: str, api_url: str, model_name: str) -> str:
    """调用LLM，返回文本结果"""
    llm = Tongyi(
        model_name=model_name,
        dashscope_api_key=api_key,
        base_url=api_url,
        model_kwargs={}
    )
    # 直接调用，不使用模板
    return llm.invoke(prompt_text)

def safe_json_parse(text: str, default_key: str) -> dict:
    """安全解析JSON，失败返回兜底数据"""
    if not text:
        return {default_key: "无返回内容", "raw_output": ""}
    try:
        # 尝试提取JSON块
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        return json.loads(text)
    except Exception:
        return {default_key: text[:500], "raw_output": text}

# ==================== 五阶段执行函数 ====================

def run_perception(topic: str, industry: str, horizon: str, api_key: str, api_url: str, model_name: str) -> dict:
    """感知阶段"""
    logging.info("阶段1: 感知 - 收集市场数据")
    prompt = PERCEPTION_PROMPT.format(
        research_topic=topic,
        industry_focus=industry,
        time_horizon=horizon
    )
    for i in range(MAX_RETRIES):
        try:
            result_text = call_llm(prompt, api_key, api_url, model_name)
            return safe_json_parse(result_text, "market_overview")
        except Exception as e:
            logging.warning(f"感知阶段重试 {i+1}/{MAX_RETRIES}: {e}")
    logging.error("感知阶段最终失败，使用兜底数据")
    return {"market_overview": "数据获取失败", "key_indicators": {}, "recent_news": [], "industry_trends": {}}

def run_modeling(topic: str, industry: str, horizon: str, perception_data: dict, api_key: str, api_url: str, model_name: str) -> dict:
    """建模阶段"""
    logging.info("阶段2: 建模 - 构建内部模型")
    prompt = MODELING_PROMPT.format(
        research_topic=topic,
        industry_focus=industry,
        time_horizon=horizon,
        perception_data=json.dumps(perception_data, ensure_ascii=False)
    )
    for i in range(MAX_RETRIES):
        try:
            result_text = call_llm(prompt, api_key, api_url, model_name)
            return safe_json_parse(result_text, "market_state")
        except Exception as e:
            logging.warning(f"建模阶段重试 {i+1}/{MAX_RETRIES}: {e}")
    logging.error("建模阶段最终失败，使用兜底数据")
    return {"market_state": "模型构建失败", "economic_cycle": "未知", "risk_factors": [], "opportunity_areas": [], "market_sentiment": "中性"}

def run_reasoning(topic: str, industry: str, horizon: str, world_model: dict, api_key: str, api_url: str, model_name: str) -> dict:
    """推理阶段"""
    logging.info("阶段3: 推理 - 生成候选方案")
    prompt = REASONING_PROMPT.format(
        research_topic=topic,
        industry_focus=industry,
        time_horizon=horizon,
        world_model=json.dumps(world_model, ensure_ascii=False)
    )
    for i in range(MAX_RETRIES):
        try:
            result_text = call_llm(prompt, api_key, api_url, model_name)
            return safe_json_parse(result_text, "plans")
        except Exception as e:
            logging.warning(f"推理阶段重试 {i+1}/{MAX_RETRIES}: {e}")
    logging.error("推理阶段最终失败，使用兜底数据")
    return {"plans": "方案生成失败", "raw_output": ""}

def run_decision(topic: str, industry: str, horizon: str, world_model: dict, reasoning_plans: dict, api_key: str, api_url: str, model_name: str) -> dict:
    """决策阶段"""
    logging.info("阶段4: 决策 - 选择最优方案")
    prompt = DECISION_PROMPT.format(
        research_topic=topic,
        industry_focus=industry,
        time_horizon=horizon,
        world_model=json.dumps(world_model, ensure_ascii=False),
        reasoning_plans=json.dumps(reasoning_plans, ensure_ascii=False)
    )
    for i in range(MAX_RETRIES):
        try:
            result_text = call_llm(prompt, api_key, api_url, model_name)
            return safe_json_parse(result_text, "decision")
        except Exception as e:
            logging.warning(f"决策阶段重试 {i+1}/{MAX_RETRIES}: {e}")
    logging.error("决策阶段最终失败，使用兜底数据")
    return {"selected_plan_id": "默认方案", "investment_thesis": "决策生成失败", "supporting_evidence": [], "risk_assessment": "未知", "recommendation": "谨慎观望", "timeframe": horizon}

def run_report(topic: str, industry: str, horizon: str, perception_data: dict, world_model: dict, selected_plan: dict, api_key: str, api_url: str, model_name: str) -> str:
    """报告阶段"""
    logging.info("阶段5: 报告 - 生成完整报告")
    prompt = REPORT_PROMPT.format(
        research_topic=topic,
        industry_focus=industry,
        time_horizon=horizon,
        perception_data=json.dumps(perception_data, ensure_ascii=False),
        world_model=json.dumps(world_model, ensure_ascii=False),
        selected_plan=json.dumps(selected_plan, ensure_ascii=False)
    )
    for i in range(MAX_RETRIES):
        try:
            return call_llm(prompt, api_key, api_url, model_name)
        except Exception as e:
            logging.warning(f"报告阶段重试 {i+1}/{MAX_RETRIES}: {e}")
    logging.error("报告阶段最终失败")
    return "报告生成失败，请重试。"

def run_research_pipeline(topic: str, industry: str, horizon: str, api_key: str, api_url: str, model_name: str) -> dict:
    """运行完整五阶段分析流程"""
    # 阶段1: 感知
    perception_data = run_perception(topic, industry, horizon, api_key, api_url, model_name)
    
    # 阶段2: 建模
    world_model = run_modeling(topic, industry, horizon, perception_data, api_key, api_url, model_name)
    
    # 阶段3: 推理
    reasoning_plans = run_reasoning(topic, industry, horizon, world_model, api_key, api_url, model_name)
    
    # 阶段4: 决策
    selected_plan = run_decision(topic, industry, horizon, world_model, reasoning_plans, api_key, api_url, model_name)
    
    # 阶段5: 报告
    final_report = run_report(topic, industry, horizon, perception_data, world_model, selected_plan, api_key, api_url, model_name)
    
    return {
        "perception_data": perception_data,
        "world_model": world_model,
        "reasoning_plans": reasoning_plans,
        "selected_plan": selected_plan,
        "final_report": final_report
    }

# ==================== PDF生成 ====================

def generate_pdf_report(report_text: str, topic: str, industry: str, horizon: str) -> bytes:
    """生成PDF格式的投资报告"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=18*mm, bottomMargin=18*mm)
    
    # 字体设置
    font_name = 'Helvetica'
    try:
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
        font_name = 'STSong-Light'
    except:
        try:
            font_path = os.path.join(os.environ.get("SYSTEMROOT", "C:\\Windows"), "Fonts", "simhei.ttf")
            pdfmetrics.registerFont(TTFont('SimHei', font_path))
            font_name = 'SimHei'
        except:
            pass
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontName=font_name, fontSize=18, textColor=colors.HexColor('#2c3e50'), spaceAfter=8*mm, alignment=1)
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'], fontName=font_name, fontSize=10, textColor=colors.HexColor('#7f8c8d'), spaceAfter=6*mm, alignment=1)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontName=font_name, fontSize=10, leading=15, spaceAfter=3*mm, alignment=0)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontName=font_name, fontSize=13, textColor=colors.HexColor('#3498db'), spaceBefore=6*mm, spaceAfter=3*mm)
    
    story = []
    story.append(Paragraph("智能投研助手 - 投资分析报告", title_style))
    story.append(Paragraph(f"研究主题: {topic} | 行业焦点: {industry} | 时间范围: {horizon}", subtitle_style))
    story.append(Spacer(2*mm, 2*mm))
    
    if not report_text:
        story.append(Paragraph("（无报告内容）", body_style))
    else:
        for line in report_text.split('\n'):
            line = line.strip()
            if not line:
                story.append(Spacer(1*mm, 1*mm))
                continue
            if line.startswith('#') or (line[0].isdigit() and '.' in line[:5]):
                story.append(Paragraph(line.replace('#', '').strip(), heading_style))
            else:
                story.append(Paragraph(line, body_style))
    
    story.append(Spacer(8*mm, 8*mm))
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontName=font_name, fontSize=8, textColor=colors.HexColor('#95a5a6'), alignment=1)
    story.append(Paragraph(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 智能投研助手", footer_style))
    
    doc.build(story)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data

# ==================== Session State 初始化 ====================

if 'api_key' not in st.session_state: st.session_state.api_key = ""
if 'api_url' not in st.session_state.api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
if 'model_name' not in st.session_state.model_name = "qwen-plus"
if 'current_analysis' not in st.session_state.current_analysis = None
if 'analysis_complete' not in st.session_state.analysis_complete = False
if 'conversation_history' not in st.session_state.conversation_history = []
if 'show_history' not in st.session_state.show_history = False
if 'pdf_data' not in st.session_state.pdf_data = None
if 'pdf_ready' not in st.session_state.pdf_ready = False

HISTORY_FILE = "analysis_history.json"

def load_history():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logging.error(f"加载历史记录失败: {e}")
    return []

def save_history(history):
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"保存历史记录失败: {e}")

# ==================== 侧边栏 ====================

with st.sidebar:
    st.header("🔧 模型配置")
    api_key_input = st.text_input("API密钥", value=st.session_state.api_key, type="password")
    api_url_input = st.text_input("API地址", value=st.session_state.api_url)
    model_name_input = st.text_input("模型名称", value=st.session_state.model_name)
    
    if api_key_input: st.session_state.api_key = api_key_input
    if api_url_input: st.session_state.api_url = api_url_input
    if model_name_input: st.session_state.model_name = model_name_input
    
    st.markdown("---")
    st.header("📋 分析配置")
    
    PRESET_TOPICS = [
        "人工智能芯片市场", "新能源汽车产业链", "创新药研发管线", "跨境电商出海机遇",
        "低空经济产业链", "储能技术商业化", "人形机器人产业", "消费级AR/VR市场",
        "半导体设备国产化", "金融科技监管趋势", "碳中和绿色投资", "量子计算商用化"
    ]
    PRESET_INDUSTRIES = [
        "半导体", "人工智能", "新能源", "生物医药", "消费电子", "金融科技",
        "智能制造", "航空航天", "新材料", "互联网", "汽车", "传媒娱乐"
    ]
    
    topic_col1, topic_col2 = st.columns([1, 1])
    with topic_col1:
        topic_option = st.selectbox("选择研究主题", options=["自定义输入"] + PRESET_TOPICS, index=1, key="topic_preset")
    with topic_col2:
        if topic_option == "自定义输入":
            research_topic = st.text_input("输入研究主题", value="", key="topic_custom")
        else:
            research_topic = st.text_input("研究主题", value=topic_option, disabled=True, key="topic_fixed")
            research_topic = topic_option
    
    industry_col1, industry_col2 = st.columns([1, 1])
    with industry_col1:
        industry_option = st.selectbox("选择行业焦点", options=["自定义输入"] + PRESET_INDUSTRIES, index=1, key="industry_preset")
    with industry_col2:
        if industry_option == "自定义输入":
            industry_focus = st.text_input("输入行业焦点", value="", key="industry_custom")
        else:
            industry_focus = st.text_input("行业焦点", value=industry_option, disabled=True, key="industry_fixed")
            industry_focus = industry_option
    
    time_horizon = st.selectbox("时间范围", ["短期(1-3个月)", "中期(3-6个月)", "中期(6-12个月)", "长期(1年以上)"], index=2)
    
    analyze_button = st.button("🚀 开始智能分析", use_container_width=True, type="primary")
    
    st.markdown("---")
    st.header("📜 历史记录")
    if st.button("🗑️ 清空历史", use_container_width=True):
        st.session_state.conversation_history = []
        save_history([])
        st.success("历史记录已清空")
        st.rerun()

# ==================== 主内容区 ====================

st.header("📈 智能投资分析")

if analyze_button:
    if not st.session_state.api_key:
        st.error("⚠️ 请输入API密钥")
    else:
        progress_bar = st.progress(0, text="准备开始分析...")
        status_text = st.empty()
        
        try:
            # 阶段1: 感知 (20%)
            status_text.info("🔄 阶段1/5: 感知 - 收集市场数据...")
            progress_bar.progress(20, text="阶段1/5: 感知 - 收集市场数据")
            perception_data = run_perception(research_topic, industry_focus, time_horizon, st.session_state.api_key, st.session_state.api_url, st.session_state.model_name)
            
            # 阶段2: 建模 (40%)
            status_text.info("🔄 阶段2/5: 建模 - 构建内部模型...")
            progress_bar.progress(40, text="阶段2/5: 建模 - 构建内部模型")
            world_model = run_modeling(research_topic, industry_focus, time_horizon, perception_data, st.session_state.api_key, st.session_state.api_url, st.session_state.model_name)
            
            # 阶段3: 推理 (60%)
            status_text.info("🔄 阶段3/5: 推理 - 生成候选方案...")
            progress_bar.progress(60, text="阶段3/5: 推理 - 生成候选方案")
            reasoning_plans = run_reasoning(research_topic, industry_focus, time_horizon, world_model, st.session_state.api_key, st.session_state.api_url, st.session_state.model_name)
            
            # 阶段4: 决策 (80%)
            status_text.info("🔄 阶段4/5: 决策 - 选择最优方案...")
            progress_bar.progress(80, text="阶段4/5: 决策 - 选择最优方案")
            selected_plan = run_decision(research_topic, industry_focus, time_horizon, world_model, reasoning_plans, st.session_state.api_key, st.session_state.api_url, st.session_state.model_name)
            
            # 阶段5: 报告 (100%)
            status_text.info("🔄 阶段5/5: 报告 - 生成完整报告...")
            progress_bar.progress(95, text="阶段5/5: 报告 - 生成完整报告")
            final_report = run_report(research_topic, industry_focus, time_horizon, perception_data, world_model, selected_plan, st.session_state.api_key, st.session_state.api_url, st.session_state.model_name)
            
            # 完成
            progress_bar.progress(100, text="✅ 分析完成")
            status_text.success("✅ 五阶段分析完成！")
            
            # 保存结果
            st.session_state.current_analysis = {
                "research_topic": research_topic,
                "industry_focus": industry_focus,
                "time_horizon": time_horizon,
                "perception_data": perception_data,
                "world_model": world_model,
                "reasoning_plans": reasoning_plans,
                "selected_plan": selected_plan,
                "final_report": final_report
            }
            st.session_state.analysis_complete = True
            
            # 保存历史
            history_entry = {
                "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "research_topic": research_topic,
                "industry_focus": industry_focus,
                "time_horizon": time_horizon,
                "final_report": final_report[:500] + "..." if len(final_report) > 500 else final_report
            }
            st.session_state.conversation_history.insert(0, history_entry)
            if len(st.session_state.conversation_history) > 20:
                st.session_state.conversation_history = st.session_state.conversation_history[:20]
            save_history(st.session_state.conversation_history)
            
            # 清除PDF状态
            st.session_state.pdf_data = None
            st.session_state.pdf_ready = False
            
            st.rerun()
            
        except Exception as e:
            progress_bar.empty()
            status_text.error(f"❌ 分析过程中出错: {e}")
            logging.error(f"分析流程错误: {e}")

# ==================== 结果显示 ====================

if st.session_state.analysis_complete and st.session_state.current_analysis:
    analysis = st.session_state.current_analysis
    
    # 阶段标签
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.markdown('<div style="background:#28a745;color:white;padding:8px;border-radius:5px;text-align:center;font-size:12px;">✅ 感知阶段</div>', unsafe_allow_html=True)
    with col2: st.markdown('<div style="background:#28a745;color:white;padding:8px;border-radius:5px;text-align:center;font-size:12px;">✅ 建模阶段</div>', unsafe_allow_html=True)
    with col3: st.markdown('<div style="background:#28a745;color:white;padding:8px;border-radius:5px;text-align:center;font-size:12px;">✅ 推理阶段</div>', unsafe_allow_html=True)
    with col4: st.markdown('<div style="background:#28a745;color:white;padding:8px;border-radius:5px;text-align:center;font-size:12px;">✅ 决策阶段</div>', unsafe_allow_html=True)
    with col5: st.markdown('<div style="background:#28a745;color:white;padding:8px;border-radius:5px;text-align:center;font-size:12px;">✅ 报告阶段</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 最终投资报告
    st.subheader("📄 最终投资报告")
    st.markdown(f"**主题:** {analysis['research_topic']} ({analysis['industry_focus']}) | **范围:** {analysis['time_horizon']}")
    st.markdown('<div class="report-content">' + analysis['final_report'].replace('\n', '<br>') + '</div>', unsafe_allow_html=True)
    
    # PDF导出
    col1, col2 = st.columns([1, 1])
    with col1:
        if not st.session_state.pdf_ready:
            if st.button("📥 导出PDF报告", use_container_width=True):
                with st.spinner("正在生成PDF..."):
                    try:
                        pdf_data = generate_pdf_report(
                            analysis['final_report'],
                            analysis['research_topic'],
                            analysis['industry_focus'],
                            analysis['time_horizon']
                        )
                        st.session_state.pdf_data = pdf_data
                        st.session_state.pdf_ready = True
                        st.success("PDF生成成功！")
                        st.rerun()
                    except Exception as e:
                        st.error(f"PDF生成失败: {e}")
        else:
            st.download_button(
                label="💾 下载PDF报告",
                data=st.session_state.pdf_data,
                file_name=f"投研报告_{analysis['research_topic']}_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    with col2:
        if st.session_state.pdf_ready:
            if st.button("🔄 重新生成PDF", use_container_width=True):
                st.session_state.pdf_ready = False
                st.session_state.pdf_data = None
                st.rerun()
    
    # 清空结果
    if st.button("🗑️ 清空分析结果", use_container_width=True):
        st.session_state.current_analysis = None
        st.session_state.analysis_complete = False
        st.session_state.pdf_data = None
        st.session_state.pdf_ready = False
        st.rerun()
    
    # 中间结果展示
    with st.expander("📊 查看中间分析结果"):
        st.subheader("1. 市场感知数据")
        st.json(analysis['perception_data'])
        
        st.subheader("2. 市场模型")
        st.json(analysis['world_model'])
        
        st.subheader("3. 候选方案")
        st.json(analysis['reasoning_plans'])
        
        st.subheader("4. 最优决策")
        st.json(analysis['selected_plan'])

# ==================== 历史记录展示 ====================

if st.session_state.conversation_history:
    with st.expander(f"📜 历史分析记录 ({len(st.session_state.conversation_history)}条)"):
        for idx, entry in enumerate(st.session_state.conversation_history):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{entry['timestamp']}** - {entry['research_topic']} ({entry['industry_focus']}) | {entry['time_horizon']}")
            with col2:
                if st.button("🗑️ 删除", key=f"del_{entry['id']}"):
                    st.session_state.conversation_history.pop(idx)
                    save_history(st.session_state.conversation_history)
                    st.rerun()
            st.markdown(f"> {entry['final_report'][:200]}...")
            st.markdown("---")
else:
    st.info("📭 暂无历史分析记录")
