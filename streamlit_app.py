import streamlit as st
import pandas as pd
import numpy as np

from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

from PIL import Image
from pathlib import Path

# 새로 만든 분석 도구들 import
from tools import cafe_marketing_tool, revisit_rate_analysis_tool, store_strength_weakness_tool, floating_population_strategy_tool, lunch_turnover_strategy_tool, get_score_from_raw

# 환경변수
ASSETS = Path("assets")
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

system_prompt = """당신은 사용자의 요청을 분석하여 최적의 솔루션을 제공하는 AI 데이터 분석가입니다.

**[핵심 임무]**
사용자의 질문에서 '분석 의도'와 '가게 ID'를 파악한 후, 아래의 '질문 의도별 도구 선택 가이드'에 따라 가장 적합한 분석 도구를 단 하나만 선택하여 실행하고, 그 결과를 전문적인 보고서 형태로 제공해야 합니다.

**[질문 의도별 도구 선택 가이드]**
사용자 질문의 핵심 의도를 파악하고, 아래 가이드에 따라 가장 적합한 도구를 하나만 선택하여 실행하세요.

- **카페 고객 분석**: 질문에 **'카페'** 업종이 명시되고, **'고객 특성', '마케팅 채널', '홍보 방안'** 중 하나라도 언급되면
  -> `cafe_marketing_tool` 사용

- **재방문율 개선**: 질문에 **'재방문율', '단골 고객', '재방문'** 키워드가 있고, 30% 이하 개선 목적이면
  -> `revisit_rate_analysis_tool` 사용

- **유동인구 활용 전략**: 질문에 **'유동인구', '지하철역', '출퇴근', '재방문 유도'** 키워드가 있으면
  -> `floating_population_strategy_tool` 사용

- **점심시간 효율 증대**: 질문에 **'직장인', '점심시간', '회전율', '효율'** 키워드가 있으면
  -> `lunch_turnover_strategy_tool` 사용

- **종합 강점/약점 진단**: 질문에 **'가장 큰 문제점', '종합적인 진단', '강점과 약점', '문제점'** 키워드가 있으면
  -> `store_strength_weakness_tool` 사용

**[행동 순서]**
1.  사용자 질문에서 가게 ID를 찾습니다. (예: '(가게 ID: ABC12345)')
2.  위 가이드라인에 따라 질문의 핵심 의도를 분석하고, 가장 적합한 도구 **단 하나**를 결정합니다.
3.  찾아낸 가게 ID를 결정한 도구의 입력값으로 사용하여 실행합니다.
4.  만약 가게 ID가 없다면, 사용자에게 정중하게 ID를 요청합니다.

**[최종 결과물 생성 규칙]**
이것은 매우 중요한 규칙입니다. 반드시 따르세요.

1.  도구를 실행하여 얻은 결과(Observation)를 바탕으로 최종 답변을 생성합니다.
2.  가장 먼저, 도구 결과에 포함된 **'가맹점 기본 정보' 블록을 일체의 수정이나 생략 없이 그대로** 최종 답변의 맨 앞에 붙여넣어야 합니다. 이것은 **어떤 도구가 실행되든 예외 없이 적용**되는 규칙입니다.
3.  그 바로 아래에, 도구의 나머지 분석 내용을 바탕으로 전문적인 분석 보고서를 자연스럽게 이어서 작성하세요. 전체 답변이 하나의 통일된 보고서 형식이어야 합니다.
"""
greeting = "안녕하세요! 사장님의 든든한 AI 성장 파트너, 솔비(SOL-B)입니다. 질문과 함께 가게 ID를 알려주세요."

# 데이터 로딩 함수
@st.cache_data 
def load_data():
    """5개의 핵심 데이터셋을 로드하고 전처리하는 함수"""
    try:
        # 데이터 파일 경로
        data_path = Path("data")
        
        # 1. 전체 JOIN 데이터 로드 및 전처리
        df_all_join = pd.read_csv(data_path / "전체JOIN_업종정규화_v2.csv", encoding='utf-8-sig')
        df_all_join.replace(-999999.9, np.nan, inplace=True)
        
        # 숫자형 컬럼 변환
        numeric_cols = ['MCT_UE_CLN_NEW_RAT', 'MCT_UE_CLN_REU_RAT', 'RC_M1_SHC_RSD_UE_CLN_RAT', 'DLV_SAA_RAT', 'M12_SME_RY_SAA_PCE_RT']
        for col in numeric_cols:
            if col in df_all_join.columns:
                df_all_join[col] = pd.to_numeric(df_all_join[col], errors='coerce')
        
        # 점수 컬럼 생성
        score_cols = ['MCT_OPE_MS_CN', 'RC_M1_TO_UE_CT', 'RC_M1_SAA', 'RC_M1_AV_NP_AT']
        for col in score_cols:
            if col in df_all_join.columns:
                df_all_join[f'{col}_SCORE'] = df_all_join[col].apply(get_score_from_raw)
        
        # 필수 컬럼이 있는 행만 유지
        df_all_join.dropna(subset=['ENCODED_MCT', '업종_정규화2_대분류'], inplace=True)
        
        # 2. AI상담사 핵심전략 프롬프트 데이터 로드
        df_prompt_dna = pd.read_csv(data_path / "AI상담사_핵심전략_프롬프트.csv", encoding='utf-8-sig')
        
        # 3. 특화 질문용 유동인구 데이터 로드 (7개 파일)
        df_gender_age = pd.read_csv(data_path / "성별연령대별_유동인구.csv", encoding='utf-8-sig')
        df_gender_age_selected = pd.read_csv(data_path / "성별연령대별_유동인구_선택영역.csv", encoding='utf-8-sig')
        df_weekday_weekend = pd.read_csv(data_path / "요일별_유동인구.csv", encoding='utf-8-sig')
        df_weekday_weekend_selected = pd.read_csv(data_path / "요일별_유동인구_선택영역.csv", encoding='utf-8-sig')
        df_dayofweek = pd.read_csv(data_path / "요일별_유동인구.csv", encoding='utf-8-sig')
        df_timeband = pd.read_csv(data_path / "시간대별_유동인구.csv", encoding='utf-8-sig')
        df_timeband_selected = pd.read_csv(data_path / "시간대별_유동인구_선택영역.csv", encoding='utf-8-sig')
        df_workplace_population = pd.read_csv(data_path / "성별연령대별_직장인구.csv", encoding='utf-8-sig')
        
        return df_all_join, df_prompt_dna, df_gender_age, df_gender_age_selected, df_weekday_weekend, df_weekday_weekend_selected, df_dayofweek, df_timeband, df_timeband_selected, df_workplace_population
        
    except Exception as e:
        st.error(f"데이터 로딩 중 오류가 발생했습니다: {e}")
        return None, None, None, None, None, None, None, None, None, None


# Streamlit App UI
@st.cache_data 
def load_image(name: str):
    return Image.open(ASSETS / name)

def clear_chat_history():
    st.session_state.messages = [
        SystemMessage(content=system_prompt),
        AIMessage(content=greeting)
    ]

# 페이지 설정
st.set_page_config(
    page_title="솔비(SOL-B) | AI 성장 파트너",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 사이드바
with st.sidebar:
    st.image(load_image("shc_ci_basic_00.png"), width=300)
    st.title("✨ AI 성장 파트너, 솔비(SOL-B)")
    st.info("🏆  **2025 신한카드 빅콘테스트** 출품작")
    st.write("")
    st.button('새로운 상담 시작하기', on_click=clear_chat_history, use_container_width=True)

# 메인 컨테이너
with st.container():
    # 헤더 섹션
    st.markdown('<h1 class="main-title"> 소상공인 AI 성장 파트너,  솔비(SOL-B)</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">✨복잡한 마케팅 고민, 이제 AI 파트너에게 맡겨주세요.✨</p>', unsafe_allow_html=True)
    
    # 메인 이미지
    st.image(load_image("AI_Marketing_Counselor.png"), width='stretch', caption="✨사장님의 비즈니스 고민, 솔비가 데이터로 해결해 드릴게요.")
    
    # 구분선
    st.divider()

# LLM 모델은 데이터 로드 후 초기화

class ToolExecutor:
    def __init__(self, df_all, df_dna, df_gender_age, df_gender_age_selected, df_weekday_weekend, df_weekday_weekend_selected, df_dayofweek, df_timeband, df_timeband_selected, df_workplace_population):
        self.df_all_join = df_all
        self.df_prompt_dna = df_dna
        self.df_gender_age = df_gender_age
        self.df_gender_age_selected = df_gender_age_selected
        self.df_weekday_weekend = df_weekday_weekend
        self.df_weekday_weekend_selected = df_weekday_weekend_selected
        self.df_dayofweek = df_dayofweek
        self.df_timeband = df_timeband
        self.df_timeband_selected = df_timeband_selected
        self.df_workplace_population = df_workplace_population


    def cafe_marketing_tool(self, store_id: str) -> str:
        """'카페' 가맹점의 고객 특성 분석 및 마케팅/홍보 전략을 제안할 때 사용합니다."""
        try:
            return cafe_marketing_tool.invoke({
                "store_id": store_id, 
                "df_all_join": self.df_all_join, 
                "df_prompt_dna": self.df_prompt_dna
            })
        except Exception as e:
            return f"🚨 카페 마케팅 분석 중 오류가 발생했습니다: {str(e)}"

    def revisit_rate_analysis_tool(self, store_id: str) -> str:
        """가맹점의 '재방문율'을 높이는 아이디어를 제안할 때 사용합니다."""
        try:
            return revisit_rate_analysis_tool.invoke({
                "store_id": store_id, 
                "df_all_join": self.df_all_join, 
                "df_prompt_dna": self.df_prompt_dna
            })
        except Exception as e:
            return f"🚨 재방문율 분석 중 오류가 발생했습니다: {str(e)}"

    def store_strength_weakness_tool(self, store_id: str) -> str:
        """가맹점의 '가장 큰 문제점'을 진단하고 강점/약점 기반의 해결책을 제안할 때 사용합니다."""
        try:
            return store_strength_weakness_tool.invoke({
                "store_id": store_id, 
                "df_all_join": self.df_all_join
            })
        except Exception as e:
            return f"🚨 전방위 분석 중 오류가 발생했습니다: {str(e)}"

    def floating_population_strategy_tool(self, store_id: str) -> str:
        """지하철역 인근 가맹점의 유동인구 데이터를 분석하여 재방문 유도 전략을 제안할 때 사용합니다."""
        try:
            return floating_population_strategy_tool.invoke({
                "store_id": store_id,
                "df_all_join": self.df_all_join,
                "df_gender_age": self.df_gender_age_selected,
                "df_weekday_weekend": self.df_weekday_weekend_selected,
                "df_dayofweek": self.df_dayofweek,
                "df_timeband": self.df_timeband_selected
            })
        except Exception as e:
            return f"🚨 유동인구 전략 분석 중 오류가 발생했습니다: {str(e)}"

    def lunch_turnover_strategy_tool(self, store_id: str) -> str:
        """직장인 상권 가맹점의 점심시간 회전율 극대화 전략을 제안할 때 사용합니다."""
        try:
            return lunch_turnover_strategy_tool.invoke({
                "store_id": store_id,
                "df_all_join": self.df_all_join,
                "df_gender_age": self.df_gender_age,
                "df_weekday_weekend": self.df_weekday_weekend,
                "df_dayofweek": self.df_dayofweek,
                "df_timeband": self.df_timeband
            })
        except Exception as e:
            return f"🚨 점심시간 회전율 전략 분석 중 오류가 발생했습니다: {str(e)}"

    def get_all_tools(self):
        from langchain_core.tools import tool
        
        @tool
        def cafe_marketing_wrapper(store_id: str) -> str:
            """'카페' 가맹점의 고객 특성 분석 및 마케팅/홍보 전략을 제안할 때 사용합니다."""
            return self.cafe_marketing_tool(store_id)
        
        @tool
        def revisit_rate_analysis_wrapper(store_id: str) -> str:
            """가맹점의 '재방문율'을 높이는 아이디어를 제안할 때 사용합니다."""
            return self.revisit_rate_analysis_tool(store_id)
        
        @tool
        def store_strength_weakness_wrapper(store_id: str) -> str:
            """가맹점의 '가장 큰 문제점'을 진단하고 강점/약점 기반의 해결책을 제안할 때 사용합니다."""
            return self.store_strength_weakness_tool(store_id)
        
        @tool
        def floating_population_strategy_wrapper(store_id: str) -> str:
            """지하철역 인근 가맹점의 유동인구 데이터를 분석하여 재방문 유도 전략을 제안할 때 사용합니다."""
            return self.floating_population_strategy_tool(store_id)
        
        @tool
        def lunch_turnover_strategy_wrapper(store_id: str) -> str:
            """직장인 상권 가맹점의 점심시간 회전율 극대화 전략을 제안할 때 사용합니다."""
            return self.lunch_turnover_strategy_tool(store_id)
        
        return [
            cafe_marketing_wrapper,
            revisit_rate_analysis_wrapper,
            store_strength_weakness_wrapper,
            floating_population_strategy_wrapper,
            lunch_turnover_strategy_wrapper
        ]


# 데이터 로드
df_all_join, df_prompt_dna, df_gender_age, df_gender_age_selected, df_weekday_weekend, df_weekday_weekend_selected, df_dayofweek, df_timeband, df_timeband_selected, df_workplace_population = load_data()

if df_all_join is not None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content=system_prompt),
            AIMessage(content=greeting)
        ]

    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user", avatar="🧑‍💼"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant", avatar="🤖"):
                st.write(message.content)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.2)
    
    tool_executor = ToolExecutor(df_all_join, df_prompt_dna, df_gender_age, df_gender_age_selected, df_weekday_weekend, df_weekday_weekend_selected, df_dayofweek, df_timeband, df_timeband_selected, df_workplace_population)
    agent = create_react_agent(llm, tool_executor.get_all_tools())

    if query := st.chat_input("질문과 함께 가게 ID를 입력하세요.(예: 재방문율 분석해줘 (가게 ID: ABC12345))"):
        st.session_state.messages.append(HumanMessage(content=query))
        with st.chat_message("user", avatar="🧑‍💼"):
            st.write(query)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("AI 비밀상담사가 분석 중입니다..."):
                try:
                    response = agent.invoke({"messages": st.session_state.messages})
                    reply = response["messages"][-1].content
                    st.session_state.messages.append(AIMessage(content=reply))
                    st.write(reply)
                except Exception as e:
                    error_msg = f"죄송합니다, 분석 중 오류가 발생했습니다: {e}"
                    st.session_state.messages.append(AIMessage(content=error_msg))
                    st.error(error_msg)
else:
    st.warning("데이터 파일을 로드하지 못했습니다. `data` 폴더에 필요한 파일이 있는지 확인해주세요.")
# CSS 스타일링
st.markdown("""
<style>
    /* 신한카드 시그니처 블루 색상 */
    .main-title {
        color: #2A69B3;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        color: #666;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* 사이드바 스타일링 */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* 버튼 스타일링 */
    .stButton > button {
        background-color: #2A69B3;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #1e4d8c;
    }
    
    /* 채팅 메시지 스타일링 */
    .stChatMessage {
        border-radius: 12px;
        margin: 1rem 0;
    }
    /* 구분선 스타일링 */
    .stDivider {
        border-color: #2A69B3;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)