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
from tools import cafe_marketing_tool, revisit_rate_analysis_tool, store_strength_weakness_tool, search_merchant_tool, get_score_from_raw

# 환경변수
ASSETS = Path("assets")
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

system_prompt = """당신은 사용자의 요청을 분석하여 최적의 솔루션을 제공하는 AI 데이터 분석가입니다.

**[핵심 임무]**
사용자의 질문에서 '분석 의도'와 '가게 ID'를 파악한 후, 단 하나의 가장 적합한 분석 도구를 선택하여 실행하고, 그 결과를 전문적인 보고서 형태로 제공해야 합니다.

**[의사결정 가이드라인]**
사용자의 질문 의도를 파악할 때 아래 가이드라인을 참고하여 최적의 도구를 선택하세요.

- 만약 사용자가 **'카페'의 '고객 특성', '마케팅 채널', '홍보 방안'**에 대해 묻는다면,
  -> `cafe_marketing_tool`이 가장 적합합니다.

- 만약 사용자가 **'재방문율'을 높이거나 '단골 고객' 확보**를 위한 아이디어를 묻는다면,
  -> `revisit_rate_analysis_tool`이 가장 적합합니다.

- 만약 사용자가 가게의 **'가장 큰 문제점', '종합적인 진단', '강점과 약점'**에 대해 포괄적으로 묻는다면,
  -> `store_strength_weakness_tool`이 가장 적합합니다.
  
- 만약 위 경우에 해당하지 않고, 단순히 **가게의 '기본 정보'**를 묻는다면,
  -> `search_merchant_tool`을 사용하세요.

**[행동 순서]**
1. 사용자 질문에서 가게 ID를 찾습니다. (예: '(가게 ID: ABC12345)')
2. 위 가이드라인에 따라 질문의 핵심 의도를 분석하고, 가장 적합한 도구 **단 하나**를 결정합니다.
3. 찾아낸 가게 ID를 결정한 도구의 입력값으로 사용하여 실행합니다.
4. 만약 가게 ID가 없다면, 사용자에게 정중하게 ID를 요청합니다.
"""
greeting = "안녕하세요, AI 비밀상담사입니다. 분석하고 싶은 내용과 함께 가게의 고유 ID를 알려주세요. 예: '우리 가게 강점 알려줘 (가게 ID: ABC12345)'"

# 데이터 로딩 함수
@st.cache_data 
def load_data():
    """3개의 핵심 데이터셋을 로드하고 전처리하는 함수"""
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
        
        # 2. 카페 가맹점별 주요고객 데이터 로드
        df_cafe_customers = pd.read_csv(data_path / "카페_가맹점별_주요고객.csv", encoding='utf-8-sig')
        
        # 3. AI상담사 핵심전략 프롬프트 데이터 로드
        df_prompt_dna = pd.read_csv(data_path / "AI상담사_핵심전략_프롬프트.csv", encoding='utf-8-sig')
        
        return df_all_join, df_cafe_customers, df_prompt_dna
        
    except Exception as e:
        st.error(f"데이터 로딩 중 오류가 발생했습니다: {e}")
        return None, None, None


# Streamlit App UI
@st.cache_data 
def load_image(name: str):
    return Image.open(ASSETS / name)

st.set_page_config(page_title="2025년 빅콘테스트 AI데이터 활용분야 - 맛집을 수호하는 AI비밀상담사")

def clear_chat_history():
    st.session_state.messages = [SystemMessage(content=system_prompt), AIMessage(content=greeting)]

# 사이드바
with st.sidebar:
    st.image(load_image("shc_ci_basic_00.png"), width='stretch')
    st.markdown("<p style='text-align: center;'>2025 Big Contest</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>AI DATA 활용분야</p>", unsafe_allow_html=True)
    st.write("")
    col1, col2, col3 = st.columns([1,2,1])  # 비율 조정 가능
    with col2:
        st.button('Clear Chat History', on_click=clear_chat_history)

# 헤더
st.title("신한카드 소상공인 🔑 비밀상담소")
st.subheader("#우리동네 #숨은맛집 #소상공인 #마케팅 #전략 .. 🤤")
st.image(load_image("image_gen3.png"), width='stretch', caption="🌀 머리아픈 마케팅 📊 어떻게 하면 좋을까?")
st.write("")



# LLM 모델 선택
llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # 최신 Gemini 2.5 Flash 모델
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1
    )

class ToolExecutor:
    def __init__(self, df_all, df_cafe, df_dna):
        self.df_all_join = df_all
        self.df_cafe_customers = df_cafe
        self.df_prompt_dna = df_dna

    def search_merchant_tool(self, store_id: str) -> str:
        """가게의 기본 정보만 간단히 조회할 때 사용합니다."""
        try:
            return search_merchant_tool.invoke({"merchant_name": store_id, "df_all_join": self.df_all_join})
        except Exception as e:
            return f"🚨 가맹점 검색 중 오류가 발생했습니다: {str(e)}"

    def cafe_marketing_tool(self, store_id: str) -> str:
        """'카페' 가맹점의 고객 특성 분석 및 마케팅/홍보 전략을 제안할 때 사용합니다."""
        try:
            return cafe_marketing_tool.invoke({
                "store_id": store_id, 
                "df_cafe_customers": self.df_cafe_customers, 
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

    def get_all_tools(self):
        from langchain_core.tools import tool
        
        @tool
        def search_merchant_wrapper(store_id: str) -> str:
            """가게의 기본 정보만 간단히 조회할 때 사용합니다."""
            return self.search_merchant_tool(store_id)
        
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
        
        return [
            search_merchant_wrapper,
            cafe_marketing_wrapper,
            revisit_rate_analysis_wrapper,
            store_strength_weakness_wrapper
        ]


# 데이터 로드
df_all_join, df_cafe_customers, df_prompt_dna = load_data()

if df_all_join is not None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content=system_prompt),
            AIMessage(content=greeting)
        ]

    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.2)
    
    tool_executor = ToolExecutor(df_all_join, df_cafe_customers, df_prompt_dna)
    agent = create_react_agent(llm, tool_executor.get_all_tools())

    if query := st.chat_input("질문과 함께 가게 ID를 입력하세요. (예: 재방문율 분석해줘 (가게 ID: ABC12345))"):
        st.session_state.messages.append(HumanMessage(content=query))
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
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
