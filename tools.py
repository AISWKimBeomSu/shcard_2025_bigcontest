"""
AI 비밀상담사 분석 도구 모음
3개의 독립적인 분석 모델을 LangChain Tool로 리팩토링
"""

import pandas as pd
import numpy as np
import requests
import json
import os
from typing import Dict, Any, List, Tuple
import google.generativeai as genai
from langchain_core.tools import tool

# =============================================================================
# 공통 유틸리티 함수들
# =============================================================================

def call_gemini_llm(prompt: str) -> str:
    """Gemini 2.5 Flash LLM을 호출하여 응답을 반환하는 함수"""
    try:
        # API 키 설정
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            return "🚨 오류: Google API 키가 설정되지 않았습니다."
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"🚨 LLM 호출 중 오류가 발생했습니다: {str(e)}"

def get_score_from_raw(tier_string):
    """
    '1_10%이하' 또는 '6_90%초과' 같은 구간(tier) 문자열에서 
    앞의 숫자(1~6)를 추출하여 반환합니다. 
    이 숫자는 '낮을수록 좋음'을 의미합니다. 
    """
    if pd.isna(tier_string):
        return np.nan
    try:
        # '1_10%이하' -> '1' -> 1
        score = int(str(tier_string).split('_')[0])
        return score
    except (ValueError, IndexError, TypeError):
        # 예외 발생 시 (e.g., 'N/A' 또는 잘못된 형식)
        return np.nan

def translate_metric(metric_type, raw_value):
    """지표를 사람이 이해하기 쉬운 텍스트로 변환"""
    if pd.isna(raw_value):
        return "정보 없음"
    translation_map = {
        "tenure": {"10%이하": "상위 10% 이내 (가장 오래 운영)", "10-25%": "상위 10-25%", "25-50%": "중상위 25-50%", "50-75%": "중하위 50-75%", "75-90%": "하위 10-25%", "90%초과": "하위 10% 이내 (가장 최근 시작)"},
        "level": {"10%이하": "상위 10% 이내 (가장 높음)", "10-25%": "상위 10-25%", "25-50%": "중상위 25-50%", "50-75%": "중하위 50-75%", "75-90%": "하위 10-25%", "90%초과": "하위 10% 이내 (가장 낮음)"}
    }
    explanation_map = translation_map.get(metric_type, {})
    for key, explanation in explanation_map.items():
        if key in str(raw_value):
            return f"{raw_value} ({explanation})"
    return str(raw_value)

def score_to_level_text(score):
    """점수를 레벨 텍스트로 변환"""
    if pd.isna(score):
        return "정보 없음"
    if score >= 5.5:
        return "최상위 10% 수준"
    if score >= 4.5:
        return "상위 10-25% 수준"
    if score >= 3.5:
        return "중상위 25-50% 수준"
    if score >= 2.5:
        return "중하위 50-75% 수준"
    if score >= 1.5:
        return "하위 10-25% 수준"
    return "하위 10% 수준"

def _get_store_basic_info(store_id: str, df_all_join: pd.DataFrame) -> tuple[str, pd.Series | None]:
    """
    가맹점 ID로 기본 정보를 조회하고, 포맷팅된 리포트와 데이터 Series를 반환하는 내부 헬퍼 함수.
    가맹점을 찾지 못하면 오류 메시지와 None을 반환한다.
    """
    store_data = df_all_join[df_all_join['ENCODED_MCT'] == store_id]
    
    if store_data.empty:
        error_report = f"""
======================================================================
      🚨 분석 시작 불가 - 가맹점 정보 없음
======================================================================

'({store_id})'에 해당하는 가맹점 정보를 찾을 수 없습니다.
가게 ID를 다시 확인해주세요.

"""
        return error_report, None

    latest_result = store_data.sort_values(by='TA_YM', ascending=False).iloc[0]
    
    # 기본 정보 추출
    store_name = latest_result.get('MCT_NM', '정보 없음')
    industry = latest_result.get('업종_정규화2_대분류', '정보 없음')
    commercial_area = latest_result.get('HPSN_MCT_BZN_CD_NM', '비상권')
    operation_period = latest_result.get('MCT_OPE_MS_CN', '정보 없음')
    latest_month = latest_result.get('TA_YM', '정보 없음')
    
    # 매출 관련 정보
    revenue_level = latest_result.get('RC_M1_SAA', '정보 없음')
    customer_count_level = latest_result.get('RC_M1_UE_CUS_CN', '정보 없음')
    avg_amount_level = latest_result.get('RC_M1_AV_NP_AT', '정보 없음')
    
    # 고객 비율 정보
    new_customer_ratio = latest_result.get('MCT_UE_CLN_NEW_RAT', 0)
    revisit_ratio = latest_result.get('MCT_UE_CLN_REU_RAT', 0)
    resident_ratio = latest_result.get('RC_M1_SHC_RSD_UE_CLN_RAT', 0)
    delivery_ratio = latest_result.get('DLV_SAA_RAT', 0)
    
    # 기본 정보 '내용' 생성 (헤더 제외)
    basic_info_content = f"""
### 📋 기본 정보
- **가맹점명:** {store_name}
- **업종:** {industry}
- **상권:** {commercial_area if pd.notna(commercial_area) else '비상권'}
- **운영 기간:** {operation_period}
- **최신 데이터:** {latest_month}

### 💰 매출 현황 (최신월 기준)
- **매출 수준:** {revenue_level}
- **방문 고객 수:** {customer_count_level}
- **객단가 수준:** {avg_amount_level}

### 👥 고객 분석 (최신월 기준)
- **신규 고객 비율:** {new_customer_ratio:.1f}%
- **재방문 고객 비율:** {revisit_ratio:.1f}%
- **거주 고객 비율:** {resident_ratio:.1f}%
- **배달 매출 비율:** {delivery_ratio:.1f}%

---
""" # 구분을 위해 가로선을 추가합니다.
    return basic_info_content, latest_result

# =============================================================================
# 모델 1: 카페 업종 주요 고객 분석 및 마케팅 채널/홍보안 추천
# =============================================================================

# AI의 지식 베이스 - 페르소나 맵
PERSONA_MAP = {
    'M12_FME_1020_RAT': {'name': '디지털 큐레이터', 'desc': '10-20대 여성', 'features': 'SNS 공유를 통한 정체성 형성, 디토 소비, 가성비와 가심비의 전략적 혼합'},
    'M12_MAL_1020_RAT': {'name': '트렌드 검증가', 'desc': '10-20대 남성', 'features': '온라인 트렌드의 오프라인 경험 및 검증, 가성비 중시, 베이스캠프로서의 공간 활용'},
    'M12_FME_30_RAT': {'name': '전략적 최적화 전문가', 'desc': '30대 여성', 'features': '시간/비용/에너지의 최적화, 가치 경험 극대화, 분초사회'},
    'M12_MAL_30_RAT': {'name': '효율 추구 프로페셔널', 'desc': '30대 남성', 'features': '업무와 일상 속 모든 접점에서 시간과 노력 절약, 기능적 소비'},
    'M12_FME_40_RAT': {'name': '가족 웰빙 설계자', 'desc': '40대 여성', 'features': '가족의 건강과 경험에 대한 프리미엄 가치 투자, 커뮤니티 정보 교류'},
    'M12_MAL_40_RAT': {'name': '안정 추구 리더', 'desc': '40대 남성', 'features': '신뢰할 수 있는 브랜드를 통한 위험 최소화, 검증된 품질 선호'},
    'M12_FME_50_RAT': {'name': '커뮤니티 앵커', 'desc': '50대 여성', 'features': '사회적 관계망의 중심으로서 소통과 교류 주도, 편안함과 관계 중시'},
    'M12_MAL_50_RAT': {'name': '로컬 허브', 'desc': '50대 남성', 'features': '단골 문화를 통해 지역 커뮤니티의 구심점 역할, 익숙함 선호'},
    'M12_FME_60_RAT': {'name': '웰니스 라이프 추구자', 'desc': '60대 이상 여성', 'features': '신체적/정서적 웰빙을 위한 적극적인 소비와 활동'},
    'M12_MAL_60_RAT': {'name': '경험 가치 투자자', 'desc': '60대 이상 남성', 'features': '축적된 자산을 통해 관계와 의미 있는 경험에 투자'}
}

@tool
def customer_based_marketing_tool(store_id: str, df_all_join: pd.DataFrame, df_prompt_dna: pd.DataFrame) -> str:
    """
    카페 업종 가맹점의 주요 방문 고객 특성을 데이터 기반으로 분석하고, 
    가장 효과적인 마케팅 채널과 구체적인 홍보 문구를 추천하는 전문 도구.
    '카페', '고객 분석', '홍보' 관련 질문에 사용된다.
    
    Args:
        store_id: 분석할 가맹점 ID
        df_all_join: 전체 JOIN 데이터
        df_prompt_dna: AI상담사 핵심전략 프롬프트 데이터
    
    Returns:
        분석 결과 및 마케팅 전략 제안 리포트
    """
    try:
        # 1. 공통 헬퍼 함수 호출 (반환 변수명 변경)
        basic_info_content, latest_store_data = _get_store_basic_info(store_id, df_all_join)
        
        if latest_store_data is None:
            return basic_info_content # 오류 메시지는 그대로 반환

        # 2. 카페 업종 확인
        if latest_store_data['업종_정규화2_대분류'] != '카페':
            return basic_info_content + f"\n🚨 분석 실패: '{store_id}' 가맹점은 '카페' 업종이 아닙니다."

        # 1단계: 데이터 분석 엔진
        persona_columns = list(PERSONA_MAP.keys())

        store_df = df_all_join[(df_all_join['ENCODED_MCT'] == store_id) & (df_all_join['업종_정규화2_대분류'] == '카페')].copy()

        analysis_df = store_df[persona_columns]
        store_persona_data = analysis_df.mean()
        
        store_persona_data = store_persona_data.dropna()
        if store_persona_data.empty:
             return f"분석 실패: '{store_id}' 가맹점의 유효한 고객 비중 데이터를 찾을 수 없습니다."

        max_value = store_persona_data.max()
        if pd.isna(max_value):
             return f"분석 실패: '{store_id}' 가맹점의 최대 고객 비중 값을 계산할 수 없습니다."
        threshold = max_value - 5.0

        top_segments = store_persona_data[store_persona_data >= threshold]
        
        if len(top_segments) > 2:
            top_segments = top_segments.sort_values(ascending=False).head(2)
            
        if top_segments.empty:
             top_segments = store_persona_data.sort_values(ascending=False).head(2)
             if top_segments.empty:
                 return f"분석 실패: '{store_id}' 가맹점의 유효한 주요 고객층을 찾을 수 없습니다. (모든 고객 비중 데이터가 유효하지 않거나 0에 가까울 수 있습니다.)"

        # 데이터 근거 추출 1: 핵심 고객 세그먼트 컬럼명 및 페르소나 이름
        top_segment_cols = top_segments.index.tolist()
        main_personas_info = [PERSONA_MAP[seg] for seg in top_segment_cols]
        main_personas_str = ", ".join([f"{p['name']}({p['desc']})" for p in main_personas_info])
        # 컬럼명과 페르소나 이름을 매칭하여 문자열 생성
        persona_col_mapping_str = ", ".join([f"{col} -> {PERSONA_MAP[col]['name']}" for col in top_segment_cols])
        
        main_personas_details_list = []
        for p in main_personas_info:
            detail = f"  - **{p['name']} ({p['desc']})**: {p['features']}"
            main_personas_details_list.append(detail)
        main_personas_details_str = "\n".join(main_personas_details_list) # LLM 프롬프트용 상세 설명
        
        # 핵심 성공 전략 분석
        store_data = df_all_join[df_all_join['ENCODED_MCT'] == store_id]
        if store_data.empty:
            return f"분석 실패: '{store_id}' 가맹점의 상세 정보를 찾을 수 없습니다."
        
        latest_data = store_data.sort_values(by='TA_YM', ascending=False).iloc[0]
        store_commercial_area = latest_data['HPSN_MCT_BZN_CD_NM']
        
        if pd.isna(store_commercial_area):
            store_commercial_area = '비상권'
        
        dna_row = df_prompt_dna[(df_prompt_dna['상권'] == store_commercial_area) & (df_prompt_dna['업종'] == '카페')]
        if dna_row.empty:
            return f"분석 실패: '{store_commercial_area}' 상권의 '카페' 업종 성공 DNA를 찾을 수 없습니다."
        
        # 데이터 근거 추출 2: 핵심 성공 변수 컬럼명 및 전략 내용
        key_factor_col = dna_row['핵심성공변수(DNA)'].iloc[0] # 컬럼명 저장
        core_strategy = dna_row['핵심경영전략'].iloc[0] # 전략 내용 저장

        # 2단계: LLM 호출을 위한 프롬프트 생성 (이 부분은 이전과 동일)
        prompt_for_gemini = f"""
# MISSION
너는 **'고객 심리 분석가'이자 '실행 중심 전략 컨설턴트'**인 'AI 비밀상담사'다.
너의 임무는 사장님 가게의 **핵심 고객(WHO) 데이터**와 **검증된 성공 전략(WHAT) 데이터**를 분석하여, 사장님이 **'그래서 당장 뭘 해야 하는지'** 명확히 알 수 있는 **구체적인 마케팅 액션 플랜**을 제시하는 것이다.

---

# DATA INPUT (너가 분석할 핵심 데이터)
1. **[WHO] 우리 가게 핵심 고객 페르소나:**
{main_personas_details_str}

2. **[WHAT] 우리 가게의 시장 성공 전략:**
- '{core_strategy}'

---

# OUTPUT INSTRUCTION (아래 지시사항과 형식을 엄격하게 따라서 결과물을 생성해줘)

## 1. 전체 톤앤매너
- **고객 중심의 조언가:** 모든 설명과 제안의 주어를 '고객'으로 삼아, 사장님이 고객의 입장에서 생각하도록 자연스럽게 유도해줘.
- **논리적 스토리텔러:** '고객은 ~을 원하기 때문에(WHO 분석) → 우리 가게는 ~방식으로 성공할 수 있는데(WHAT 분석) → 따라서 이렇게 공략해야 합니다(핵심 방향 및 액션 플랜)' 라는 명확한 논리적 흐름으로 스토리를 전개해줘.

## 2. 최종 결과물 형식 (이 구조를 완벽하게 따라줘)

**(고객에 대한 깊은 이해를 강조하며, 따뜻한 인사말로 시작)**

---

**1. 우리 가게의 핵심 주요 고객층 정보 (WHO: 고객 페르소나 심층 분석)**
(DATA의 **[WHO]**를 바탕으로, 이 고객들이 '왜' 우리 카페를 찾는지, 그들의 **일상, 가치관, 소비 패턴, 욕망(Needs/Wants)** 등을 깊이 있게 파고들어 사장님이 고객을 **마치 옆에서 보는 것처럼** 생생하게 이해할 수 있도록 상세히 묘사해줘. 단순히 특징 나열이 아니라, 스토리가 느껴지도록 설명해야 한다.)

---

**2. 우리 가게에 대입할 핵심 성공 전략 (WHAT: 핵심 성공 전략 심층 분석)**
(DATA의 **[WHAT]** 전략을 설명하되, 이 전략이 **왜 (1)번 고객들의 마음을 얻는 데 가장 효과적인 '무기'가 될 수 있는지** 그 이유를 명확하고 설득력 있게 설명해줘. 이 전략의 본질과 기대 효과를 사장님이 정확히 이해하도록 돕는 것이 목표다.)

---

**3. 최적화된 결론 도출 (WHO + WHAT 연결 및 핵심 마케팅 방향)**
(이제 (1)번의 고객 분석과 (2)번의 전략 분석을 **논리적으로 연결**해야 한다. **"(1)번 고객의 어떤 구체적인 심리/욕구** 때문에, **(2)번 전략을 활용하는 것이 왜 우리 가게에게 최적의 선택인지"** 명확하게 설명해줘. 이 연결고리를 바탕으로, 우리 가게가 앞으로 나아가야 할 **가장 중요한 마케팅 방향**을 한 문장으로 명확하게 정의한다.)
    * 예시) "SNS 공유를 즐기고 특별한 경험을 중시하는 1020 여성 고객의 특성을 고려할 때, '기존 고객 충성도 강화 및 관계 심화' 전략은 이들에게 '나만 아는 특별한 공간'이라는 인식을 심어주고 자발적인 바이럴을 유도하는 최적의 경로입니다. 따라서 우리 가게의 핵심 마케팅 방향은 **'인스타그램 중심의 참여형 콘텐츠를 통해 단골 고객과의 유대감을 강화하는 것'** 입니다."

---

**4. 사장님 맞춤형 실전 마케팅 액션 플랜**
(이제 앞서 정의한 **(3)번 핵심 마케팅 방향**에 따라, 사장님이 바로 실행할 수 있는 구체적인 액션 플랜을 제시한다.)

**📈 최적 마케팅 채널 추천**
(타겟 고객 **[WHO]**가 가장 활발하게 이용하고, **[WHAT]** 전략을 펼치기 좋은 채널 1~2개를 선정한다.)

* **[채널 이름]:**
    * **선정 이유 (Why):**
        1. **(고객 접점):** 왜 이 채널이 **(1)번 고객**의 정보 소비 패턴 및 라이프스타일에 가장 적합한가? (**[WHO] 데이터** 언급 필수)
        2. **(전략 실행):** 어떻게 이 채널에서 **(2)번 전략**을 가장 효과적으로 구현하여 고객의 마음을 사로잡을 수 있는가? (**[WHAT] 데이터** 언급 필수)

**💡 즉시 실행 가능한 홍보 방안 (1~2가지)**
(사장님이 **'내일부터 당장'** 실행할 수 있는 **구체적이고 창의적인 아이디어**를 제안한다. 모든 아이디어는 (3)번 핵심 방향과 일치해야 한다.)

**[아이디어 명칭]:** (예: '#카페로그 챌린지' 인스타그램 이벤트)
* **무엇을 (What):** (이 아이디어를 통해 **(1)번 고객**에게 어떤 **'구체적인 가치나 즐거움'**을 제공할 것인가?)
* **어떻게 (How):** (사장님이 **'바로 따라 할 수 있도록'** 실행 방법, 예상 비용, 준비물, 기간 등을 **단계별로 매우 상세하게** 설명한다.)
    * 예시) 1단계: 포토존 만들기 (비용: 5만원 이하, 소품: 전신 거울, 감성 조명, 작은 식물), 2단계: 인스타그램 공식 계정에 이벤트 공지 (기간: 4주, 필수 해시태그: #가게이름 #챌린지명 #단골인증), 3단계: 매주 금요일 우수작 1명 선정 및 DM 발표 (상품: 시그니처 디저트 + 음료 1잔 무료 쿠폰), 4단계: 선정작 리그램 및 참여자 감사 메시지 전달
* **기대 효과 (Why this works):**
    1. **(고객 반응):** 이 아이디어가 왜 **(1)번 고객**의 **[WHO] 데이터**에 나타난 심리/욕구(예: SNS 공유 통한 정체성 형성, 가심비 추구)를 정확히 충족시켜 **'참여하고 싶다', '자랑하고 싶다'**는 마음을 유발하는가?
    2. **(전략 달성):** 이 아이디어가 어떻게 **(2)번 전략 ([WHAT] 데이터)** (예: 기존 고객 충성도 강화 및 관계 심화) 달성에 **직접적으로 기여**하며, 최종적으로 재방문율과 매출 증대에 어떤 긍정적 영향을 미칠 것으로 예상되는가?

---

**(사장님께 데이터 기반 마케팅의 중요성을 강조하며, 실행을 독려하는 진심 어린 메시지로 마무리)**
"""
        
        # [*** 여기가 수정된 부분 ***]
        # LLM 직접 호출 추가 및 final_report 구조 변경

        # LLM 호출
        llm_response = call_gemini_llm(prompt_for_gemini)

        # 데이터 근거 요약 섹션 생성
        data_summary_section = f"""
---
## 📊 AI 분석 핵심 데이터 근거

* **핵심 고객층 (고객 데이터 -> 페르소나):** {persona_col_mapping_str}
* **핵심 성공 변수 (데이터):** {key_factor_col}
* **적용될 핵심 성공 전략:** '{core_strategy}' (상권: {store_commercial_area})

---
"""

        # 최종 통합 리포트 생성 (LLM 응답을 포함하도록 수정)
        final_report = f"""
======================================================================
      🤖 AI 비밀상담사 - '{store_id}' 가맹점 맞춤 전략 리포트
======================================================================

{basic_info_content}

{data_summary_section} 

## 💡 AI 컨설턴트 상세 분석 및 전략 제안
{llm_response}
"""
        # 2. 통합된 리포트 하나만 반환
        return final_report

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"""🚨 카페 마케팅 분석 중 오류가 발생했습니다.

**오류 상세 정보:**
- 오류 유형: {type(e).__name__}
- 오류 메시지: {str(e)}
- 가맹점 ID: {store_id}

**해결 방법:**
1. 가맹점 ID가 올바른지 확인해주세요
2. 데이터베이스에 해당 가맹점 정보가 있는지 확인해주세요
3. 문제가 지속되면 관리자에게 문의해주세요

**기술적 세부사항:**
{error_details}"""
# =============================================================================
# 모델 2: 재방문율 30% 이하 가맹점 원인 진단 및 A/B 전략 제안
# =============================================================================

@tool
def revisit_rate_analysis_tool(store_id: str, df_all_join: pd.DataFrame, df_prompt_dna: pd.DataFrame) -> str:
    """
    재방문율이 30% 이하로 낮은 가맹점의 근본적인 원인을 3가지 핵심 동인(가격, 고객층, 채널)으로 
    경쟁 그룹과 비교 분석하고, 재방문율을 높이기 위한 A/B 전략을 제안하는 도구.
    '재방문', '단골' 문제에 특화되어 있다.
    
    Args:
        store_id: 분석할 가맹점 ID
        df_all_join: 전체 JOIN 데이터
        df_prompt_dna: AI상담사 핵심전략 프롬프트 데이터
    
    Returns:
        재방문율 분석 및 개선 전략 리포트
    """
    try:
        # 1. 공통 헬퍼 함수 호출 (반환 변수명 변경)
        basic_info_content, latest_store_data = _get_store_basic_info(store_id, df_all_join)
        
        if latest_store_data is None:
            return basic_info_content # 오류 메시지는 그대로 반환

        # 점수 컬럼 생성 (필요한 경우)
        score_cols = ['MCT_OPE_MS_CN', 'RC_M1_TO_UE_CT', 'RC_M1_SAA', 'RC_M1_AV_NP_AT']
        for col in score_cols:
            if col in df_all_join.columns and f'{col}_SCORE' not in df_all_join.columns:
                df_all_join[f'{col}_SCORE'] = df_all_join[col].apply(get_score_from_raw)
        
        target_store_all_months = df_all_join[df_all_join['ENCODED_MCT'] == store_id]
        target_store = latest_store_data

        # 재방문율 계산 (월별 평균)
        target_revisit_rate_series = target_store_all_months['MCT_UE_CLN_REU_RAT'].dropna()
        target_revisit_rate = 0.0 if target_revisit_rate_series.empty else target_revisit_rate_series.mean()

        if target_revisit_rate >= 30:
            latest_month = target_store.get('TA_YM', '최신')
            return f"📣 분석 결과: 월 평균 재방문율이 {target_revisit_rate:.1f}%로 양호합니다. (최신월: {latest_month})"

        industry, commercial_area = target_store['업종_정규화2_대분류'], target_store['HPSN_MCT_BZN_CD_NM']
        area_name = commercial_area if pd.notna(commercial_area) else "비상권"
        
        # 경쟁 그룹 설정
        peer_group_filter = (df_all_join['업종_정규화2_대분류'] == industry) & (df_all_join['ENCODED_MCT'] != store_id)
        if pd.isna(commercial_area):
            peer_group = df_all_join[peer_group_filter & (df_all_join['HPSN_MCT_BZN_CD_NM'].isna())]
        else:
            peer_group = df_all_join[peer_group_filter & (df_all_join['HPSN_MCT_BZN_CD_NM'] == commercial_area)]

        if len(peer_group) < 3:
            return f"📣 분석 보류: 비교 분석을 위한 경쟁 그룹이 부족합니다."
            
        revisit_threshold = peer_group['MCT_UE_CLN_REU_RAT'].quantile(0.5)
        successful_peers = peer_group[peer_group['MCT_UE_CLN_REU_RAT'] >= revisit_threshold]
        if successful_peers.empty:
            return f"📣 분석 보류: 성공 그룹을 찾을 수 없습니다."

        # 표준화된 평균 계산 함수
        def get_series_mean(series):
            series_cleaned = series.dropna()
            return 0.0 if series_cleaned.empty else series_cleaned.mean()

        def get_group_mean(df, group_col, value_col):
            if df.empty or value_col not in df.columns:
                return 0.0
            group_means = df.groupby(group_col)[value_col].mean()
            final_mean = group_means.mean()
            return 0.0 if pd.isna(final_mean) else final_mean

        # 내 가게의 전체 기간 평균 계산
        target_price_score_avg = get_series_mean(target_store_all_months['RC_M1_AV_NP_AT_SCORE'])
        target_resident_ratio_avg = get_series_mean(target_store_all_months['RC_M1_SHC_RSD_UE_CLN_RAT'])
        target_delivery_ratio_avg = get_series_mean(target_store_all_months['DLV_SAA_RAT'])

        # 성공 그룹의 가게별 평균 계산 후 -> 전체 평균
        peer_price_score_avg = get_group_mean(successful_peers, 'ENCODED_MCT', 'RC_M1_AV_NP_AT_SCORE')
        peer_resident_ratio_avg = get_group_mean(successful_peers, 'ENCODED_MCT', 'RC_M1_SHC_RSD_UE_CLN_RAT')
        peer_delivery_avg = get_group_mean(successful_peers, 'ENCODED_MCT', 'DLV_SAA_RAT')
        
        # 배달 미운영 가게 처리 로직
        is_delivery_not_operated = (target_delivery_ratio_avg == 0.0)
        
        delivery_target_for_analysis = target_delivery_ratio_avg
        delivery_peer_for_analysis = peer_delivery_avg

        if is_delivery_not_operated:
            delivery_peer_for_analysis = 0.0
            delivery_target_for_analysis = 0.0

        analysis_results = {
            'price_competitiveness': {'target': target_price_score_avg, 'peer_avg': peer_price_score_avg},
            'audience_alignment': {'target': target_resident_ratio_avg, 'peer_avg': peer_resident_ratio_avg},
            'channel_expansion': {'target': delivery_target_for_analysis, 'peer_avg': delivery_peer_for_analysis}
        }
        
        for key in analysis_results:
            analysis_results[key]['gap'] = analysis_results[key].get('target', np.nan) - analysis_results[key].get('peer_avg', np.nan)
        
        # 페르소나 진단 (사전질문 2번 전용)
        gaps = {k: v['gap'] for k, v in analysis_results.items() if 'gap' in v and pd.notna(v['gap'])}
        if not gaps:
            persona = "분석 불가"
        else:
            op_tenure_score = target_store.get('MCT_OPE_MS_CN_SCORE', 6)
            new_ratio = target_store.get('MCT_UE_CLN_NEW_RAT', 0)
            
            if op_tenure_score <= 2 and new_ratio > 60:
                persona = "첫인상만 좋은 신규 매장"
            elif gaps.get('price_competitiveness', 0) < -1:
                persona = "재방문하기엔 부담스러운 가격"
            elif gaps.get('audience_alignment', 0) < -15:
                persona = "동네 주민을 사로잡지 못하는 매장"
            elif gaps.get('channel_expansion', 0) < -20:
                persona = "배달 채널 부재"
            else:
                persona = "총체적 마케팅 부재"

        # 전략 데이터 조회
        strategy_row = df_prompt_dna[(df_prompt_dna['업종'] == industry) & (df_prompt_dna['상권'] == area_name)]
        if not strategy_row.empty:
            key_factor = strategy_row.iloc[0]['핵심성공변수(DNA)']
            key_strategy = strategy_row.iloc[0]['핵심경영전략']
        else:
            key_factor = "데이터 없음"
            key_strategy = "일반적인 개선 전략 필요"

        # STAGE 3: 페르소나와 핵심성공전략 상세 분석
        def get_stage3_analysis(persona, key_factor, key_strategy, analysis_results):
            # 페르소나 상세 설명
            persona_explanation = ""
            if persona == "첫인상만 좋은 신규 매장":
                persona_explanation = f"""
**📊 페르소나 진단: '{persona}'**

[데이터 근거]
- 운영 기간: {translate_metric('tenure', target_store.get('MCT_OPE_MS_CN'))} (신규 매장)
- 신규 고객 비율: {target_store.get('MCT_UE_CLN_NEW_RAT', 0):.1f}% (60% 초과)
- 현재 재방문율: {target_revisit_rate:.2f}% (30% 미만)

[페르소나 특성 분석]
이 유형은 '첫 방문은 많지만 재방문이 적은' 전형적인 신규 매장 패턴입니다. 
고객들이 첫 방문에서는 만족하지만, 재방문할 만한 충분한 동기나 명분을 제공하지 못하고 있습니다.
신규 고객이 충성 고객으로 전환되는 '온보딩(Onboarding)' 과정에서 실패하고 있는 상황입니다."""
            elif persona == "재방문하기엔 부담스러운 가격":
                persona_explanation = f"""
**📊 페르소나 진단: '{persona}'**

[데이터 근거]
- 객단가 수준: {translate_metric('level', target_store.get('RC_M1_AV_NP_AT'))} (최신월 기준)
- 성공 그룹 대비 격차: {analysis_results['price_competitiveness']['gap']:.2f}점 (객단가 점수가 낮음)
- 현재 재방문율: {target_revisit_rate:.2f}% (30% 미만)

[페르소나 특성 분석]
이 유형은 가격 경쟁력에서 성공 그룹 대비 명확한 격차를 보이는 매장입니다.
고객들이 첫 방문 후 '가격이 부담스럽다'는 인식을 가지고 재방문을 꺼리는 상황입니다.
단순히 가격을 낮추는 것이 아니라, 가격 대비 가치를 높이는 전략이 필요합니다."""
            elif persona == "동네 주민을 사로잡지 못하는 매장":
                persona_explanation = f"""
**📊 페르소나 진단: '{persona}'**

[데이터 근거]
- 거주 고객 비율: {analysis_results['audience_alignment'].get('target', 0):.1f}% (월 평균)
- 성공 그룹 대비 격차: {analysis_results['audience_alignment']['gap']:.1f}%p (거주 고객 부족)
- 현재 재방문율: {target_revisit_rate:.2f}% (30% 미만)

[페르소나 특성 분석]
이 유형은 동네 주민 고객 확보에 실패한 매장입니다.
성공 그룹 대비 거주 고객 비율이 현저히 낮아, 단골 고객 기반이 약한 상황입니다.
지역 커뮤니티와의 관계 구축이 시급한 과제입니다."""
            elif persona == "배달 채널 부재":
                persona_explanation = f"""
**📊 페르소나 진단: '{persona}'**

[데이터 근거]
- 배달 매출 비중: {target_delivery_ratio_avg:.1f}% (월 평균, 배달 미운영)
- 성공 그룹 대비 격차: {analysis_results['channel_expansion']['gap']:.1f}%p (배달 채널 부재)
- 현재 재방문율: {target_revisit_rate:.2f}% (30% 미만)

[페르소나 특성 분석]
이 유형은 배달 채널을 전혀 활용하지 않는 매장입니다.
현대 고객의 다양한 소비 패턴(매장 방문 + 배달 주문)을 충족시키지 못하고 있습니다.
온라인 고객 접점 확보가 시급한 과제입니다."""
            else:  # "총체적 마케팅 부재"
                persona_explanation = f"""
**📊 페르소나 진단: '{persona}'**

[데이터 근거]
- 가격 경쟁력 격차: {analysis_results['price_competitiveness']['gap']:.2f}점
- 거주 고객 격차: {analysis_results['audience_alignment']['gap']:.1f}%p  
- 배달 채널 격차: {analysis_results['channel_expansion']['gap']:.1f}%p
- 현재 재방문율: {target_revisit_rate:.2f}% (30% 미만)

[페르소나 특성 분석]
이 유형은 3가지 핵심 동인 모두에서 성공 그룹 대비 부족한 매장입니다.
개별 지표의 문제라기보다는 전체적인 마케팅 전략과 고객 관리 시스템이 부재한 상황입니다.
체계적인 마케팅 인프라 구축이 시급한 과제입니다."""

            # 핵심성공전략 상세 설명
            strategy_explanation = f"""
**🎯 핵심 성공 전략: '{key_factor}'**

[데이터 근거]
- 업종/상권: {industry} / {area_name}
- 성공 그룹 분석 결과 도출된 핵심 성공 변수

[전략 상세 분석]
'{key_strategy}'

이 전략은 동일 업종, 동일 상권 내에서 재방문율이 높은 성공 그룹의 공통된 특징을 분석한 결과입니다.
데이터 기반으로 검증된 성공 방정식이므로, 이를 기반으로 한 마케팅 전략 수립이 효과적입니다."""

            return persona_explanation + strategy_explanation

        # STAGE 4: 페르소나 + 핵심성공전략 융합 마케팅 전략
        # [*** 여기가 수정된 부분 ***]
        def get_stage4_integrated_strategy(persona, key_factor, key_strategy, analysis_results):
            # 페르소나별 기본 전략 템플릿 (이 내용은 그대로 사용)
            persona_strategies = {
                "첫인상만 좋은 신규 매장": f"""
[데이터 기반 전략 근거]
- 신규 고객 비율: {target_store.get('MCT_UE_CLN_NEW_RAT', 0):.1f}% (60% 초과), 운영 기간: {translate_metric('tenure', target_store.get('MCT_OPE_MS_CN'))} -> 첫 방문은 많지만 재방문 전환율 낮음
[구체적 실행 방안]
1. '첫 만남 각인': 첫 방문 100% 대상 "다음 방문 시 대표메뉴 1개 무료" 쿠폰 증정 (유효기간 7일)
2. '단골 네비게이션': 2-3회차 방문 고객 대상 "사장님 추천 숨겨진 메뉴 조합" 안내, VIP 전용 메뉴 제공
3. '재방문 동기 부여': 월간 "단골 전환 이벤트", 연속 방문 누적 혜택 (3회→5회→10회 차등)
""",
                "재방문하기엔 부담스러운 가격": f"""
[데이터 기반 전략 근거]
- 객단가 수준: {translate_metric('level', target_store.get('RC_M1_AV_NP_AT'))}, 성공 그룹 대비 격차: {analysis_results['price_competitiveness']['gap']:.2f}점 -> 가격 대비 가치 인식 부족
[구체적 실행 방안]
1. '디코이 효과' 메뉴 재구성: 대표메뉴 옆 "실속 메뉴"(양 80%, 가격 60%) 배치 (메뉴판: 고가→실속→일반 순)
2. '가치부가형 번들링': 메인 주문 시 마진 높은 음료/사이드 무료 증정, "오늘의 특별 조합" 20% 할인 ("총 가치 OO원 → OO원" 표시)
3. '가격 투명성' 마케팅: 비용 구조 공개, "왜 이 가격인가?" 스토리텔링, 리뷰에 가성비 만족도 유도
""",
                "동네 주민을 사로잡지 못하는 매장": f"""
[데이터 기반 전략 근거]
- 거주 고객 비율: {analysis_results['audience_alignment'].get('target', 0):.1f}%, 성공 그룹 대비 격차: {analysis_results['audience_alignment']['gap']:.1f}%p -> 동네 주민 확보 실패, 단골 기반 약함
[구체적 실행 방안]
1. '우리 동네 멤버십': 주민 인증 시 혜택(포인트 2배, 월간 스페셜 메뉴), 주민 전용 이벤트("동네 맛집 투어")
2. '로컬 인플루언서' 파트너십: 지역 맘카페 운영진/맛집 블로거 초청 시식회, 지역 SNS 그룹 소통
3. '지역 사회 기여': 동네 행사 참여, 지역 단체 후원, 지역 취약계층 할인
""",
                "배달 채널 부재": f"""
[데이터 기반 전략 근거]
- 배달 매출 비중: {target_delivery_ratio_avg:.1f}%, 성공 그룹 대비 격차: {analysis_results['channel_expansion']['gap']:.1f}%p -> 온라인 고객 접점 부재
[구체적 실행 방안]
1. '배달앱 최적화': 주요 배달앱 3개 이상 등록, 배달 전용 메뉴/사이즈 구성, 리뷰 4.5점 이상 관리
2. 'O2O 연계 프로모션': 배달 주문 시 매장 쿠폰 동봉, 매장 방문 시 배달 쿠폰 제공
3. '배달 데이터 활용': 인기 배달 메뉴 오프라인 타임 세일, 배달 패턴 분석 후 피크타임 예측/재고 관리
""",
                "총체적 마케팅 부재": f"""
[데이터 기반 전략 근거]
- 가격 경쟁력({analysis_results['price_competitiveness']['gap']:.2f}점), 거주 고객({analysis_results['audience_alignment']['gap']:.1f}%p), 배달 채널({analysis_results['channel_expansion']['gap']:.1f}%p) 모두 부족 -> 마케팅/고객관리 시스템 부재
[구체적 실행 방안]
1. '고객 자산화': 카카오톡 채널 개설("친구 추가 시 1+1 쿠폰"), 고객 DB 구축(방문 빈도, 선호 메뉴 등) 및 세분화(VIP/일반/신규)
2. '자동화 CRM': 가입 1달 후(감사 쿠폰), 생일 당일(생일 쿠폰), 2주 미방문(리마인드 쿠폰), 계절별(시즌 쿠폰/알림) 자동 발송
3. '통합 마케팅 플랫폼': 온/오프라인 통합 포인트, 고객 여정(인지→관심→방문→재방문→추천) 추적, 데이터 기반 개인화 마케팅
"""
            }
            
            selected_persona_strategy_details = persona_strategies.get(persona, "맞춤형 전략 수립이 필요합니다.")
            
            # --- Stage 4 프롬프트 수정 ---
            stage4_prompt = f"""
너는 사장님의 실행을 돕는 **'AI 마케팅 액션 플래너'**다.
앞서 분석된 Stage 1~3의 데이터를 바탕으로, 사장님이 **'그래서 당장 뭘 해야 하는지'** 명확히 알 수 있도록 구체적인 액션 플랜을 제시해야 한다.
추상적인 조언 대신, **데이터에 근거한 실행 가능한 아이템**을 명확하게 제시하라.

**[Stage 1~3 분석 결과 요약]**
* **진단된 문제 유형 (페르소나):** '{persona}'
* **가장 시급한 개선 지표:** 가격 경쟁력({analysis_results['price_competitiveness']['gap']:.2f}점), 거주 고객({analysis_results['audience_alignment']['gap']:.1f}%p), 배달 채널({analysis_results['channel_expansion']['gap']:.1f}%p) 중 가장 격차가 큰 항목
* **검증된 성공 공식 (핵심 전략):** '{key_strategy}' (핵심 성공 변수: '{key_factor}')
* **현재 재방문율:** {target_revisit_rate:.2f}% (목표: 30% 이상)

**[OUTPUT INSTRUCTION]**
아래 형식에 맞춰 **사장님 맞춤형 재방문율 개선 액션 플랜**을 구체적으로 작성하라.

---
**🎯 사장님 맞춤형 액션 플랜: '{persona}' 문제 해결 전략**

(사장님 가게가 '{persona}' 유형으로 진단되었으므로, 이 문제를 해결하기 위한 **가장 효과적인 1~2가지 핵심 액션 아이템**을 아래와 같이 제안합니다. 이는 '{key_strategy}'라는 검증된 성공 공식을 적용한 것입니다.)

{selected_persona_strategy_details}

---
**💡 추가 액션 아이템: 성공 그룹 따라잡기**

(위 핵심 액션 외에도, Stage 2 분석 결과 성공 그룹 대비 격차가 있었던 지표를 개선하기 위한 **추가적인 액션 아이템 1~2가지**를 제안합니다.)

* **[개선 필요 지표 1]** (예: 거주 고객 확보)
    * **액션 아이템:** (예: '우리 동네 멤버십' 도입 - 주민 인증 시 포인트 2배 적립)
    * **데이터 근거:** (예: 현재 거주 고객 비율이 성공 그룹보다 {analysis_results['audience_alignment']['gap']:.1f}%p 낮으므로, 지역 주민 대상 혜택 강화 필요)

* **[개선 필요 지표 2]** (예: 배달 채널 활성화)
    * **액션 아이템:** (예: 주요 배달앱 2곳 이상 입점 및 배달 전용 '1인 세트 메뉴' 출시)
    * **데이터 근거:** (예: 현재 배달 미운영/저조 ({analysis_results['channel_expansion']['gap']:.1f}%p 격차)하므로, 온라인 접점 확대 필요)

---
**📅 4주 실행 로드맵 & 예상 효과**

(사장님이 바로 실행하실 수 있도록, 위 액션 아이템들을 **4주간의 로드맵**으로 정리했습니다.)

* **1주차:** [가장 먼저 해야 할 핵심 액션 1가지 구체적으로 명시] (예: '실속 메뉴' 개발 및 메뉴판 디자인 변경)
* **2주차:** [그 다음 실행할 액션 1가지 구체적으로 명시] (예: 카카오톡 채널 개설 및 '친구 추가' 이벤트 시작)
* **3-4주차:** [나머지 액션 실행 및 1-2주차 성과 측정 시작] (예: 배달앱 입점 신청 / 쿠폰 사용률, 채널 친구 수 등 데이터 확인)

* **예상 효과:** (위 로드맵을 꾸준히 실행하시면, 2-3개월 내 재방문율 {target_revisit_rate:.2f}% → **30% 달성** 및 **월 매출 20-30% 증가**를 기대할 수 있습니다. LTV 향상으로 장기적인 안정 매출 확보에도 기여할 것입니다.)
"""
            # 포맷팅된 프롬프트 문자열을 반환
            return stage4_prompt
            
        # --- 최종 리포트 생성 로직 변경 ---
        # 1. Stage 3 분석 결과 생성
        stage3_analysis = get_stage3_analysis(persona, key_factor, key_strategy, analysis_results)
        
        # 2. Stage 4 프롬프트 생성 (LLM 호출은 아직 안 함)
        stage4_prompt_for_llm = get_stage4_integrated_strategy(persona, key_factor, key_strategy, analysis_results)

        # 3. Stage 4 프롬프트를 LLM에 보내 최종 액션 플랜 생성
        stage4_llm_response = call_gemini_llm(stage4_prompt_for_llm)

        # 4. 최종 리포트 조합
        final_report = f"""
 AI 하이브리드 전략 컨설팅 - '{store_id}' 가맹점 분석 리포트
{basic_info_content}

---------- [STAGE 1] 우리 가게 현황 브리핑 ----------
  - 업종 / 상권: {industry} / {area_name}
  - 운영 기간: {translate_metric('tenure', target_store.get('MCT_OPE_MS_CN'))}
  - 현재 재방문율 (월 평균): {target_revisit_rate:.2f}% (⚠️ 개선 필요)
  - 매출액 수준 (최신월 기준): {translate_metric('level', target_store.get('RC_M1_SAA'))}
  - 객단가 수준 (최신월 기준): {translate_metric('level', target_store.get('RC_M1_AV_NP_AT'))}

---------- [STAGE 2] 재방문율 핵심 동인(Key Drivers) 3차원 분석 ----------
같은 '{industry}' 업종 '{area_name}' 상권 내 성공 그룹과 3가지 핵심 동인을 비교한 결과입니다.

  ① 가격 경쟁력 (객단가)
    - 내 가게 수준 (최신월): {translate_metric('level', target_store.get('RC_M1_AV_NP_AT'))} (월별 평균점수: {analysis_results['price_competitiveness']['target']:.2f})
    - 성공 그룹 평균: {score_to_level_text(analysis_results['price_competitiveness']['peer_avg'])} (월별 평균점수: {analysis_results['price_competitiveness']['peer_avg']:.2f})
    - 격차: {analysis_results['price_competitiveness']['gap']:.2f}점
    - ➡️ AI 분석: {'성공 그룹보다 객단가가 높아(점수가 낮아), 고객이 재방문하기에 부담을 느낄 수 있습니다.' if analysis_results['price_competitiveness'].get('gap', 0) < 0 else '객단가 수준은 경쟁 그룹 대비 양호합니다. 가격보다는 다른 요소를 먼저 개선해야 합니다.'}

  ② 핵심 고객층 (거주자)
    - 내 가게 비율 (월 평균): {analysis_results['audience_alignment'].get('target', 0):.1f}%
    - 성공 그룹 평균: {analysis_results['audience_alignment'].get('peer_avg', 0):.1f}%
    - 격차: {analysis_results['audience_alignment']['gap']:.1f}%p
    - ➡️ AI 분석: {f"동네 주민 고객을 '{abs(analysis_results['audience_alignment']['gap']):.1f}%p' 만큼 더 확보할 수 있는 기회가 있습니다." if analysis_results['audience_alignment'].get('gap', 0) < 0 else '동네 주민 고객 확보는 양호한 수준입니다.'}
        
  ③ 채널 확장성 (배달)
    - 내 가게 배달매출 비중 (월 평균): {target_delivery_ratio_avg:.1f}% {'(배달 미운영)' if is_delivery_not_operated else ''}
    - 성공 그룹 평균: {analysis_results['channel_expansion'].get('peer_avg', 0):.1f}%
    - 격차: {analysis_results['channel_expansion']['gap']:.1f}%p
    - ➡️ AI 분석: {'배달을 운영하지 않는 것으로 확인됩니다.' if is_delivery_not_operated else (f"성공 그룹 대비 배달 매출 비중이 '{abs(analysis_results['channel_expansion']['gap']):.1f}%p' 낮아, 온라인 잠재 고객을 놓치고 있습니다." if analysis_results['channel_expansion'].get('gap', 0) < 0 else '배달 채널은 경쟁력 있게 운영되고 있습니다.')}

📊 AI 종합 진단: 위 3가지 동인을 종합 분석한 결과, 사장님 가게의 현재 가장 시급한 개선 과제는 '{persona}' 유형에 해당합니다.

---------- [STAGE 3] 페르소나 & 핵심성공전략 상세 분석 ----------
{stage3_analysis}

---------- [STAGE 4] AI 마케팅 액션 플랜 ----------
{stage4_llm_response}

======================================================================
"""
        # 2. 통합된 리포트 하나만 반환
        return final_report

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"""🚨 재방문율 분석 중 오류가 발생했습니다.

**오류 상세 정보:**
- 오류 유형: {type(e).__name__}
- 오류 메시지: {str(e)}
- 가맹점 ID: {store_id}

**해결 방법:**
1. 가맹점 ID가 올바른지 확인해주세요
2. 데이터베이스에 해당 가맹점 정보가 있는지 확인해주세요
3. 문제가 지속되면 관리자에게 문의해주세요

**기술적 세부사항:**
{error_details}"""

# =============================================================================
# 모델 3: 요식업종 가맹점의 전방위 강점/약점 분석 및 솔루션 제안
# =============================================================================

def apply_emphasis(score):
    """점수를 0-100 범위에서 양 극단으로 스트레칭하여 차이를 명확하게 만듭니다."""
    x = (score - 50) / 50
    y = np.sign(x) * (abs(x) ** 0.7)
    return max(0, min(100, (y * 50) + 50))

def get_percentile_score(store_value, benchmark_series, higher_is_better=True):
    """경쟁 그룹 내 백분위 순위를 0-100점 척도의 '경영 점수'로 변환하고 가중치를 적용"""
    if pd.isna(store_value) or benchmark_series.empty:
        return 50
    combined = pd.concat([benchmark_series, pd.Series([store_value])])
    percentile = combined.rank(pct=True, na_option='bottom').iloc[-1]
    raw_score = percentile * 100 if higher_is_better else (1 - percentile) * 100
    return apply_emphasis(raw_score)

def parse_segment(segment_name):
    """세그먼트 이름을 파싱하여 성별과 연령 정보 추출"""
    parts = segment_name.replace('M12_', '').replace('_RAT', '').split('_')
    return {'name': segment_name, 'gender': parts[0], 'age': parts[1]}

def get_age_tier(age_str):
    """연령대를 숫자로 변환"""
    tiers = {'1020': 1, '30': 2, '40': 3, '50': 4, '60': 5}
    return tiers.get(age_str, 0)

def calculate_advanced_match_score(area_top2_names, store_top2_names):
    """'Top 2 비교 및 유사도 보너스' 로직으로 적합도 점수 계산"""
    intersection_count = len(set(area_top2_names) & set(store_top2_names))
    
    base_score = 0
    if intersection_count == 2:
        base_score = 85
    elif intersection_count == 1:
        base_score = 45
    else:
        base_score = 5

    bonus_score = 0
    non_matching_area = [parse_segment(s) for s in set(area_top2_names) - set(store_top2_names)]
    non_matching_store = [parse_segment(s) for s in set(store_top2_names) - set(store_top2_names)]

    for s_seg in non_matching_store:
        for a_seg in non_matching_area:
            if s_seg['age'] == a_seg['age']:
                bonus_score = max(bonus_score, 10)
            if s_seg['gender'] == a_seg['gender'] and abs(get_age_tier(s_seg['age']) - get_age_tier(a_seg['age'])) == 1:
                bonus_score = max(bonus_score, 5)
    
    return max(0, min(100, base_score + bonus_score))

@tool
def store_strength_weakness_tool(store_id: str, df_all_join: pd.DataFrame) -> str:
    """
    요식업종 가맹점의 전반적인 건강 상태를 다각도로 분석하여, 
    경쟁 그룹 대비 명확한 강점과 약점을 진단하고, 
    이를 바탕으로 개선 솔루션을 제안하는 도구.
    가게의 '문제점', '가장 큰 문제점' 에 대한 포괄적인 진단이 필요할 때 사용된다.
    
    Args:
        store_id: 분석할 가맹점 ID
        df_all_join: 전체 JOIN 데이터
    
    Returns:
        강점/약점 분석 및 개선 솔루션 리포트
    """
    try:
        # 1. 공통 헬퍼 함수 호출 (반환 변수명 변경)
        basic_info_content, latest_store_data = _get_store_basic_info(store_id, df_all_join)
        
        if latest_store_data is None:
            return basic_info_content # 오류 메시지는 그대로 반환

        store_df = df_all_join[df_all_join['ENCODED_MCT'] == store_id].tail(12)
        category, commercial_area = latest_store_data['업종_정규화2_대분류'], latest_store_data['HPSN_MCT_BZN_CD_NM']
        
        if pd.notna(commercial_area):
            benchmark_type = "동일 상권 내 동종업계"
            benchmark_df = df_all_join[(df_all_join['업종_정규화2_대분류'] == category) & (df_all_join['HPSN_MCT_BZN_CD_NM'] == commercial_area)]
        else:
            benchmark_type = "비상권 지역의 동종업계"
            benchmark_df = df_all_join[(df_all_join['업종_정규화2_대분류'] == category) & (df_all_join['HPSN_MCT_BZN_CD_NM'].isna())]
        
        # 분석할 지표들 (실제 존재하는 컬럼명으로 수정)
        metrics_to_analyze = [
            {'name': '매출 규모', 'col': 'RC_M1_SAA', 'type': 'tier', 'higher_is_better': False},
            {'name': '유니크 고객 수', 'col': 'RC_M1_UE_CUS_CN', 'type': 'tier', 'higher_is_better': False},
            {'name': '고객당 지출액(객단가)', 'col': 'RC_M1_AV_NP_AT', 'type': 'tier', 'higher_is_better': False},
            {'name': '업종 평균 대비 매출', 'col': 'M1_SME_RY_SAA_RAT', 'type': 'ratio', 'higher_is_better': True},
            {'name': '신규 고객 비율', 'col': 'MCT_UE_CLN_NEW_RAT', 'type': 'ratio', 'higher_is_better': True},
            {'name': '재방문 고객 비율', 'col': 'MCT_UE_CLN_REU_RAT', 'type': 'ratio', 'higher_is_better': True},
            {'name': '배달 매출 비율', 'col': 'DLV_SAA_RAT', 'type': 'ratio', 'higher_is_better': True},
            {'name': '거주 고객 비율', 'col': 'RC_M1_SHC_RSD_UE_CLN_RAT', 'type': 'ratio', 'higher_is_better': True},
            {'name': '직장인 고객 비율', 'col': 'RC_M1_SHC_WP_UE_CLN_RAT', 'type': 'ratio', 'higher_is_better': True},
            {'name': '유동인구 고객 비율', 'col': 'RC_M1_SHC_FLP_UE_CLN_RAT', 'type': 'ratio', 'higher_is_better': True},
        ]

        # 배달 데이터 확인
        has_delivery_data = store_df['DLV_SAA_RAT'].sum() > 0
        if not has_delivery_data:
            metrics_to_analyze = [m for m in metrics_to_analyze if m['name'] != '배달 매출 비율']

        # 벤치마크 기준 데이터 변경: 루프 시작 전에, 벤치마크 그룹의 '최신' 데이터만 모은 데이터프레임을 생성
        benchmark_latest_df = benchmark_df.loc[benchmark_df.groupby('ENCODED_MCT')['TA_YM'].idxmax()]

        all_scores = []
        for metric in metrics_to_analyze:
            if metric['type'] == 'tier':
                # tier (구간) 타입 지표 처리
                store_val = get_score_from_raw(latest_store_data[metric['col']])
                benchmark_series = benchmark_latest_df[metric['col']].apply(get_score_from_raw)
                score = get_percentile_score(store_val, benchmark_series.dropna(), metric['higher_is_better'])
                
                # [수정] translate_metric을 사용하여 LLM이 이해할 수 있는 텍스트로 변환
                store_display = translate_metric('level', latest_store_data[metric['col']])
                benchmark_display_mode = "N/A"
                if not benchmark_latest_df.empty and not benchmark_latest_df[metric['col']].mode().empty:
                    benchmark_display_mode = benchmark_latest_df[metric['col']].mode()[0]
                benchmark_display = translate_metric('level', benchmark_display_mode)
                    
            elif metric['type'] == 'ratio':
                # ratio (비율) 타입 지표 처리
                # [수정] 점수 계산(store_val)과 표시(store_display_val)의 기준을 '최신 월' 데이터로 통일
                store_val = latest_store_data[metric['col']]
                # [수정] 벤치마크도 '최신 월' 데이터(benchmark_latest_df)를 기준으로 변경 (경쟁사들의 최신 월 값 리스트)
                benchmark_series = benchmark_latest_df[metric['col']] 
                
                score = get_percentile_score(store_val, benchmark_series.dropna(), metric['higher_is_better'])
                
                store_display_val = store_val # 이미 최신 값이므로 그대로 사용
                benchmark_display_val = benchmark_series.dropna().mean() # 경쟁사 최신 값들의 평균
                
                store_display = f"{store_display_val:.1f}%" if pd.notna(store_display_val) else "N/A"
                benchmark_display = f"{benchmark_display_val:.1f}%" if pd.notna(benchmark_display_val) else "N/A"
            
            # NaN 값 처리
            if pd.isna(score):
                score = 50.0 # 중간값으로 처리

            all_scores.append({
                'metric': metric['name'], 
                'score': f"{score:.1f}점",
                'store_value_display': store_display,
                'benchmark_value_display': benchmark_display,
                'raw_score': score,
                'higher_is_better': metric.get('higher_is_better', True) # 나중 LLM 프롬프트에 활용
            })

        # 고객 세그먼트 분석
        demo_cols = [col for col in df_all_join.columns if ('MAL' in col or 'FME' in col) and 'RAT' in col]
        area_detailed_profile = benchmark_df[demo_cols].mean()
        store_detailed_profile = store_df[demo_cols].mean()
        area_top2, store_top2 = area_detailed_profile.nlargest(2), store_detailed_profile.nlargest(2)

        match_score = calculate_advanced_match_score(area_top2.index.tolist(), store_top2.index.tolist())
        all_scores.append({
            'metric': '상권-고객 적합도', 
            'score': f"{match_score:.1f}점",
            'store_value_display': f"Top 2: {[n.replace('M12_','').replace('_RAT','') for n in store_top2.index.tolist()]}",
            'benchmark_value_display': f"Top 2: {[n.replace('M12_','').replace('_RAT','') for n in area_top2.index.tolist()]}",
            'raw_score': match_score,
            'higher_is_better': True
        })
        
        # 강점과 약점 분류
        strengths = [r for r in all_scores if float(r['score'].replace('점','')) > 80]
        weaknesses = [r for r in all_scores if (r['metric'] != '상권-고객 적합도' and float(r['score'].replace('점','')) < 20) or \
                                           (r['metric'] == '상권-고객 적합도' and float(r['score'].replace('점','')) < 40)]
        
        if not weaknesses and all_scores:
            weakest_link = min(all_scores, key=lambda x: float(x['score'].replace('점','')))
            weaknesses.append(weakest_link)
            
        # 정렬
        sorted_strengths = sorted(strengths, key=lambda x: float(x['score'].split('점')[0]), reverse=True)
        sorted_weaknesses = sorted(weaknesses, key=lambda x: float(x['score'].split('점')[0]))

        # 1. LLM에 전달할 데이터 정리
        # 리스트를 문자열로 변환
        def format_metric_list(metric_list):
            return "\n".join([
                f"- {item['metric']} (경영 점수: {item['score']}): 우리 가게({item['store_value_display']}) vs 경쟁점({item['benchmark_value_display']})" 
                for item in metric_list
            ])
            
        strengths_str = format_metric_list(sorted_strengths)
        weaknesses_str = format_metric_list(sorted_weaknesses)
        
        # 2. LLM 호출을 위한 프롬프트 구성
        # (이 프롬프트는 Gemini 2.5 Flash를 위한 것입니다)
        llm_prompt = f"""
당신은 2025 신한카드 빅콘테스트의 'AI 비밀상담사'입니다. [cite: 2]
당신의 임무는 영세/중소 요식 가맹점 점주에게 [cite: 12, 14] 데이터를 기반으로 [cite: 53] 실질적인 마케팅 전략을 제안하는 것입니다.

지금부터 다음 데이터를 기반으로 '경영 분석 리포트'를 작성해주세요.
점주가 바로 이해하고 실행할 수 있도록 [cite: 16] 구체적이고 설득력 있게 작성해야 합니다.

[가맹점 기본 정보]
{basic_info_content}
분석 기준: {benchmark_type} (최근 12개월 데이터)
업종/상권: {category} / {commercial_area if pd.notna(commercial_area) else '비상권'}

[분석 결과: 강점]
{strengths_str if sorted_strengths else "특별한 강점이 발견되지 않았습니다."}

[분석 결과: 약점]
{weaknesses_str if sorted_weaknesses else "특별한 약점이 발견되지 않았습니다."}

---
[리포트 작성 지시사항]

1.  **📊 AI 전방위 진단 - '{store_id}' 가맹점 리포트**
    * (제공된 [가맹점 기본 정보]를 여기에 먼저 포함하세요.)

2.  **📈 분석 개요**
    * (제공된 '분석 기준', '업종/상권' 정보를 여기에 포함하세요.)

3.  **✅ 강점 요약 (Top 3)**
    * (제공된 [분석 결과: 강점] 목록을 여기에 포함하세요.)

4.  **❌ 약점 요약 (Top 3)**
    * (제공된 [분석 결과: 약점] 목록을 여기에 포함하세요.)

5.  **📣 핵심 문제 진단**
    * [중요!] '약점 요약'에 나열된 **모든 약점을 종합적으로 분석**하여 이 가맹점이 겪는 **가장 근본적인 '핵심 문제'**를 한 문단으로 명확하게 진단해주세요.
    * (예: "단순히 재방문율이 낮은 것이 문제가 아니라, [약점 1]과 [약점 2]가 결합되어 [구체적인 문제 상황]이 발생하고 있습니다.")

6.  **💡 강점 기반 맞춤형 개선 솔루션**
    * [매우 중요!] '핵심 문제 진단'에서 도출된 문제를 해결하기 위해, **[분석 결과: 강점]을 적극적으로 활용**하는 **구체적이고 실행 가능한 마케팅 전략**을 제안해주세요.
    * '강점을 활용하여 약점을 보완'하는 전략이어야 합니다. [cite: 57]
    * 점주가 **'무엇을, 왜, 어떻게'** 해야 하는지 명확히 알 수 있도록 구체적인 실행 방안을 제시해야 합니다. [cite: 16]
    * (예: "[강점: 20대 여성 고객 비율 높음]을 활용하여 [약점: 객단가 낮음]을 개선하기 위한 '인스타그램 감성 세트 메뉴' 출시 전략...")

7.  **💡 즉시 실행 가능한 액션 플랜 (1-4주)**
    * 위 '개선 솔루션'을 바탕으로, 점주가 당장 1주차부터 4주차까지 실행할 수 있는 **구체적인 주차별 액션 플랜**을 2~3가지 제안해주세요.
    * (예: "1주차: [구체적 활동 A]", "2-3주차: [구체적 활동 B]"...)
"""
        
        # 3. LLM 호출 및 결과 반환
        final_report = call_gemini_llm(llm_prompt)
        # 2. 통합된 리포트 하나만 반환
        return final_report

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"""🚨 전방위 분석 중 오류가 발생했습니다.

**오류 상세 정보:**
- 오류 유형: {type(e).__name__}
- 오류 메시지: {str(e)}
- 가맹점 ID: {store_id}

**해결 방법:**
1. 가맹점 ID가 올바른지 확인해주세요
2. 데이터베이스에 해당 가맹점 정보가 있는지 확인해주세요
3. 문제가 지속되면 관리자에게 문의해주세요

**기술적 세부사항:**
{error_details}"""
        
# =============================================================================
##특화질문 도구
# =============================================================================

@tool
def floating_population_strategy_tool(store_id: str, df_all_join: pd.DataFrame, df_gender_age: pd.DataFrame, df_weekday_weekend: pd.DataFrame, df_timeband: pd.DataFrame) -> str:
    """
    [특화 질문 4번 모델]
    지하철역 인근 가맹점의 '선택영역' 유동인구 데이터(Long Format) 3종(성별연령, 요일(평일/주말), 시간대)을 심층 분석하여,
    신규 방문객을 단골로 전환하기 위한 '재방문 유도 전략'을 전문적으로 제안하는 도구.
    '유동인구', '지하철역', '출퇴근', '재방문 유도' 관련 질문에 사용된다.
    
    Args:
        store_id: 분석할 가맹점 ID
        df_all_join: 전체 JOIN 데이터 (가게 정보 조회용)
        df_gender_age: 성별연령대별_유동인구_선택영역.csv (Long Format)
        df_weekday_weekend: 요일별_유동인구_선택영역.csv (Long Format)
        df_timeband: 시간대별_유동인구_선택영역.csv (Long Format)
    
    Returns:
        LLM이 생성한 완성된 분석 리포트 문자열
    """
    try:
        # 1. 공통 헬퍼 함수 호출 (기본정보 출력을 위해 basic_info_content 다시 사용)
        basic_info_content, latest_store_data = _get_store_basic_info(store_id, df_all_join)
        
        if latest_store_data is None:
            return basic_info_content # 오류 메시지는 그대로 반환

        # --- 데이터 파싱 함수 (Long Format 전용) ---

        # 데이터 포맷팅 유틸
        def fmt(x, digits=1):
            try:
                return f"{float(x):,.{digits}f}"
            except Exception:
                return str(x)

        # 컬럼명 정규화
        def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            def norm(s):
                return str(s).replace("\u3000", "").replace(" ", "").strip()
            df.columns = [norm(c) for c in df.columns]
            rename_map = {}
            # Long Format 컬럼명 통일
            for c in df.columns:
                if c in ["지표","항목","분류","구분값","구분*","구분_"]:
                    rename_map[c] = "구분"
                elif c in ["인구", "인구수", "유동인구"]:
                    rename_map[c] = "인구(명)"
                elif c == "요일":
                    rename_map[c] = "요일"
                elif c == "시간대":
                    rename_map[c] = "시간대"
            if rename_map:
                df = df.rename(columns=rename_map)
            return df

        # Long Format CSV를 Dict로 변환하는 헬퍼
        def get_long_data_dict(df, key_col_candidates, val_col_candidates):
            df = normalize_columns(df)
            key_col = next((c for c in df.columns if c in key_col_candidates), df.columns[0])
            val_col = next((c for c in df.columns if c in val_col_candidates), df.columns[1])
            # [수정] errors='coerce'를 추가하여 숫자가 아닌 값(예: '선택 영역')을 NaN으로 처리
            return pd.to_numeric(df.set_index(key_col)[val_col], errors='coerce').dropna().to_dict()

        # 시간대 표기를 한글로
        def format_time_idx_to_korean(idx):
            try:
                if '~' in idx and '시' in idx:
                    parts = idx.split('~')
                    start = parts[0]
                    end = parts[1].replace('시', '')
                    if not start.endswith('시'):
                        start += '시'
                    return f"{start}부터 {end}시"
                return idx
            except Exception:
                return idx

        # DATA_BLOCK 생성 함수 (Long Format 파싱하도록 전면 수정)
        def make_data_block(gender_age_df, weekday_weekend_df, timeband_df, shop_row) -> str:
            lines = []
            shop = shop_row.iloc[0].to_dict() if len(shop_row) else {}
            shop_name = shop.get("MCT_NM", "가게명 미상")
            shop_addr = shop.get("MCT_BSE_AR", "주소 미상")
            # [*** 여기를 수정 ***] '지하철역' 대신 '상권' 컬럼을 사용
            shop_commercial_area = shop.get("HPSN_MCT_BZN_CD_NM", "상권 미상")
            shop_cat = shop.get("업종_정규화1", shop.get("업종_정규화2_대분류", "업종 미상"))

            meta = [
                f"* **가맹점 ID:** {store_id}",
                f"* **업종:** {shop_cat} (가게명: {shop_name})",
                f"* **위치:** {shop_addr}",
                # [*** 여기를 수정 ***] '인근 지하철역' -> '상권'으로 텍스트 변경
                f"* **상권:** {shop_commercial_area}",
            ]
            lines.append("\n".join(meta))
            
            try:
                ga_data = get_long_data_dict(gender_age_df, ['구분', '항목'], ['인구(명)'])
                ga_male = ga_data.get("남성", 0)
                ga_female = ga_data.get("여성", 0)
                # '일일' 키가 없을 수 있으므로, 남+여 합계로 ga_total을 계산
                ga_total = ga_male + ga_female 
                lines.append(f"* **일 평균 유동인구:** {fmt(ga_total,0)}명 (남성 {fmt(ga_male,0)}명, 여성 {fmt(ga_female,0)}명)")
            except Exception as e:
                lines.append(f"* **성/연령 데이터:** (파싱 오류: {e})")
            
            # '요일' (df_dayofweek) 분석 로직 아예 삭제
            
            try:
                ww_data = get_long_data_dict(weekday_weekend_df, ['구분', '항목'], ['인구(명)'])
                # '평일' 또는 '주중' 키를 모두 시도
                wk_val = float(ww_data.get("평일", ww_data.get("주중", 0)))
                we_val = float(ww_data.get("주말", 0))
                compare_text = "많음" if we_val > wk_val else "적음"
                if wk_val == we_val: compare_text = "비슷함"
                lines.append(f"* **평일/주말 유동인구:** 주말 {fmt(we_val,0)}명/일, 평일 {fmt(wk_val,0)}명/일 (주말이 평일보다 {compare_text})")
            except Exception as e:
                lines.append(f"* **평일/주말 데이터:** (파싱 오류: {e})")

            try:
                tb_data = get_long_data_dict(timeband_df, ['시간대'], ['인구(명)'])
                s = pd.Series(tb_data).astype(float).sort_values(ascending=False)[:3]
                time_lines = [f"{format_time_idx_to_korean(idx)}({fmt(val,0)}명)" for idx, val in s.items()]
                lines.append(f"* **주요 유동 시간대:** {', '.join(time_lines)}")
            except Exception as e:
                lines.append(f"* **시간대 데이터:** (파싱 오류: {e})")

            return "\n".join(lines)

        # 가맹점 정보 조회
        shop_row = df_all_join[df_all_join["ENCODED_MCT"] == store_id]
        if shop_row.empty:
            return f"🚨 분석 불가: '{store_id}' 가맹점의 데이터를 찾을 수 없습니다."

        # --- SYSTEM_PROMPT 및 build_prompt 함수 수정 ---
        
        # 프롬프트 수정 (주말 특성 분석 방식을 '평일 vs 주말'로 변경)
        SYSTEM_PROMPT = """
너는 '{shop_cat}' 업종 전문 상권 분석가이자 마케팅 전략가다.
너의 임무는 '{shop_station}' 역세권의 [DATA_BLOCK]을 심층 분석하고, 유동인구를 '재방문 단골'로 만들기 위한 창의적이고 구체적인 전략을 제안하는 것이다.
추상적인 조언은 금지. [DATA_BLOCK]의 숫자를 직접 언급하며 데이터에 기반한 전략만 제시하라.

[DATA_BLOCK]
{data_block}

[OUTPUT INSTRUCTION]
아래 1번, 2번 항목을 반드시 포함하여 리포트를 완성하라.

---
**1. {shop_station} 인근 유동인구 특성 및 출퇴근 시간대 변화 (심층 분석)**
([DATA_BLOCK]을 심층적으로 분석하라.)

* **출근 시간 (05시~09시):** 이 시간대 유동인구(약 {morning_pop}명)는 주로 직장인/학생일 것이다. 이들의 특성(예: 빠른 아침 식사, 테이크아웃 커피)을 추론하고, 현재 {shop_cat} 업종과의 연관성을 분석하라.
* **점심 시간 (09시~14시):** 점심시간대 유동인구(약 {lunch_pop}명)는 {shop_station} 인근 직장인 수요를 의미한다. 이 시간대 인구가 다른 시간대에 비해 어떤 수준인지 **숫자를 들어 비교**하고, {shop_cat} 업종에 어떤 기회가 있는지 분석하라.
* **퇴근 시간 (18시~23시):** 퇴근 시간대 유동인구(약 {evening_pop}명)는 이 상권의 **핵심 잠재 고객**이다. 이 인구가 하루 중 가장 많은지(혹은 적은지) **숫자로 비교**하고, 이들이 원하는 것(예: 간단한 식사, 동료와의 한잔)이 무엇일지 {shop_cat} 업종과 연결하여 추론하라.
* **주말 특성 (평일 vs 주말):** 주말 유동인구(약 {weekend_pop}명)가 평일(약 {weekday_pop}명) 대비 어떤지 **숫자로 비교**하라. 이 데이터는 {shop_station} 상권이 '오피스 상권'인지 '주말 나들이 상권'인지 판단할 핵심 근거다. 이것이 {shop_cat} 업종의 주말 운영에 어떤 의미를 주는지 분석하라.

---
**2. 재방문 고객 유도를 위한 집중 전략 (데이터 기반 제안)**
(신규 유동인구를 단골로 전환하는 것이 핵심 목표다. 위 1번 분석과 [DATA_BLOCK]을 근거로, **창의적이고 구체적인 전략 3-4가지**를 제안하라. '이상적인 이미지'의 예시처럼 상세하게 작성하되, **절대 하드코딩하지 말고 데이터에 맞춰 생성**하라.)

**[전략 예시 1: {morning_pop}명의 출근 고객 타겟 전략]** (←데이터에 근거하여 AI가 직접 소제목 생성)
* **아이디어:** (예: '딤섬 미식 여권' 같은 스탬프/포인트 시스템 도입)
* **구체적 실행 방안:** (예: 5회 방문 시 메뉴 1개 무료, 10회 방문 시 식사권 1만원 할인 등...)
* **데이터 근거:** (예: {shop_station} 역세권은 {total_pop}명이라는 유동인구가 있지만 재방문율이 낮다. 따라서 명확한 보상 체계로 재방문 동기를 부여해야 한다...)

**[전략 예시 2: {evening_pop}명의 퇴근 고객 타겟 전략]** (←데이터에 근거하여 AI가 직접 소제목 생성)
* **아이디어:** (예: '퇴근 후 딤맥 & 맥주' 프로모션)
* **구체적 실행 방안:** (예: 오후 6시~8시 사이 '딤섬 2종 + 생맥주 2잔' 세트를 2만원에 제공...)
* **데이터 근거:** (예: 하루 중 가장 많은 {evening_pop}명의 유동인구가 몰리는 퇴근 시간대에, {shop_cat} 업종의 특성을 살린 '퇴근길 즐거움'이라는 경험을 제공하여 재방문을 유도한다...)

**[전략 예시 3: {weekend_pop}명의 주말 고객 타겟 전략]** (←데이터에 근거하여 AI가 직접 소제목 생성)
* **아이디어:** (예: '주말 가족/연인 세트' 및 SNS 인증 이벤트)
* **구체적 실행 방안:** (예: 3-4인 가족 세트, 2인 연인 세트 구성. 인스타그램에 특정 해시태그(#뚝섬맛집 #{shop_name}) 인증 시 음료 1개 무료 제공...)
* **데이터 근거:** (예: 주말 유동인구가 {weekend_pop}명으로 평일({weekday_pop}명)보다 많다는 것은, 주말 나들이/데이트 고객이 많다는 의미다. 이들에게 '메뉴 선택의 고민'을 줄여주고 'SNS 인증'의 즐거움을 제공하여...)
"""

        # build_prompt 함수: SYSTEM_PROMPT에 동적 데이터를 주입
        def build_prompt(data_block: str, shop_row, gender_age_df, weekday_weekend_df, timeband_df) -> str:
            
            shop_dict = shop_row.iloc[0].to_dict()
            
            try:
                ga_data = get_long_data_dict(gender_age_df, ['구분', '항목'], ['인구(명)'])
                ga_male = ga_data.get("남성", 0)
                ga_female = ga_data.get("여성", 0)
                # '일일' 대신 합계 사용
                ga_total = ga_male + ga_female
            except Exception:
                ga_data = {}
                ga_male = 0
                ga_female = 0
                ga_total = 0
            
            try:
                ww_data = get_long_data_dict(weekday_weekend_df, ['구분', '항목'], ['인구(명)'])
                # '평일' 또는 '주중' 키를 모두 시도
                wk_val = float(ww_data.get("평일", ww_data.get("주중", 0)))
                we_val = float(ww_data.get("주말", 0))
            except Exception:
                ww_data = {}
                wk_val = 0
                we_val = 0
            
            try:
                tb_data = get_long_data_dict(timeband_df, ['시간대'], ['인구(명)'])
            except Exception:
                tb_data = {}

            # 프롬프트 포맷팅에 사용할 딕셔너리 생성
            format_data = {
                "data_block": data_block,
                "shop_cat": shop_dict.get("업종_정규화1", "요식업"),
                "shop_name": shop_dict.get("MCT_NM", "우리 가게"),
                # [*** 여기를 수정 ***] '지하철역' 대신 '상권' 컬럼을 프롬프트에 주입
                "shop_station": shop_dict.get("HPSN_MCT_BZN_CD_NM", "현 상권"),
                "total_pop": fmt(ga_total, 0), # 합계 total 사용
                "morning_pop": fmt(tb_data.get("05~09시", 0), 0),
                "lunch_pop": fmt(tb_data.get("09~12시", 0) + tb_data.get("12~14시", 0), 0),
                "evening_pop": fmt(tb_data.get("18~23시", 0), 0),
                "weekday_pop": fmt(wk_val, 0), # 로직이 적용된 wk_val 사용
                "weekend_pop": fmt(we_val, 0),
            }
            
            return SYSTEM_PROMPT.format(**format_data)

        # --- 함수 메인 로직 수정 (LLM 직접 호출) ---

        # 데이터 블록 생성
        # [수정] df_dayofweek 제거
        data_block = make_data_block(df_gender_age, df_weekday_weekend, df_timeband, shop_row)
        
        # 프롬프트 빌드 (더 많은 인자 전달)
        # [수정] df_dayofweek 제거
        prompt = build_prompt(data_block, shop_row, df_gender_age, df_weekday_weekend, df_timeband)
        
        # LLM 직접 호출
        llm_response = call_gemini_llm(prompt)
        
        # 최종 통합 리포트 생성
        # '기본정보' 다시 추가
        final_report = f"""
🚇 유동인구 기반 재방문 유도 전략 - '{store_id}' 가맹점 분석 리포트


{basic_info_content}

---
## 📊 유동인구 데이터 분석 리포트
{data_block}

---
## 🤖 AI 컨설턴트 상세 전략 제안
{llm_response}
"""
        # 2. 통합된 리포트 하나만 반환
        return final_report

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"""🚨 유동인구 전략 분석 중 오류가 발생했습니다.

**오류 상세 정보:**
- 오류 유형: {type(e).__name__}
- 오류 메시지: {str(e)}
- 가맹점 ID: {store_id}

**해결 방법:**
1. 가맹점 ID가 올바른지 확인해주세요
2. 데이터베이스에 해당 가맹점 정보가 있는지 확인해주세요
3. 문제가 지속되면 관리자에게 문의해주세요

**기술적 세부사항:**
{error_details}"""

@tool
def lunch_turnover_strategy_tool(store_id: str, df_all_join: pd.DataFrame, df_gender_age: pd.DataFrame, df_weekday_weekend: pd.DataFrame, df_dayofweek: pd.DataFrame, df_timeband: pd.DataFrame) -> str:
    """
    직장인 상권에 위치한 가맹점의 데이터를 분석하여, 점심 피크타임의 '회전율'을 극대화하기 위한 구체적인 운영 전략을 제안하는 도구.
    '직장인', '점심시간', '회전율', '효율' 관련 질문에 사용된다.
    
    Args:
        store_id: 분석할 가맹점 ID
        df_all_join: 전체 JOIN 데이터
        df_gender_age: 성별연령대별 유동인구 데이터
        df_weekday_weekend: 요일별 유동인구 데이터
        df_dayofweek: 요일별 유동인구 데이터
        df_timeband: 시간대별 유동인구 데이터
    
    Returns:
        LLM이 생성한 완성된 분석 리포트 문자열
    """
    try:
        # 1. 공통 헬퍼 함수 호출 (반환 변수명 변경)
        basic_info_content, latest_store_data = _get_store_basic_info(store_id, df_all_join)
        
        if latest_store_data is None:
            return basic_info_content # 오류 메시지는 그대로 반환

        # 데이터 도우미 함수
        def fmt(x, digits=1):
            try:
                return f"{float(x):,.{digits}f}"
            except Exception:
                return str(x)

        # DATA_BLOCK 생성 함수
        def make_data_block(monthly, gender_age, weekday_weekend, dayofweek, timeband, shop_row) -> str:
            lines = []
            shop = shop_row.iloc[0].to_dict() if len(shop_row) else {}
            shop_name = shop.get("MCT_NM", "가게명 미상")
            shop_addr = shop.get("MCT_BSE_AR", "주소 미상")
            # [*** 여기를 수정 ***] '지하철역' 대신 '상권' 컬럼을 사용하고 변수명은 shop_station을(를) shop_commercial_area로 변경
            shop_commercial_area = shop.get("HPSN_MCT_BZN_CD_NM", "상권 미상")
            shop_cat = shop.get("업종_정규화1", shop.get("업종_정규화2_대분류", "업종 미상"))
            shop_month = shop.get("TA_YM", "NA")
            
            # 시간대 표기를 한글로 바꿔주는 헬퍼 함수 (여기 추가)
            def format_time_idx_to_korean(idx):
                try:
                    if '~' in idx and '시' in idx:
                        parts = idx.split('~')
                        start = parts[0]
                        end = parts[1].replace('시', '')
                        if not start.endswith('시'):
                            start += '시'
                        return f"{start}부터 {end}시" # 예: "18~23시" -> "18시부터 23시"
                    return idx
                except Exception:
                    return idx # 오류 시 원본 반환

            # '데이터 분석 요약' 섹션에 맞게 포맷 변경 (## SHOP 제거, 불렛 포인트 사용)
            meta = [
                f"* **가맹점 ID:** {store_id}", # store_id 추가
                f"* **업종:** {shop_cat} (가게명: {shop_name})",
                f"* **위치:** {shop_addr}",
                # [*** 여기를 수정 ***] '인근 지하철역' -> '상권'으로 텍스트 변경
                f"* **상권:** {shop_commercial_area}",
                f"* **기준월:** {shop_month}",
            ]
            lines.append("\n".join(meta))

            # 성/연령
            ga_row = gender_age.iloc[0]
            ga_total = ga_row.get("일일")
            ga_male = ga_row.get("남성")
            ga_female = ga_row.get("여성")
            ga_lines = []
            if ga_total is not None:
                ga_lines.append(f"* **일 평균 유동인구:** {fmt(ga_total,0)}명 (남성 {fmt(ga_male,0)}명, 여성 {fmt(ga_female,0)}명)")
            lines.append("\n".join(ga_lines) if ga_lines else "")

            # 평/주말
            if "주중" in weekday_weekend.columns:
                row = weekday_weekend[weekday_weekend["구분"] == "인구"].iloc[0]
                wk_val = float(row['주중'])
                we_val = float(row['주말'])
                wk = fmt(wk_val, 0)
                we = fmt(we_val, 0)
                
                compare_text = "많음" if wk_val > we_val else "적음"
                if wk_val == we_val: compare_text = "비슷함"
                
                lines.append(f"* **평일/주말 유동인구:** 평일 {wk}명/일, 주말 {we}명/일 (평일이 주말보다 {compare_text})")
            else:
                lines.append("* **평일/주말 유동인구:** 정보 없음")

            # 시간대
            row = timeband[timeband["구분"] == "인구"].iloc[0]
            # [수정] .astype(float) 추가
            top3_series = row[["05~09시", "09~12시", "12~14시", "14~18시", "18~23시", "23~05시"]].astype(float).sort_values(ascending=False)[:3]
            # 헬퍼 함수 적용 및 포맷 변경
            time_lines = [f"{format_time_idx_to_korean(idx)}({fmt(val,0)}명)" for idx, val in top3_series.items()]
            lines.append(f"* **주요 유동 시간대:** {', '.join(time_lines)}")
            
            # 요일 (참고용으로 추가)
            if "월" in dayofweek.columns:
                pop_row = dayofweek[dayofweek["구분"] == "인구"].iloc[0]
                # [수정] .astype(float) 추가
                top2 = pop_row[["월", "화", "수", "목", "금", "토", "일"]].astype(float).sort_values(ascending=False)[:2]
                top_lines = [f"{idx}요일({fmt(val,0)}명)" for idx, val in top2.items()]
                lines.append(f"* **주요 유동 요일:** {', '.join(top_lines)}")

            return "\n".join(lines)

        # 가맹점 정보 조회
        shop_row = df_all_join[df_all_join["ENCODED_MCT"] == store_id]
        if shop_row.empty:
            return f"🚨 분석 불가: '{store_id}' 가맹점의 데이터를 찾을 수 없습니다."

        # 프롬프트 구성
        SYSTEM_PROMPT = """
너는 '백반/가정식' 업종 전문 외식 컨설턴트다.
너의 임무는 사장님에게 [DATA_BLOCK]을 근거로 점심시간 '회전율'을 극대화할 구체적이고 독창적인 액션 플랜을 제시하는 것이다.
추상적인 조언('열심히 하세요', '홍보하세요')은 절대 금지. [DATA_BLOCK]의 숫자를 직접 언급하며 데이터에 기반한 전략만 제시하라.

[DATA_BLOCK]
{data_block}

[OUTPUT INSTRUCTION]
아래 1번, 2번 항목을 반드시 포함하여 리포트를 완성하라.

---
**1) 점심시간 직장인구 특성 요약 (심층 분석)**
([DATA_BLOCK]의 데이터를 심층적으로 분석하라.)

* **평일 vs 주말 분석:** 평일 유동인구(예: {wk}명)와 주말 유동인구(예: {we}명)를 **숫자로 직접 비교**하라. 이 차이가 '직장인 상권'이라는 가게 특성과 맞는지, 혹은 주말에 다른 기회가 있는지 분석하라.
* **시간대별 심층 분석:** 점심 피크 시간대(예: 09시~12시 또는 12시~14시)의 유동인구를 **숫자로 언급**하며, 저녁 등 다른 피크 시간대(예: 18시~23시)와 **숫자로 비교**하라. 점심 유동인구가 상대적으로 적다면(혹은 많다면) 이것이 사장님에게 어떤 의미(위기/기회)인지 해석하라.
* **고객 성별 분석:** 남성(예: {male}명)과 여성(예: {female}명) 비율을 **숫자로 비교**하라. 이 성비가 백반/가정식 업종의 주 타겟(예: 빠른 한 끼를 원하는 남성 직장인)과 일치하는지, 이들이 무엇을 선호할지 **데이터에 근거하여 추론**하라.
* **종합 결론:** 위 3가지 분석을 토대로, 사장님이 인지해야 할 '우리 가게 점심 고객'의 핵심 프로필을 1-2문장으로 최종 정의하라.

---
**2) 점심 피크타임 회전율 극대화 전략 (데이터 기반 제안)**
(백반집의 핵심 KPI는 '회전율'임을 명심하고, 아래 4가지 카테고리에 맞춰 **[DATA_BLOCK]의 데이터를 근거로** 구체적인 전략을 제안하라. 절대로 정해진 답변을 하지 말고 데이터에 맞춰 창의적으로 제안하라.)

**1. 메뉴 전략 (Menu Simplification)**
* **전략:** [DATA_BLOCK]의 **고객 특성(예: 남/여 비율, 평일/주말 차이)**을 근거로, 가장 효과적인 점심 메뉴 구성을 제안하라. (예: 단일 특선 메뉴, 2-3가지 선택 메뉴 등)
* **근거:** 왜 이 메뉴 구성이 [DATA_BLOCK]의 고객(예: **남성 {male}명**으로 여성이 {female}명보다 많아... / **평일 {wk}명**으로 주말보다 적어...)에게 매력적이며 회전율을 높이는지 **숫자를 들어** 설명하라.

**2. 주문/결제 시스템 (Ordering/Payment Flow)**
* **전략:** [DATA_BLOCK]의 **피크 시간대(예: '18시부터 23시({top1_pop}명)')**의 혼잡도를 점심시간과 비교하여, 점심시간에 적용할 가장 효율적인 주문/결제 방식을 제안하라.
* **근거:** 이 방식이 어떻게 주문 병목현상을 해결하고, 직원이 **{shop_cat}** 업종의 핵심인 테이블 정리/반찬 리필에 집중하게 만드는지 설명하라.

**3. 좌석 배치 및 동선 (Seating & Flow)**
* **전략:** [DATA_BLOCK]의 **고객 성비(남성 {male}명 vs 여성 {female}명)**와 **상권(예: {shop_station} 인근)** 특성을 고려할 때, 1인/2인/4인석의 이상적인 비율을 제안하라. (예: 1-2인석 비중 확대, 바(Bar) 좌석 도입 등)
* **근거:** 이 좌석 배치가 어떻게 1인 고객(혹은 2인 고객)의 빠른 식사를 유도하고, 전체적인 테이블 회전 속도를 높이는지 설명하라.

**4. 회전율 촉진 프로모션 (Turnover Promotion)**
* **전략:** [DATA_BLOCK]의 **데이터(예: 점심 유동인구가 {lunch_pop}명으로 저녁보다 적음 / 평일이 주말보다 적음)**를 활용하여, 고객의 자발적인 빠른 식사를 유도하거나 혹은 점심시간대 방문을 유도할 창의적인 프로모션 1가지를 제안하라.
* **근거:** 이 프로모션이 어떻게 고객 경험을 해치지 않으면서(오히려 만족도를 높이면서) 평균 식사 시간을 단축시키거나, 혹은 가장 한가한 시간대의 매출을 보완할 수 있는지 **데이터에 기반하여** 설명하라.
"""

        QUESTION = (
            "우리 가게는 직장인 고객이 주요 타겟이며, 점심시간에 해당하는 유동인구와 직장인구의 분석을 상세히 설명해줘.\n"
            "1) 점심시간 직장인구 특성을 요약해줘\n"
            "2) 점심 피크타임에 회전율을 높이기 위한 전략을 제시해줘"
        )

        def build_prompt(question: str, data_block: str, shop_row, gender_age, weekday_weekend, timeband) -> str:
            # 프롬프트에 동적으로 데이터를 주입하기 위해 변수 추가
            
            # 데이터 추출 (데이터프레임에서 직접)
            ga_row = gender_age.iloc[0]
            ww_row = weekday_weekend[weekday_weekend["구분"] == "인구"].iloc[0]
            tb_row = timeband[timeband["구분"] == "인구"].iloc[0]
            shop_dict = shop_row.iloc[0].to_dict()

            # 'top1_pop' 계산 버그 수정
            time_cols = ["05~09시", "09~12시", "12~14시", "14~18시", "18~23시", "23~05시"]
            # tb_row[time_cols]로 숫자 컬럼만 선택 -> astype(float)로 변환 -> 정렬 -> 첫번째 값(iloc[0]) 추출
            top1_val = tb_row[time_cols].astype(float).sort_values(ascending=False).iloc[0]

            # 프롬프트 포맷팅에 사용할 딕셔너리 생성
            format_data = {
                "data_block": data_block,
                "wk": fmt(ww_row.get('주중', 0), 0),
                "we": fmt(ww_row.get('주말', 0), 0),
                "male": fmt(ga_row.get('남성', 0), 0),
                "female": fmt(ga_row.get('여성', 0), 0),
                "top1_pop": fmt(top1_val, 0), # 수정된 top1_val 사용
                "lunch_pop": fmt(pd.to_numeric(tb_row.get('09~12시', 0)) + pd.to_numeric(tb_row.get('12~14시', 0)), 0), # 09-14시 인구 합산 및 숫자 변환
                "shop_cat": shop_dict.get("업종_정규화1", "요식업"),
                # [*** 여기를 수정 ***] '지하철역' 대신 '상권' 컬럼을 프롬프트에 주입
                "shop_station": shop_dict.get("HPSN_MCT_BZN_CD_NM", "현 상권")
            }
            
            # .format()을 사용하여 SYSTEM_PROMPT에 데이터 주입
            return SYSTEM_PROMPT.format(**format_data)

        # 데이터 블록 생성
        monthly = pd.DataFrame()  # 사용 안함
        data_block = make_data_block(monthly, df_gender_age, df_weekday_weekend, df_dayofweek, df_timeband, shop_row)
        
        # 프롬프트 빌드 (더 많은 인자 전달)
        prompt = build_prompt(QUESTION, data_block, shop_row, df_gender_age, df_weekday_weekend, df_timeband)
        
        # LLM 직접 호출
        llm_response = call_gemini_llm(prompt)
        
        # 최종 리포트 구성 변경
        final_report = f"""
  점심시간 회전율 극대화 전략 - '{store_id}' 가맹점 분석 리포트

{basic_info_content}

---
## 📊 유동인구 데이터 분석
{data_block}

---
## 🤖 AI 컨설턴트 상세 전략 제안
{llm_response}
"""
        # 2. 통합된 리포트 하나만 반환
        return final_report

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"""🚨 점심시간 회전율 전략 분석 중 오류가 발생했습니다.

**오류 상세 정보:**
- 오류 유형: {type(e).__name__}
- 오류 메시지: {str(e)}
- 가맹점 ID: {store_id}

**해결 방법:**
1. 가맹점 ID가 올바른지 확인해주세요
2. 데이터베이스에 해당 가맹점 정보가 있는지 확인해주세요
3. 문제가 지속되면 관리자에게 문의해주세요

**기술적 세부사항:**
{error_details}"""