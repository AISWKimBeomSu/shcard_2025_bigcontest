"""
AI 비밀상담사 분석 도구 모음
3개의 독립적인 분석 모델을 LangChain Tool로 리팩토링
"""

import pandas as pd
import numpy as np
import requests
import json
from typing import Dict, Any, List, Tuple
from langchain_core.tools import tool

# =============================================================================
# 공통 유틸리티 함수들
# =============================================================================

def get_score_from_raw(raw_value):
    """원시 값을 점수로 변환하는 헬퍼 함수"""
    if pd.isna(raw_value):
        return np.nan
    raw_str = str(raw_value)
    score_map = {"10%이하": 6, "10-25%": 5, "25-50%": 4, "50-75%": 3, "75-90%": 2, "90%초과": 1}
    for key, score in score_map.items():
        if key in raw_str:
            return score
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
def cafe_marketing_tool(store_id: str, df_cafe_customers: pd.DataFrame, df_all_join: pd.DataFrame, df_prompt_dna: pd.DataFrame) -> str:
    """
    카페 업종 가맹점의 주요 방문 고객 특성을 데이터 기반으로 분석하고, 
    가장 효과적인 마케팅 채널과 구체적인 홍보 문구를 추천하는 전문 도구.
    '카페', '고객 분석', '홍보' 관련 질문에 사용된다.
    
    Args:
        store_id: 분석할 가맹점 ID
        df_cafe_customers: 카페 가맹점별 주요고객 데이터
        df_all_join: 전체 JOIN 데이터
        df_prompt_dna: AI상담사 핵심전략 프롬프트 데이터
    
    Returns:
        분석 결과 및 마케팅 전략 제안 리포트
    """
    try:
        # 1단계: 데이터 분석 엔진
        customer_data = df_cafe_customers[df_cafe_customers['ENCODED_MCT'] == store_id]
        if customer_data.empty:
            return f"분석 실패: '{store_id}' 가맹점의 주요 고객 데이터를 찾을 수 없습니다."
        
        # 주요 고객층 분석 (격차 방식 적용)
        persona_columns = list(PERSONA_MAP.keys())
        store_persona_data = customer_data[persona_columns].iloc[0].astype(float)
        
        max_value = store_persona_data.max()
        threshold = max_value - 5.0
        
        top_segments = store_persona_data[store_persona_data >= threshold]
        
        # 결과가 2개를 초과할 경우, 가장 높은 상위 2개만 선택
        if len(top_segments) > 2:
            top_segments = top_segments.sort_values(ascending=False).head(2)
            
        if top_segments.empty:
            return f"분석 실패: '{store_id}' 가맹점의 유효한 주요 고객층을 찾을 수 없습니다."
        
        main_personas_info = [PERSONA_MAP[seg] for seg in top_segments.index]
        main_personas_str = ", ".join([f"{p['name']}({p['desc']})" for p in main_personas_info])
        
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
        
        core_strategy = dna_row['핵심경영전략'].iloc[0]

        # 2단계: LLM 호출을 위한 프롬프트 생성
        prompt_for_gemini = f"""
너는 소상공인 카페 사장님을 위한 전문 마케팅 전략가 'AI 비밀상담사'야.

### 데이터 분석 결과
- **[WHO] 우리 가게 핵심 고객:** {main_personas_str}
- **[WHAT] 가게의 성공 전략:** '{core_strategy}'

### 과업 지시
위 분석 결과를 바탕으로, 사장님이 **즉시 실행할 수 있는 '마케팅 채널'과 '홍보 방안'**을 추천해줘.

### 결과물 형식 및 작성 가이드라인
1.  **답변 형식:** 결과물은 **'📈 추천 마케팅 채널'**과 **'💡 추천 홍보 방안'** 두 부분으로 명확하게 나누어줘.
2.  **채널 추천:** 각 채널을 추천할 때, **왜 이 고객들에게 이 채널이 효과적인지** 핵심 이유를 한 문장으로 요약해줘.
3.  **홍보 방안:**
    - **최소 3가지 이상**의 구체적인 액션 플랜을 제시해줘.
    - 각 방안마다 **'무엇을(What)'**과 **'어떻게(How)'** 할 수 있는지 사장님이 바로 이해하도록 실제 예시를 포함해줘.
    - 만약 핵심 고객이 여러 그룹이라면, 그들의 **공통점을 공략하거나 시너지를 낼 수 있는** 통합적인 아이디어를 중심으로 제안해줘.
4.  **전체 톤앤매너:** 사장님이 쉽게 이해하고 바로 용기를 얻어 실행해 볼 수 있도록, 친근하고 명확하며 설득력 있는 톤앤매너를 사용해줘.
"""

        # 3단계: 최종 리포트 조합
        final_report = f"""
======================================================================
      🤖 AI 비밀상담사 - '{store_id}' 가맹점 맞춤 전략 리포트
======================================================================

### 📊 데이터 분석 요약

* **[WHO] 우리 가게 핵심 고객:** {main_personas_str}
* **[WHAT] 시장에서 살아남을 열쇠:** {core_strategy}

----------------------------------------------------------------------

### 🚀 AI가 제안하는 맞춤 마케팅 전략

{prompt_for_gemini}

**💡 즉시 실행 가능한 마케팅 전략:**

1. **타겟 고객 맞춤 채널 활용**
   - {main_personas_str} 고객층의 특성을 고려한 맞춤형 마케팅 채널 선택
   - 각 고객층의 선호하는 소통 방식과 플랫폼 활용

2. **핵심 전략 기반 메시지 개발**
   - '{core_strategy}' 전략을 반영한 마케팅 메시지 구성
   - 고객의 니즈와 가게의 강점을 연결하는 스토리텔링

3. **단계별 실행 계획**
   - 1단계: 고객층별 맞춤 채널 설정
   - 2단계: 핵심 메시지 개발 및 테스트
   - 3단계: 성과 측정 및 최적화

"""
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
        target_store_all_months = df_all_join[df_all_join['ENCODED_MCT'] == store_id]
        if target_store_all_months.empty:
            return f"🚨 분석 불가: 데이터셋에서 '{store_id}' 가맹점 정보를 찾을 수 없습니다."

        target_store = target_store_all_months.sort_values(by='TA_YM', ascending=False).iloc[0]

        # 재방문율 계산 (월별 평균)
        target_revisit_rate_series = target_store_all_months['MCT_UE_CLN_REU_RAT'].dropna()
        target_revisit_rate = 0.0 if target_revisit_rate_series.empty else target_revisit_rate_series.mean()

        if target_revisit_rate >= 30:
            latest_month = target_store.get('TA_YM', '최신')
            return f"✅ 분석 결과: 월 평균 재방문율이 {target_revisit_rate:.1f}%로 양호합니다. (최신월: {latest_month})"

        industry, commercial_area = target_store['업종_정규화2_대분류'], target_store['HPSN_MCT_BZN_CD_NM']
        area_name = commercial_area if pd.notna(commercial_area) else "비상권"
        
        # 경쟁 그룹 설정
        peer_group_filter = (df_all_join['업종_정규화2_대분류'] == industry) & (df_all_join['ENCODED_MCT'] != store_id)
        if pd.isna(commercial_area):
            peer_group = df_all_join[peer_group_filter & (df_all_join['HPSN_MCT_BZN_CD_NM'].isna())]
        else:
            peer_group = df_all_join[peer_group_filter & (df_all_join['HPSN_MCT_BZN_CD_NM'] == commercial_area)]

        if len(peer_group) < 3:
            return f"🟡 분석 보류: 비교 분석을 위한 경쟁 그룹이 부족합니다."
            
        revisit_threshold = peer_group['MCT_UE_CLN_REU_RAT'].quantile(0.5)
        successful_peers = peer_group[peer_group['MCT_UE_CLN_REU_RAT'] >= revisit_threshold]
        if successful_peers.empty:
            return f"🟡 분석 보류: 성공 그룹을 찾을 수 없습니다."

        # 3대 지표 계산
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
        
        # 페르소나 진단
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

        # 최종 리포트 생성
        final_report = f"""
======================================================================
      🩺 AI 재방문율 진단 - '{store_id}' 가맹점 분석 리포트
======================================================================

### 📊 현재 상황 진단

* **업종/상권:** {industry} / {area_name}
* **현재 재방문율:** {target_revisit_rate:.1f}% (월 평균)
* **AI 진단 페르소나:** {persona}

### 🔍 3대 핵심 동인 분석

**① 가격 경쟁력 (객단가)**
- 내 가게: {target_price_score_avg:.2f}점
- 성공 그룹 평균: {peer_price_score_avg:.2f}점
- 격차: {analysis_results['price_competitiveness']['gap']:.2f}점

**② 핵심 고객층 (거주자 비율)**
- 내 가게: {target_resident_ratio_avg:.1f}%
- 성공 그룹 평균: {peer_resident_ratio_avg:.1f}%
- 격차: {analysis_results['audience_alignment']['gap']:.1f}%p

**③ 채널 확장성 (배달 비율)**
- 내 가게: {target_delivery_ratio_avg:.1f}%
- 성공 그룹 평균: {peer_delivery_avg:.1f}%
- 격차: {analysis_results['channel_expansion']['gap']:.1f}%p

### 🚀 개선 전략 제안

**핵심 성공 변수:** {key_factor}
**핵심 경영 전략:** {key_strategy}

**A/B 전략 옵션:**

**전략 A (강점 강화/차별화):**
- 현재 잘하고 있는 부분을 더욱 강화
- 시장의 규칙을 따르는 대신 새로운 규칙을 만드는 전략

**전략 B (약점 보완/동기화):**
- 성공 그룹의 전략을 벤치마킹
- 안정적인 성공 방정식을 따르는 전략

### 💡 즉시 실행 가능한 액션 플랜

1. **단기 긴급 처방 (1-2주)**
   - {persona} 문제 해결을 위한 즉시 실행 가능한 솔루션

2. **중장기 핵심 전략 (1-3개월)**
   - {key_strategy} 기반의 체계적인 개선 계획

3. **성과 측정 및 최적화**
   - 재방문율 변화 모니터링 및 전략 조정

"""
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
    """경쟁 그룹 내 백분위 순위를 0-100점 척도의 '건강 점수'로 변환하고 가중치를 적용"""
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

@tool # 여기 수정해야함(가맹점 명 형식 오류)
def search_merchant_tool(merchant_name: str, df_all_join: pd.DataFrame) -> str:
    """
    가맹점 이름을 입력받아, 해당 가맹점의 기본 정보(업종, 주소, 개설일 등)를 
    데이터베이스에서 검색하는 도구.
    사용자가 가게의 기본 정보만 요청할 때 사용된다.
    
    Args:
        merchant_name: 검색할 가맹점명 (부분 일치 지원)
        df_all_join: 전체 JOIN 데이터
    
    Returns:
        가맹점 검색 결과 리포트
    """
    try:
        # 가맹점명으로 검색 (exact match)
        result = df_all_join[df_all_join['가맹점명'].astype(str).str.replace('*', '') == merchant_name.replace('*', '')]
        
        if len(result) == 0:
            return f"""
🚨 검색 결과 없음

'{merchant_name}'에 해당하는 가맹점을 찾을 수 없습니다.

💡 검색 팁:
- 정확한 가맹점명을 입력해주세요
- '*' 기호는 자동으로 제거됩니다
- 대소문자는 구분하지 않습니다

조회 가능한 예시: 동대*, 유유*, 똥파*, 본죽*, 본*, 원조*, 희망*, 혁이*, H커*, 케키*
"""
        
        # 최신 데이터 선택 (TA_YM 기준)
        latest_result = result.sort_values(by='TA_YM', ascending=False).iloc[0]
        
        # 기본 정보 추출
        store_name = latest_result.get('가맹점명', '정보 없음')
        industry = latest_result.get('업종_정규화2_대분류', '정보 없음')
        address = latest_result.get('HPSN_MCT_BZN_CD_NM', '정보 없음')
        commercial_area = latest_result.get('HPSN_MCT_BZN_CD_NM', '비상권')
        
        # 매출 관련 정보
        revenue_level = latest_result.get('RC_M1_SAA', '정보 없음')
        customer_count_level = latest_result.get('RC_M1_UE_CUS_CN', '정보 없음')
        avg_amount_level = latest_result.get('RC_M1_AV_NP_AT', '정보 없음')
        
        # 고객 비율 정보
        new_customer_ratio = latest_result.get('MCT_UE_CLN_NEW_RAT', 0)
        revisit_ratio = latest_result.get('MCT_UE_CLN_REU_RAT', 0)
        resident_ratio = latest_result.get('RC_M1_SHC_RSD_UE_CLN_RAT', 0)
        delivery_ratio = latest_result.get('DLV_SAA_RAT', 0)
        
        # 운영 기간
        operation_period = latest_result.get('MCT_OPE_MS_CN', '정보 없음')
        
        # 최신 월
        latest_month = latest_result.get('TA_YM', '정보 없음')
        
        # 검색 결과 리포트 생성
        report = f"""
======================================================================
      🏪 가맹점 기본 정보 - '{store_name}' 검색 결과
======================================================================

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

### 🔍 추가 분석 가능
이 가맹점에 대해 더 자세한 분석을 원하시면 다음을 요청해주세요:
- **마케팅 전략 분석** (카페 업종인 경우)
- **재방문율 개선 방안** (재방문율이 낮은 경우)
- **전체적인 강점/약점 진단**

======================================================================
"""
        
        return report
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"""🚨 가맹점 검색 중 오류가 발생했습니다.

**오류 상세 정보:**
- 오류 유형: {type(e).__name__}
- 오류 메시지: {str(e)}
- 검색어: {merchant_name}

**해결 방법:**
1. 가맹점명이 올바른지 확인해주세요
2. 데이터베이스에 해당 가맹점 정보가 있는지 확인해주세요
3. 문제가 지속되면 관리자에게 문의해주세요

**기술적 세부사항:**
{error_details}"""

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
        store_df = df_all_join[df_all_join['ENCODED_MCT'] == store_id].tail(12)
        if store_df.empty:
            return f"🚨 분석 불가: '{store_id}' 가맹점의 데이터를 찾을 수 없습니다."
            
        category, commercial_area = store_df[['업종_정규화2_대분류', 'HPSN_MCT_BZN_CD_NM']].iloc[0]
        
        if pd.notna(commercial_area):
            benchmark_type = "동일 상권 내 동종업계"
            benchmark_df = df_all_join[(df_all_join['업종_정규화2_대분류'] == category) & (df_all_join['HPSN_MCT_BZN_CD_NM'] == commercial_area)]
        else:
            benchmark_type = "비상권 지역의 동종업계"
            benchmark_df = df_all_join[(df_all_join['업종_정규화2_대분류'] == category) & (df_all_join['HPSN_MCT_BZN_CD_NM'].isna())]
        
        # 분석할 지표들 (실제 존재하는 컬럼명으로 수정)
        metrics_to_analyze = [
            {'name': '매출 규모', 'col': 'RC_M1_SAA', 'type': 'tier', 'higher_is_better': False},
            {'name': '방문 고객 수', 'col': 'RC_M1_UE_CUS_CN', 'type': 'tier', 'higher_is_better': False},
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

        all_scores = []
        for metric in metrics_to_analyze:
            if metric['type'] == 'tier':
                # tier 타입은 텍스트 값이므로 숫자로 변환
                store_val = get_score_from_raw(store_df[metric['col']].iloc[-1]) if not store_df.empty else np.nan
                benchmark_series = benchmark_df[metric['col']].apply(get_score_from_raw)
            else:
                # ratio 타입은 숫자 값
                store_val = store_df[metric['col']].mean()
                benchmark_series = benchmark_df.groupby('ENCODED_MCT')[metric['col']].mean()
            
            score = get_percentile_score(store_val, benchmark_series, metric['higher_is_better'])
            
            store_display, benchmark_display = "", ""
            if metric['type'] == 'ratio':
                store_display = f"{store_val:.1%}"
                benchmark_display = f"{benchmark_series.mean():.1%}"
            elif metric['type'] == 'tier':
                store_display = store_df[metric['col']].iloc[-1] if not store_df.empty else "N/A"
                benchmark_display = benchmark_df[metric['col']].mode()[0] if not benchmark_df.empty and not benchmark_df[metric['col']].mode().empty else "N/A"

            all_scores.append({
                'metric': metric['name'], 
                'score': f"{score:.1f}점",
                'store_value_display': store_display,
                'benchmark_value_display': benchmark_display
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
            'benchmark_value_display': f"Top 2: {[n.replace('M12_','').replace('_RAT','') for n in area_top2.index.tolist()]}"
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

        # 최종 리포트 생성
        final_report = f"""
======================================================================
      📊 AI 전방위 진단 - '{store_id}' 가맹점 건강도 분석 리포트
======================================================================

### 📈 분석 개요

* **분석 대상:** {store_id}
* **분석 기준:** {benchmark_type} (최근 12개월 데이터)
* **업종/상권:** {category} / {commercial_area if pd.notna(commercial_area) else '비상권'}

### ✅ 강점 요약 (Top 3)

"""
        
        if sorted_strengths:
            for i, s in enumerate(sorted_strengths[:3], 1):
                score = float(s['score'].replace('점',''))
                interpretation = f"경쟁점 대비 상위 {100-score:.0f}% 수준의 뛰어난 성과"
                final_report += f"""
**{i}. {s['metric']} (건강 점수: {s['score']})**
- 해석: {interpretation}
- 데이터: 우리 가게({s['store_value_display']}) vs 경쟁점({s['benchmark_value_display']})
"""
        else:
            final_report += "특별한 강점이 발견되지 않았습니다.\n"

        final_report += """
### ❌ 약점 요약 (Top 3)

"""
        
        if sorted_weaknesses:
            for i, w in enumerate(sorted_weaknesses[:3], 1):
                score = float(w['score'].replace('점',''))
                interpretation = f"경쟁점 대비 하위 {score:.0f}% 수준으로 개선이 필요함" if score < 40 else "전반적으로 양호하나 상대적으로 아쉬운 지표"
                final_report += f"""
**{i}. {w['metric']} (건강 점수: {w['score']})**
- 해석: {interpretation}
- 데이터: 우리 가게({w['store_value_display']}) vs 경쟁점({w['benchmark_value_display']})
"""
        else:
            final_report += "특별한 약점이 발견되지 않았습니다.\n"

        final_report += """
### 🚀 종합 진단 및 개선 솔루션

**핵심 문제 진단:**
"""
        
        if sorted_weaknesses:
            main_weakness = sorted_weaknesses[0]
            final_report += f"- 가장 시급한 개선 과제: {main_weakness['metric']}\n"
            final_report += f"- 현재 수준: {main_weakness['store_value_display']}\n"
            final_report += f"- 목표 수준: {main_weakness['benchmark_value_display']}\n"
        else:
            final_report += "- 전반적으로 양호한 상태입니다.\n"

        final_report += """
**마케팅 제안:**

1. **강점 활용 전략**
   - 현재 잘하고 있는 부분을 더욱 강화하여 경쟁 우위 확보
   - 강점을 마케팅 포인트로 활용한 차별화 전략

2. **약점 보완 전략**
   - 가장 약한 부분부터 단계적으로 개선
   - 성공 사례 벤치마킹을 통한 빠른 개선

3. **통합 최적화 전략**
   - 강점과 약점의 시너지 효과 창출
   - 고객 경험 전반의 품질 향상

### 💡 즉시 실행 가능한 액션 플랜

1. **1주차: 긴급 개선**
   - 가장 약한 지표 1개에 집중한 즉시 개선 조치

2. **2-4주차: 단계적 개선**
   - 나머지 약점들을 순차적으로 개선
   - 강점을 더욱 강화하는 전략 실행

3. **1-3개월: 지속적 최적화**
   - 성과 측정 및 전략 조정
   - 장기적인 경쟁력 확보

"""
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
