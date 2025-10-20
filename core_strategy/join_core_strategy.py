import pandas as pd
import numpy as np

# --- ⚙️ 설정: 분석할 특징 파일 경로 ---
path_commercial = r"C:\Users\신승민\Downloads\상권업종별_상위군공통특징_필터링.csv"
path_non_commercial = r"C:\Users\신승민\.vscode\비상권업종별_상위군_공통특징_필터링.csv"

# ------------------------------------------------------------------------------------

# --- 💡 인코딩 자동 감지 함수 ---
def robust_read_csv(path):
    try:
        return pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        print(f"알림: '{path}' 파일을 utf-8로 읽는 데 실패하여 cp949 방식으로 다시 시도합니다.")
        return pd.read_csv(path, encoding='cp949')

# ------------------------------------------------------------------------------------

# --- 🧠 AI의 지식 베이스: 최종 핵심 전략 매핑표 ---
strategy_map = {
    'RC_M1_SHC_RSD_UE_CLN_RAT': '주민 생활권 중심의 커뮤니티 거점화', 'RC_M1_SHC_WP_UE_CLN_RAT': '직장인 일상 속 가치 경험 제공',
    'RC_M1_SHC_FLP_UE_CLN_RAT': '유동인구 대상의 발견 가치 극대화', 'MCT_UE_CLN_NEW_RAT': '첫 방문 경험 만족도 증대 및 전환 유도',
    'MCT_UE_CLN_REU_RAT': '기존 고객 충성도 강화 및 관계 심화', 'RC_M1_AV_NP_AT': '객단가 상승을 위한 상품 가치 제안',
    'DLV_SAA_RAT': '운영 효율성 개선을 통한 고객 만족도 증대', 'APV_CE_RAT': '운영 효율성 개선을 통한 고객 만족도 증대',
    'M12_FME_1020_RAT': '관계지향적 자기표현, 트렌드 동조 및 모방 소비, 감성적 만족', 'M12_MAL_1020_RAT': '경험 기반의 트렌드 검증, 사회적 상호작용, 충동적 소비 성향',
    'M12_FME_30_RAT': '가치 극대화, 전략적 시간관리, 상징적 소비(과시 소비) ', 'M12_MAL_30_RAT': '효율성 및 기능성 추구, 신뢰 기반의 합리성, 목표지향적 소비',
    'M12_FME_40_RAT': '가족 중심의 웰빙, 신뢰 기반의 정보 탐색 및 활용 , 계획 구매 ', 'M12_MAL_40_RAT': '위험 회피 및 안정성, 품질 및 신뢰성 중시, 합리적 대안 선택',
    'M12_FME_50_RAT': '사회적 관계망 형성, 상호작용 및 소통, 정서적 유대감', 'M12_MAL_50_RAT': '지역 커뮤니티 내 소속감, 익숙함과 편안함 추구, 단골 관계 중시',
    'M12_FME_60_RAT': '신체적·정서적 웰빙, 삶의 활력 추구, 자기관리 및 건강', 'M12_MAL_60_RAT': '의미 중심의 경험, 관계 가치에 대한 투자, 시간의 질적 활용',
    'MCT_OPE_MS_CN': '오래된 가게의 힘', 'RC_M1_TO_UE_CT': '방문 빈도 증대를 통한 박리다매 전략',
    'RC_M1_UE_CUS_CN': '광범위한 고객층 내 핵심 팬덤 구축'
}


def create_ai_prompt_file(commercial_path, non_commercial_path):
    """
    상권/비상권 특징 파일을 통합하고, 각 케이스별 가장 중요한 변수(성공 DNA)를 찾아
    핵심 경영 전략과 매핑한 최종 프롬프트 파일을 생성하는 함수.
    """
    try:
        print("--- AI 핵심 전략 프롬프트 생성 시작 ---")

        # --- 1단계: 데이터 로드 및 통합 ---
        df_com = robust_read_csv(commercial_path)
        df_non_com = robust_read_csv(non_commercial_path)

        df_non_com['상권'] = '비상권'
        df_total = pd.concat([df_com, df_non_com], ignore_index=True)
        print(" 상권/비상권 특징 데이터를 성공적으로 통합했습니다.")

        # --- 2단계: 각 경우의 수별 '가장 중요한' 성공 DNA 추출 ---
        # 그룹핑 기준: 상권, 업종
        group_cols = ['상권', '업종']
        
        # 각 그룹에서 Cohen's d 값이 가장 높은 행의 인덱스를 찾음
        # (가장 뚜렷한 차이를 보이는 변수가 가장 중요한 변수)
        idx = df_total.groupby(group_cols)['Cohen_d(상위-나머지)'].idxmax()
        
        # 해당 인덱스를 사용하여 가장 중요한 특징들만 담은 데이터프레임을 생성
        df_prompt = df_total.loc[idx].copy()
        print("모든 경우의 수에 대한 '가장 중요한 성공 DNA'를 추출했습니다.")

        # --- 3단계: 성공 DNA를 '핵심 경영 전략'으로 번역 및 연결 ---
        df_prompt['핵심경영전략'] = df_prompt['특징변수'].map(strategy_map)
        df_prompt['핵심경영전략'].fillna('기타 운영 효율화', inplace=True)
        print("성공 DNA를 우리가 만든 '핵심 경영 전략'과 성공적으로 연결했습니다.")

        # --- 4단계: 최종 프롬프트 파일 저장 ---
        output_df = df_prompt[['상권', '업종', '특징변수', '핵심경영전략', 'Cohen_d(상위-나머지)']]
        output_df = output_df.rename(columns={'특징변수': '핵심성공변수(DNA)', 'Cohen_d(상위-나머지)': '중요도(Cohen_d)'})
        
        output_path = "AI상담사_핵심전략_프롬프트.csv"
        output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*60)
        print("🎉 AI 핵심 로직 완성! 🎉")
        print("="*60)
        print(f"모든 경우의 수에 대한 최종 전략 방향성이 담긴")
        print(f"'{output_path}' 파일이 성공적으로 생성되었습니다.")
        print("이제 이 파일을 기반으로 AI 상담사의 최종 답변을 생성할 수 있습니다.")
        print("\n--- 생성된 프롬프트 파일 샘플 (상위 5개) ---")
        print(output_df.head().to_string())

    except FileNotFoundError:
        print("오류: 파일 경로를 찾을 수 없습니다. 상단의 경로 설정을 확인해주세요.")
    except Exception as e:
        print(f"작업 중 오류가 발생했습니다: {e}")


# --- 🚀 메인 코드 실행 ---
if __name__ == "__main__":
    create_ai_prompt_file(path_commercial, path_non_commercial)