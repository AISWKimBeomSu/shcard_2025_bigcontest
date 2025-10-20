## 솔비(SOL-B): AI 소상공인 성장 파트너 챗봇

데이터 기반으로 소상공인의 마케팅/운영 전략을 제안하는 AI 컨설팅 챗봇입니다. Streamlit UI 상에서 가게 ID를 입력하면, 상권·유동인구·매출 관련 지표를 분석하고, 상황별 특화 도구를 호출해 맞춤 리포트를 생성합니다. 이 저장소의 코드를 그대로 클론받아 로컬 환경에서 동일하게 실행할 수 있도록 구성 안내를 제공합니다.

### 주요 특징
- LangGraph 기반 ReAct 에이전트가 질문 의도에 맞는 단일 도구를 선택 실행
- Gemini 모델을 통한 고품질 자연어 전략 리포트 생성
- 유동인구/요일/시간대/고객군 지표를 활용한 데이터 기반 액션 플랜 제안
- Streamlit 채팅 UI

---

## .gitignore 분석 및 주의사항

이 리포지토리의 `.gitignore`에는 다음이 포함됩니다:

- `/.streamlit/`: Streamlit 비밀키 파일(`secrets.toml`) 등은 Git에 올라가지 않습니다. 사용자가 직접 로컬에 생성해야 합니다.
- `/data/*.csv`: `data/` 폴더 내 CSV 파일은 Git에 업로드되지 않습니다. 즉, 데이터셋은 직접 로컬에 준비해야 합니다.
- 그 외 일반적인 캐시/빌드/환경 폴더, OS별 임시파일 등이 무시됩니다.

특히 다음을 반드시 기억하세요:
- `data/` 폴더 자체는 버전에 포함될 수 있으나, 내부의 CSV 파일은 `.gitignore` 규칙(`/data/*.csv`)에 의해 업로드되지 않습니다.
- Streamlit 비밀키는 `.streamlit/secrets.toml`에 저장해야 하며, 이 파일은 Git에 포함되지 않습니다.

---

## 필수 구성 (Configuration)

이 섹션은 실행 성공을 좌우하는 가장 중요한 설정입니다. 두 가지를 반드시 완료하세요.

### A. Gemini API 키 설정

코드에서 다음과 같이 API 키를 사용합니다:
- `streamlit_app.py`는 `st.secrets["GOOGLE_API_KEY"]`를 사용합니다.
- `tools.py`의 LLM 호출(`google.generativeai`)은 `os.getenv("GOOGLE_API_KEY")`를 사용합니다.

아래 두 가지 모두를 설정해 두면 안전합니다.

1) `.streamlit/secrets.toml` 생성
- 프로젝트 루트에 `.streamlit` 폴더를 만들고 그 안에 `secrets.toml` 파일을 생성하세요.
- 내용 예시:

```toml
# 파일: .streamlit/secrets.toml
GOOGLE_API_KEY = "YOUR_API_KEY_HERE"
```

2) 터미널 환경변수도 설정 권장
- `tools.py`는 환경변수를 직접 읽습니다. 아래처럼 실행 전에 환경변수도 설정하세요.

```bash
export GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

macOS/zsh 기준 예시입니다. Windows PowerShell은 `$env:GOOGLE_API_KEY="YOUR_API_KEY_HERE"` 형태를 사용하세요.

### B. 데이터셋 설정

`.gitignore`로 인해 `data/` 폴더 내 CSV는 리포지토리에 포함되지 않습니다. 로컬에서 직접 준비해야 합니다.

1) 프로젝트 루트에 `data/` 폴더를 생성하세요.
2) 아래 정확한 파일명을 갖는 CSV 파일들을 `data/` 폴더에 넣으세요. 인코딩은 코드에서 `utf-8-sig`를 사용합니다.

`streamlit_app.py`의 `load_data()`가 로드하는 전체 목록:
- `전체JOIN_업종정규화_v2.csv`
- `AI상담사_핵심전략_프롬프트.csv`
- `성별연령대별_유동인구.csv`
- `성별연령대별_유동인구_선택영역.csv`
- `요일별_유동인구.csv`
- `요일별_유동인구_선택영역.csv`
- `시간대별_유동인구.csv`
- `시간대별_유동인구_선택영역.csv`
- `성별연령대별_직장인구.csv`

참고:
- 일부 파일은 “선택영역” 버전과 일반 버전을 모두 사용합니다.
- 파일명은 정확히 일치해야 하며, 누락 시 앱이 경고/오류를 표시합니다.

---

## 환경 요구사항

- Python: `>= 3.11` 권장 (`pyproject.toml` 기준)
- OS: macOS, Linux, Windows (Streamlit 호환 환경)

---

## 설치 (Installation)

### 1) 리포지토리 클론

```bash
git clone https://github.com/<your-account>/<your-repo>.git
cd <your-repo>
```

또는 현재 경로 기준:
```bash
cd /Users/beomsu/Cursor/shcard_2025_bigcontest
```

### 2) 가상환경 생성 및 활성화

Python 3.11로 가상환경을 권장합니다.

```bash
# macOS/Linux (zsh/bash)
python3.11 -m venv .venv
source .venv/bin/activate
```

```powershell
# Windows PowerShell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3) 의존성 설치

`requirements.txt`가 제공됩니다.

```bash
pip install -U pip
pip install -r requirements.txt
```

`requirements.txt` 및 `pyproject.toml`에 포함된 주요 라이브러리:
- Streamlit (`streamlit`)
- Gemini SDK (`google-generativeai`)
- Pandas (`pandas`)
- LangChain/Graph (`langchain`, `langchain-core`, `langchain-google-genai`, `langgraph`, `langchain-mcp-adapters`)
- MCP (`mcp`, `fastmcp`)
- Pillow (`pillow`)
- asyncio

버전 충돌 시, 가상환경을 재생성하거나 `pip install --upgrade <pkg>`로 최신화를 권장합니다.

---

## 실행 (Running the App)

1) 필수 구성 완료 확인
- `.streamlit/secrets.toml`에 `GOOGLE_API_KEY`가 설정되어 있는지 확인
- 터미널에서 환경변수 `GOOGLE_API_KEY`도 설정 권장
- `data/` 폴더에 모든 CSV 파일이 존재하는지 확인

2) Streamlit 앱 실행

메인 진입점은 `streamlit_app.py`입니다.

```bash
streamlit run streamlit_app.py
```

3) 브라우저에서 사용
- Streamlit가 출력하는 로컬 URL(일반적으로 `http://localhost:8501`)을 열고,
- 채팅 입력창에 질문과 함께 가게 ID를 입력하세요. 예: `재방문율 분석해줘 (가게 ID: ABC12345)`

---

## 폴더 구조 개요

- `streamlit_app.py`: Streamlit UI, 에이전트 구성, 데이터 로딩
- `tools.py`: 분석 도구 구현(카페 마케팅, 재방문율 개선, 유동인구/점심 회전율 전략 등)과 Gemini 호출
- `assets/`: UI 이미지 리소스
- `data/`: 로컬에 준비할 CSV 데이터 폴더(Tracked 가능하나 내부 CSV는 `.gitignore`로 미추적)
- `.streamlit/`: `secrets.toml` (Git 미추적)

---

## 트러블슈팅

- 데이터 로딩 에러
  - 메시지: “데이터 로딩 중 오류…” 또는 “`data` 폴더에 필요한 파일이 있는지 확인”
  - 조치: `data/` 폴더 생성 여부, 위에 기재한 모든 CSV가 정확한 파일명과 `utf-8-sig` 인코딩인지 확인하세요.

- Gemini 키 관련 에러
  - `tools.py`의 LLM 호출은 `os.getenv("GOOGLE_API_KEY")`를 사용하므로, 터미널 환경변수 설정을 추가하세요.
  - Streamlit 런타임에서만 `st.secrets`가 동작하므로, CLI에서 직접 도구를 호출할 경우에도 환경변수가 필요할 수 있습니다.

- 라이브러리 충돌/미설치
  - `pip install -r requirements.txt` 재실행
  - 가상환경을 재생성하고 Python 3.11 사용을 권장합니다.

---

## 라이선스

상세 내용은 `LICENSE` 파일을 참조하세요.

---

- 에이전트 도구는 `streamlit_app.py`의 의도 분류 가이드에 따라 하나만 실행되며, 리포트의 첫 섹션에 항상 `가맹점 기본 정보` 블록이 포함되도록 설계되어 있습니다.
- 실행에 필요한 모든 구성 단계(비밀키/데이터/의존성/실행 명령어)를 본문에 포함했습니다.