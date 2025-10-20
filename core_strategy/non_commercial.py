# -*- coding: utf-8 -*-
"""
비상권 업종별 상위권 점포 분석
 - 순이익지수(ProfitIndex) 계산
 - 업종별 상위군(TOP_BY_PROFIT) 라벨링
 - 주요 특징(효과크기, 가중치) 도출
"""

import os, re, numpy as np, pandas as pd
from pathlib import Path

# ---------------------------
# 1. 경로 설정
# ---------------------------
# CSV 파일 경로를 본인 환경에 맞게 수정하세요
INPUT = r"C:\Users\신승민\OneDrive\바탕 화면\비상권 데이터.csv"

# 결과 저장은 현재 실행 중인 .py 파일과 같은 폴더
OUTDIR = Path(__file__).parent

IND_COL = "업종_정규화2_대분류"
RC_COL  = "RC_M1_SAA"

# ---------------------------
# 2. CSV 로드 함수
# ---------------------------
def read_csv_robust(path_like):
    for enc in ["utf-8-sig","utf-8","cp949","euc-kr","latin1"]:
        try:
            df = pd.read_csv(path_like, encoding=enc, low_memory=False)
            print(f"[INFO] CSV 로드 성공 (encoding={enc})")
            return df
        except Exception:
            continue
    raise RuntimeError(f"[ERROR] CSV 로드 실패: {path_like}")

# ---------------------------
# 3. 매출구간 → 순위(1~6)
# ---------------------------
def rc_to_ord(s):
    if pd.isna(s): return np.nan
    s = str(s)
    if s.startswith("1_"): return 1
    if s.startswith("2_"): return 2
    if s.startswith("3_"): return 3
    if s.startswith("4_"): return 4
    if s.startswith("5_"): return 5
    if s.startswith("6_"): return 6
    m = re.match(r"(\d+)", s)
    return int(m.group(1)) if m else np.nan

# ---------------------------
# 4. 순이익지수(ProfitIndex) 계산
# ---------------------------
def compute_profit_index(df):
    df["RC_M1_SAA_ORD"] = df[RC_COL].map(rc_to_ord)
    df["HIGH_SALES_GRP"] = df["RC_M1_SAA_ORD"].apply(lambda x: 1 if pd.notna(x) and x<=2 else 0)

    # numeric features (exclude id/date/label cols)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {"ARE_D","MCT_ME_D","TA_YM","RC_M1_SAA_ORD","HIGH_SALES_GRP","PROFIT_INDEX","TOP_BY_PROFIT"}
    feats = [c for c in num_cols if c not in exclude]

    # sentinel 처리
    for c in feats:
        s = df[c]
        if s.isin([-999999.9, -999999]).any():
            df.loc[s.isin([-999999.9, -999999]), c] = np.nan

    # z-score 변환
    Z = {}
    for c in feats:
        x = df[c].astype(float)
        med = x.median()
        x = x.fillna(med)
        std = x.std(ddof=0)
        Z[c] = (x - x.mean()) / (std if std>0 else 1.0)
    Z = pd.DataFrame(Z)

    # point-biserial correlation
    y = df["HIGH_SALES_GRP"].fillna(0).astype(float).values
    w = {}
    for c in Z.columns:
        x = Z[c].values
        if np.all(np.isnan(x)): 
            w[c] = 0.0
            continue
        num = np.nansum((x - np.nanmean(x)) * (y - np.nanmean(y)))
        den = np.sqrt(np.nansum((x - np.nanmean(x))**2) * np.nansum((y - np.nanmean(y))**2))
        w[c] = (num/den) if den>0 else 0.0
    w = pd.Series(w).fillna(0.0)
    w = w[w.abs()>=0.05]
    w = w / (w.abs().sum() if w.abs().sum()!=0 else 1.0)

    # profit index
    pr = Z[w.index].dot(w.values)
    pmin, pmax = float(pr.min()), float(pr.max())
    df["PROFIT_INDEX"] = (pr - pmin) / (pmax - pmin + 1e-9) * 100.0

    return df, w.sort_values(ascending=False)

# ---------------------------
# 5. 업종별 상위군 라벨링
# ---------------------------
def label_top_by_profit(df):
    q = df.groupby(IND_COL)["PROFIT_INDEX"].quantile(0.75).to_dict()
    df["TOP_BY_PROFIT"] = df.apply(
        lambda r: 1 if r["PROFIT_INDEX"] >= q.get(str(r[IND_COL]) if pd.notna(r[IND_COL]) else "미지정", np.inf) else 0,
        axis=1
    )
    return df

# ---------------------------
# 6. 효과크기 계산 (Cohen's d)
# ---------------------------
def cohens_d(a, b):
    a = np.array(a, dtype=float); b = np.array(b, dtype=float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a)<5 or len(b)<5: return np.nan
    mu1, mu2 = a.mean(), b.mean()
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1-1)*s1*s1 + (n2-1)*s2*s2)/(n1+n2-2)) if (n1+n2-2)>0 else np.nan
    return (mu1-mu2)/sp if sp and sp>0 else np.nan

# ---------------------------
# 7. 업종별 인사이트 요약
# ---------------------------
def summarize_insights(df, used_feats):
    rows = []
    for g, gdf in df.groupby(IND_COL):
        top = gdf[gdf["TOP_BY_PROFIT"]==1]
        rest = gdf[gdf["TOP_BY_PROFIT"]==0]
        if len(top)<15 or len(rest)<15: 
            continue
        stats = []
        for c in used_feats.index:
            d = cohens_d(top[c], rest[c])
            stats.append((c, d, used_feats[c]))
        stats = sorted(stats, key=lambda x: 0 if pd.isna(x[1]) else abs(x[1]), reverse=True)[:8]
        for c, d, w in stats:
            rows.append({"업종": g, "특징변수": c, "Cohen_d(상위-나머지)": float(d) if pd.notna(d) else np.nan,
                         "가중치(ProfitIndex)": float(w)})
    return pd.DataFrame(rows)

# ---------------------------
# MAIN 실행
# ---------------------------
if __name__ == "__main__":
    df = read_csv_robust(INPUT)
    df.columns = [c.strip() for c in df.columns]
    assert IND_COL in df.columns and RC_COL in df.columns, "필수 컬럼 누락"

    df, weights = compute_profit_index(df)
    df = label_top_by_profit(df)

    # 업종 요약
    ind_summary = df.groupby(IND_COL).agg(
        n=("PROFIT_INDEX","size"),
        평균지수=("PROFIT_INDEX","mean"),
        상위25컷=("PROFIT_INDEX", lambda x: np.quantile(x, 0.75)),
        상위비율=("TOP_BY_PROFIT","mean"),
    ).reset_index().sort_values("평균지수", ascending=False)

    insights = summarize_insights(df, weights)

    # 저장
    df.to_csv(OUTDIR / "비상권_데이터_with_profitindex.csv", index=False, encoding="utf-8-sig")
    weights.to_csv(OUTDIR / "ProfitIndex_가중치.csv", index=True, header=["weight"], encoding="utf-8-sig")
    ind_summary.to_csv(OUTDIR / "업종별_ProfitIndex_요약.csv", index=False, encoding="utf-8-sig")
    insights.to_csv(OUTDIR / "업종별_상위군_공통특징.csv", index=False, encoding="utf-8-sig")

    print("[DONE]")
    print("사용 변수 개수:", len(weights))
    print("주요 가중치:")
    print(weights.head(12).to_string())
