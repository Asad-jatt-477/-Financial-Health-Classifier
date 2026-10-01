"""
03_pivot_and_clean.py  (v3 -- currency fix + DURATION fix)
------------------------------------------------------------
Ye script 02 ki 'final_joined' quarterly files ko wide dataset mein badalti hai
(ek row = ek filing, ek column = ek financial tag).

DO BUGS JO PEHLE VERSIONS MEIN THE (dono ab fix hain):

1) CURRENCY CONTAMINATION
   Foreign filers (20-F, 40-F, 6-K) apni local currency (JPY, KRW, CNY, EUR...) mein
   report karte hain. Pehle 'uom' filter nahi hota tha, isliye impossible values
   (trillions mein Assets) aa gayi thin.
   FIX: pivot se pehle sirf USD / USD-per-share rows.

2) DURATION MIXING (naya fix, raw data ke saboot ke sath)
   SEC ki 'qtrs' column batati hai ke value kitne quarters ki hai:
       qtrs = 0  -> ek tareekh ka snapshot (balance sheet: Assets, Liabilities ...)
       qtrs = 1  -> 3 mahine, 2 -> 6 mahine (YTD), 3 -> 9 mahine (YTD), 4 -> poora saal
   10-K mein income-statement / cash-flow ki values poore SAAL ki hoti hain, 10-Q mein
   ek quarter ki (ya kabhi YTD bhi). Pehle pivot 'first' se koi bhi value utha leta tha,
   isliye ek hi column mein saal bhar aur ek quarter ki values mil gayin (annual rows par
   asset_turnover ~3 guna, growth mechanical -75% / +300%).
   FIX (sirf 14 flow tags par):
       * sirf qtrs in {1, 2, 3, 4} rakho (qtrs=0 ya 5/44/45/99 jaisi ajeeb values hatao)
       * har value ko qtrs se divide karo -> "per-quarter average rate"
       * ek filing+tag ke liye sabse chhoti duration chuno (qtrs=1 mile to woh, yaani
         asli 3-mahine ki value; warna YTD/annual ka per-quarter average)
   Balance-sheet tags (8 tags, jo 100% qtrs=0 hain) bilkul pehle jaise rehte hain.

   ZAROORI LIMITATION: 10-K rows par flows "saal ka average quarter" hain, asli Q4 nahi
   (Q4 = saal - 9 mahine ke liye alag filings jodni paden). Yeh README mein documented hai.

Output:
    data/processed/final_ml_ready_dataset.parquet
    data/processed/03_duration_audit.csv    (kaun si duration kitni baar chuni gayi)
"""

from pathlib import Path
import pandas as pd

PROCESSED_DIR = Path("data/processed")
JOINED_DIR = PROCESSED_DIR / "final_joined"

QUARTERS = [
    "2024q1", "2024q2", "2024q3", "2024q4",
    "2025q1", "2025q2", "2025q3", "2025q4",
    "2026q1", "2026q2",
]

# Balance-sheet (point-in-time) tags: raw data mein 100% qtrs == 0
STOCK_TAGS = [
    "Assets", "AssetsCurrent", "Liabilities", "LiabilitiesCurrent",
    "StockholdersEquity", "CashAndCashEquivalentsAtCarryingValue",
    "InventoryNet", "LongTermDebtNoncurrent",
]
# Income-statement / cash-flow (period) tags: qtrs 1..4
FLOW_TAGS = [
    "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
    "CostOfRevenue", "GrossProfit", "OperatingIncomeLoss", "NetIncomeLoss",
    "EarningsPerShareBasic", "EarningsPerShareDiluted",
    "ResearchAndDevelopmentExpense", "SellingGeneralAndAdministrativeExpense",
    "IncomeTaxExpenseBenefit",
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInInvestingActivities",
    "NetCashProvidedByUsedInFinancingActivities",
]
CORE_TAGS = STOCK_TAGS + FLOW_TAGS
VALID_FLOW_QTRS = [1, 2, 3, 4]

ID_COLS = ["adsh", "cik", "name", "sic", "form", "period", "fy", "fp", "filed", "source_quarter"]

# EPS tags ka unit 'USD/shares' hota hai, baaki sab 'USD'
USD_UNITS = {"USD", "USD/shares"}

ANNUAL_FORMS = {"10-K", "10-K/A", "10-KT"}
QUARTERLY_FORMS = {"10-Q", "10-Q/A", "10-QT"}


def form_group(form: pd.Series) -> pd.Series:
    return form.map(lambda f: "annual (10-K family)" if f in ANNUAL_FORMS
                    else "quarterly (10-Q family)" if f in QUARTERLY_FORMS else "other")


def process_quarter(quarter: str):
    path = JOINED_DIR / f"{quarter}.parquet"
    df = pd.read_parquet(path)

    # ---- Cleaning decision 1: sirf entity-wide, primary-registrant facts ----
    df = df[df["segments"].isna() & df["coreg"].isna()]

    # ---- Cleaning decision 2: sirf 'current period' values ----
    df = df[df["ddate"] == df["period"]]

    # ---- Cleaning decision 3: sirf core/standard tags ----
    df = df[df["tag"].isin(CORE_TAGS)]

    # ---- FIX 1: sirf USD-denominated values (currency contamination) ----
    n_before = len(df)
    df = df[df["uom"].isin(USD_UNITS)]
    if n_before - len(df) > 0:
        print(f"  [CURRENCY FIX] {n_before - len(df)} non-USD rows hatayi")

    if df.empty:
        print(f"  [WARNING] {quarter}: filter ke baad koi rows nahi bachi.")
        return pd.DataFrame(), pd.DataFrame()

    # ---- FIX 2: duration normalisation (sirf flow tags) ----
    is_flow = df["tag"].isin(FLOW_TAGS)
    stock = df[~is_flow]
    flow = df[is_flow]

    n_flow_before = len(flow)
    flow = flow[flow["qtrs"].isin(VALID_FLOW_QTRS)].copy()
    n_invalid_duration = n_flow_before - len(flow)

    flow["fgroup"] = form_group(flow["form"])
    flow["value"] = flow["value"] / flow["qtrs"]          # per-quarter average rate
    flow = (
        flow.sort_values("qtrs", kind="stable")            # sabse chhoti duration pehle
        .drop_duplicates(subset=["adsh", "tag"], keep="first")
    )

    audit = (
        flow.groupby(["fgroup", "qtrs"]).size().rename("n_flow_facts").reset_index()
        .assign(source_quarter=quarter)
    )
    audit_extra = pd.DataFrame([{
        "fgroup": "(dropped: invalid duration on flow tags)", "qtrs": float("nan"),
        "n_flow_facts": n_invalid_duration, "source_quarter": quarter,
    }, {
        "fgroup": "(info: balance-sheet facts with qtrs != 0, kept as before)", "qtrs": float("nan"),
        "n_flow_facts": int((stock["qtrs"] != 0).sum()), "source_quarter": quarter,
    }])
    audit = pd.concat([audit, audit_extra], ignore_index=True)

    df = pd.concat([stock, flow.drop(columns=["fgroup"])], ignore_index=True)

    wide = df.pivot_table(
        index=ID_COLS,
        columns="tag",
        values="value",
        aggfunc="first",
    ).reset_index()

    wide.columns.name = None
    return wide, audit


def main():
    all_quarters, all_audits = [], []

    for q in QUARTERS:
        print(f"Processing (pivot): {q}")
        wide_q, audit_q = process_quarter(q)
        if not wide_q.empty:
            print(f"  -> shape={wide_q.shape}")
            all_quarters.append(wide_q)
            all_audits.append(audit_q)

    print("\nSab quarters ko combine kar rahe hain...")
    final = pd.concat(all_quarters, ignore_index=True)
    print(f"Final shape (concat ke baad): {final.shape}")

    out_path = PROCESSED_DIR / "final_ml_ready_dataset.parquet"
    final.to_parquet(out_path, index=False)
    print(f"\n[SAVED] {out_path}")

    audit = pd.concat(all_audits, ignore_index=True)
    audit_path = PROCESSED_DIR / "03_duration_audit.csv"
    audit.to_csv(audit_path, index=False)
    print(f"[SAVED] {audit_path}")

    # ---- Sanity check 1: currency fix ----
    print("\n--- Sanity Check 1 (currency fix) ---")
    print(f"  Max Assets value: {final['Assets'].max():,.0f}")

    # ---- Sanity check 2: duration fix -- kaun si duration chuni gayi ----
    print("\n--- Sanity Check 2 (duration fix): flow facts, form-type ke hisaab se, chuni gayi qtrs ---")
    chosen = audit[audit["qtrs"].notna()].pivot_table(
        index="fgroup", columns="qtrs", values="n_flow_facts", aggfunc="sum", fill_value=0)
    print(chosen.astype(int).to_string())
    for _, r in audit[audit["qtrs"].isna()].groupby("fgroup")["n_flow_facts"].sum().reset_index().iterrows():
        print(f"  {r['fgroup']}: {int(r['n_flow_facts'])}")

    # ---- Sanity check 3: EK HI COMPANY ke andar annual vs quarterly filings ki revenue ----
    # (alag companies ka mawazna galat hota: 10-K aur 10-Q bharne wali companies ka mix alag hota hai)
    rev = final["Revenues"].fillna(final["RevenueFromContractWithCustomerExcludingAssessedTax"])
    grp = form_group(final["form"])
    t = pd.DataFrame({"cik": final["cik"], "grp": grp, "rev": rev})
    t = t[t["rev"] > 0]
    med = t.groupby(["cik", "grp"])["rev"].median().unstack()
    print("\n--- Sanity Check 3: ek hi company ki annual vs quarterly filings (median revenue ratio) ---")
    if {"annual (10-K family)", "quarterly (10-Q family)"} <= set(med.columns):
        both = med.dropna(subset=["annual (10-K family)", "quarterly (10-Q family)"])
        if len(both):
            r = both["annual (10-K family)"] / both["quarterly (10-Q family)"]
            print(f"  {len(both)} companies mein dono qisam ki filings hain; median(annual/quarterly) = {r.median():.2f}x "
                  f"(25%-75%: {r.quantile(.25):.2f} - {r.quantile(.75):.2f})")
            print("  Fix se pehle ye ~3.8x tha (annual values saal bhar ki thin). Ab ~1 ke qareeb hona chahiye.")
            if r.median() > 2:
                print("  WARNING: ratio abhi bhi bara hai -- duration fix kaam nahi kar raha, audit CSV dekhein.")
        else:
            print("  Koi company aisi nahi jis ki annual aur quarterly dono filings hon (sirf ek quarter chalaya?).")
    else:
        print("  Annual ya quarterly filings maujood nahi.")

    print("\n--- Tag Coverage ---")
    for tag in CORE_TAGS:
        if tag in final.columns:
            print(f"  {tag:55s} {final[tag].notna().mean():6.1%}")

    print(f"\nTotal rows (company-filings): {final.shape[0]}")
    print(f"Unique companies (cik): {final['cik'].nunique()}")


if __name__ == "__main__":
    main()
