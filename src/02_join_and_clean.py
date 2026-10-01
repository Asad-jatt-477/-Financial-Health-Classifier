"""
02_join_and_clean.py  (chunked / low-memory version)
-------------------------------------------------------
Pehla version poore 36M rows ko ek sath merge karne ki koshish kar raha
tha, jisse RAM khatam ho gayi (ArrowMemoryError). Ye version har quarter
ko ALAG SE process karta hai — isliye ek waqt mein sirf ~3.6M rows
(ek quarter ka size) RAM mein hoti hain, poore 36M nahi.

Output: data/processed/final_joined/<quarter>.parquet  (10 alag files,
ek har quarter ke liye — inhe "partitioned dataset" kehte hain)
"""

from pathlib import Path
import gc
import pandas as pd

PROCESSED_DIR = Path("data/processed")
OUT_DIR = PROCESSED_DIR / "final_joined"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUB_COLS = ["adsh", "cik", "name", "sic", "form", "period", "fy", "fp", "filed"]
TAG_COLS = ["tag", "version", "datatype", "crdr", "tlabel"]
PRE_COLS = ["adsh", "tag", "version", "stmt", "plabel"]

QUARTERS = [
    "2024q1", "2024q2", "2024q3", "2024q4",
    "2025q1", "2025q2", "2025q3", "2025q4",
    "2026q1", "2026q2",
]


def main():
    # ---- Global reference tables (chhoti hain, ek hi baar load karo) ----
    print("Global lookup tables (sub, tag) load + dedupe kar rahe hain...")

    sub_all = pd.read_parquet(PROCESSED_DIR / "sub_all.parquet", columns=SUB_COLS + ["source_quarter"])
    sub_dedup = sub_all.drop_duplicates(subset=["adsh"], keep="first").drop(columns=["source_quarter"])
    del sub_all
    gc.collect()
    print(f"  sub_dedup: {sub_dedup.shape}")

    tag_all = pd.read_parquet(PROCESSED_DIR / "tag_all.parquet", columns=TAG_COLS + ["source_quarter"])
    tag_dedup = tag_all.drop_duplicates(subset=["tag", "version"], keep="first").drop(columns=["source_quarter"])
    del tag_all
    gc.collect()
    print(f"  tag_dedup: {tag_dedup.shape}")

    total_in = 0
    total_out = 0

    # ---- Har quarter ko ALAG se process karo ----
    for q in QUARTERS:
        print(f"\nProcessing quarter: {q}")

        # Sirf is quarter ki rows load karo (poori num_all/pre_all nahi)
        num_q = pd.read_parquet(
            PROCESSED_DIR / "num_all.parquet",
            filters=[("source_quarter", "=", q)],
        )
        pre_q = pd.read_parquet(
            PROCESSED_DIR / "pre_all.parquet",
            columns=PRE_COLS + ["source_quarter"],
            filters=[("source_quarter", "=", q)],
        )

        print(f"  num_q: {num_q.shape}  |  pre_q: {pre_q.shape}")

        pre_q_dedup = (
            pre_q.drop_duplicates(subset=["adsh", "tag", "version"], keep="first")
            .drop(columns=["source_quarter"])
        )
        del pre_q
        gc.collect()

        num_q["value"] = pd.to_numeric(num_q["value"], errors="coerce").astype("float32")
        total_in += len(num_q)

        # ---- Joins (is quarter ke andar hi) ----
        merged_q = num_q.merge(sub_dedup, on="adsh", how="left", suffixes=("", "_sub"))
        del num_q
        gc.collect()

        merged_q = merged_q.merge(tag_dedup, on=["tag", "version"], how="left", suffixes=("", "_tag"))
        merged_q = merged_q.merge(pre_q_dedup, on=["adsh", "tag", "version"], how="left", suffixes=("", "_pre"))
        del pre_q_dedup
        gc.collect()

        # ---- Date columns clean karo ----
        merged_q["ddate"] = pd.to_datetime(merged_q["ddate"], format="%Y%m%d", errors="coerce")
        merged_q["filed"] = pd.to_datetime(merged_q["filed"], format="%Y%m%d", errors="coerce")
        merged_q["period"] = pd.to_datetime(merged_q["period"], format="%Y%m%d", errors="coerce")
        merged_q["qtrs"] = pd.to_numeric(merged_q["qtrs"], errors="coerce").astype("float32")

        out_path = OUT_DIR / f"{q}.parquet"
        merged_q.to_parquet(out_path, index=False)
        total_out += len(merged_q)
        print(f"  [SAVED] {out_path}  |  shape={merged_q.shape}")

        del merged_q
        gc.collect()

    # ---- Sanity check ----
    print(f"\n--- Sanity Check ---")
    print(f"  Total rows in (num, all quarters): {total_in}")
    print(f"  Total rows out (merged, all quarters): {total_out}")
    if total_in == total_out:
        print("  [OK] Koi row explosion nahi hui.")
    else:
        print("  [WARNING] Row count farq hai — dobara check karna zaroori hai.")

    print(f"\nDone! Final dataset {len(QUARTERS)} files mein save hui hai:")
    print(f"  {OUT_DIR}/2024q1.parquet, {OUT_DIR}/2024q2.parquet, ... {OUT_DIR}/2026q2.parquet")
    print("\nInhe sab ek sath padhne ke liye (agle steps mein):")
    print('  df = pd.read_parquet("data/processed/final_joined/")')


if __name__ == "__main__":
    main()
