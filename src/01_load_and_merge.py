"""
01_load_and_merge.py
---------------------
SEC Financial Statement Data Sets ke multiple quarters (sub/num/tag/pre.txt)
ko load karta hai, har row mein 'source_quarter' column add karta hai,
aur sab quarters ko ek sath stack (concatenate) kar ke parquet files
mein save karta hai.

Expected folder structure:
    data/raw/2024q1/sub.txt, num.txt, tag.txt, pre.txt, readme.htm
    data/raw/2024q2/...
    ...
    data/raw/2026q2/...

Output:
    data/processed/sub_all.parquet
    data/processed/num_all.parquet
    data/processed/tag_all.parquet
    data/processed/pre_all.parquet
"""

from pathlib import Path
import pandas as pd

# ---- 1. Config: apne project structure ke hisaab se paths adjust karo ----
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

QUARTERS = [
    "2024q1", "2024q2", "2024q3", "2024q4",
    "2025q1", "2025q2", "2025q3", "2025q4",
    "2026q1", "2026q2",
]

# Har file ka apna schema hai (dtype=str rakhna safe hai raw stage par,
# taake leading zeros / mixed types corrupt na hon)
FILES_TO_LOAD = ["sub", "num", "tag", "pre"]


def load_quarter_file(quarter: str, file_stub: str) -> pd.DataFrame:
    """Ek quarter ki ek file (e.g. sub.txt) ko read karta hai aur
    'source_quarter' column add karta hai."""
    file_path = RAW_DATA_DIR / quarter / f"{file_stub}.txt"

    if not file_path.exists():
        print(f"  [SKIP] {file_path} nahi mili — check karo path sahi hai ya nahi")
        return pd.DataFrame()

    df = pd.read_csv(file_path, sep="\t", dtype=str, low_memory=False)
    df["source_quarter"] = quarter
    return df


def main():
    # Har file-type (sub, num, tag, pre) ke liye alag se sab quarters collect karo
    for file_stub in FILES_TO_LOAD:
        print(f"\nLoading all quarters for: {file_stub}.txt")
        frames = []

        for quarter in QUARTERS:
            print(f"  -> {quarter}")
            df = load_quarter_file(quarter, file_stub)
            if not df.empty:
                frames.append(df)

        if not frames:
            print(f"  [WARNING] {file_stub} ke liye koi data nahi mila, skip.")
            continue

        combined = pd.concat(frames, ignore_index=True)
        out_path = PROCESSED_DATA_DIR / f"{file_stub}_all.parquet"
        combined.to_parquet(out_path, index=False)

        print(f"  [SAVED] {out_path}  |  shape={combined.shape}")

    print("\nDone! data/processed/ mein 4 combined parquet files ban gayi hain:")
    print("  sub_all.parquet, num_all.parquet, tag_all.parquet, pre_all.parquet")


if __name__ == "__main__":
    main()
