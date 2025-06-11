import pandas as pd
import re
from pathlib import Path

LOG_FILE = Path("logs/run_history.log")

def parse_log_file_custom(path):
    with open(path, "r") as f:
        blocks = f.read().split("=" * 59)

    data = {}

    for block in blocks:
        if "u_count" not in block:
            continue

        # Extract fields
        u_count = int(re.search(r"u_count:\s+(\d+)", block).group(1)) # type: ignore
        m_count = int(re.search(r"m_count:\s+(\d+)", block).group(1)) # type: ignore
        c_value = float(re.search(r"c_value:\s+([\d.]+)", block).group(1)) # type: ignore
        min_rating = float(re.search(r"min_rating:\s+([\d.]+)", block).group(1)) # type: ignore

        label = f"{u_count} Users, {m_count} Movies, {c_value:.4f} C, MinRating {min_rating:.2f}"

        ll_train = float(re.search(r"Log-Likelihood \(Train\)\s+(-?[\d.]+)", block).group(1)) # type: ignore
        ll_test = float(re.search(r"Log-Likelihood \(Test\)\s+(-?[\d.]+)", block).group(1)) # type: ignore
        pl_train = float(re.search(r"Pseudo-Likelihood \(Train\)\s+(-?[\d.]+)", block).group(1)) # type: ignore
        pl_test = float(re.search(r"Pseudo-Likelihood \(Test\)\s+(-?[\d.]+)", block).group(1)) # type: ignore
        connectivity = float(re.search(r"Average Connectivity\s+([\d.]+)", block).group(1)) # type: ignore
        error_rate = float(re.search(r"Error Rate\s+([\d.]+)", block).group(1)) # type: ignore

        data[label] = [
            ll_train, ll_test, pl_train, pl_test, connectivity, error_rate
        ]

    df = pd.DataFrame.from_dict(
        data,
        orient="index",
        columns=[
            "Independent LL (Train)",
            "Independent LL (Test)",
            "Ising PL (Train)",
            "Ising PL (Test)",
            "Average Connectivity",
            "Error Rate"
        ]
    )

    return df

if __name__ == "__main__":
    df = parse_log_file_custom(LOG_FILE)
    print(df)

    # Create statistics folder next to this script
    STAT_DIR = Path(__file__).parent / "statistics"
    STAT_DIR.mkdir(parents=True, exist_ok=True)

    # Save CSV inside statistics folder
    csv_path = STAT_DIR / "stat_summary.csv"
    df.to_csv(csv_path)
