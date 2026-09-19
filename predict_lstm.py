

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACT_PATH = Path("outputs/preprocessing_and_baselines.joblib")
PANEL_PATH = Path("outputs/monthly_panel.csv")



def load_artifact(path: Path) -> dict:
    if not path.exists():
        sys.exit(
            f"Artifact not found at '{path}'.\n"
            "Run the LSTM notebook's Section 7 save cell first "
            "(the one with joblib.dump(artifact, OUT_DIR / 'preprocessing_and_baselines.joblib'))."
        )
    return joblib.load(path)


def load_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "cell" not in df.columns or "ym" not in df.columns:
        return None
    df["ym"] = pd.to_datetime(df["ym"])
    return df




def engineer_features(monthly_df: pd.DataFrame, min_mag: float, big_mag: float) -> pd.DataFrame:
    """
    monthly_df: columns ['ym', 'count', 'mag_mean', 'mag_max', 'depth_mean'],
    one row per month, in chronological order (oldest first).
    Returns the same dataframe with every engineered feature column added.
    """
    df = monthly_df.sort_values("ym").reset_index(drop=True).copy()

   
    df["energy"] = np.where(
        df["count"] > 0,
        df["count"] * (10 ** (1.5 * df["mag_mean"] + 4.8)),
        0.0,
    )
    df["log_count"] = np.log1p(df["count"])
    df["log_energy"] = np.log1p(df["energy"])

    for w in (3, 6, 12):
        df[f"count_ma{w}"] = df["count"].rolling(w, min_periods=1).mean()
        df[f"energy_ma{w}"] = df["log_energy"].rolling(w, min_periods=1).mean()

    df["count_std12"] = df["count"].rolling(12, min_periods=1).std().fillna(0)
    df["magmax_ma6"] = df["mag_max"].rolling(6, min_periods=1).max()

    df["mag_sum"] = df["mag_mean"] * df["count"]
    roll_magsum = df["mag_sum"].rolling(24, min_periods=1).sum()
    roll_n = df["count"].rolling(24, min_periods=1).sum()
    mean_mag_24 = roll_magsum / roll_n.replace(0, np.nan)
    df["b_value"] = (np.log10(np.e) / (mean_mag_24 - (min_mag - 0.05))).clip(0, 3).fillna(1.0)

    df["big_flag"] = (df["mag_max"] >= big_mag).astype(int)

    def months_since(flags):
        out, last = np.empty(len(flags)), -1
        for i, v in enumerate(flags):
            out[i] = (i - last) if last >= 0 else 120
            if v:
                last = i
        return np.minimum(out, 120)

    df["months_since_big"] = months_since(df["big_flag"].values)

    mo = df["ym"].dt.month
    df["sin_month"] = np.sin(2 * np.pi * mo / 12)
    df["cos_month"] = np.cos(2 * np.pi * mo / 12)
    return df


def predict_from_features(artifact, feature_df: pd.DataFrame):
    features = artifact["features"]
    seq_len = artifact["seq_len"]

    if len(feature_df) < seq_len:
        sys.exit(f"Need at least {seq_len} months of data, only got {len(feature_df)}.")

    window = feature_df.iloc[-seq_len:]
    missing = [f for f in features if f not in window.columns]
    if missing:
        sys.exit(f"Internal error: missing engineered feature(s) {missing}")

    X_seq = window[features].to_numpy(dtype=np.float32)   
    X_scaled = artifact["scaler"].transform(X_seq)        
    X_flat = X_scaled.reshape(1, -1)                      

    pred_log_count = artifact["gbm_regressor"].predict(X_flat)[0]
    pred_count = max(float(np.expm1(pred_log_count)), 0.0)
    prob_big = float(artifact["gbm_classifier"].predict_proba(X_flat)[0, 1])
    return pred_count, prob_big


def print_forecast(label, last_month, artifact, pred_count, prob_big):
    forecast_month = (last_month + pd.offsets.MonthBegin(1)).strftime("%Y-%m")
    print(f"\n{'='*50}")
    print(f"Region: {label}")
    print(f"History used: {artifact['seq_len']} months ending {last_month.strftime('%Y-%m')}")
    print(f"Forecast for: {forecast_month}")
    print(f"  Predicted event count (M >= {artifact['min_mag']}): {pred_count:.2f}")
    print(f"  P(at least one M >= {artifact['big_mag']} event):   {prob_big:.1%}")
    print(f"{'='*50}")
    print(
        "\nNote: this is a statistical baseline forecast, not a deterministic prediction. "
        "Treat it as an estimate of relative risk, not a guaranteed outcome -- see the "
        "notebook's baseline comparison (Section 6) for how much this actually beats "
        "just using the region's historical average."
    )



def run_panel_lookup(panel_df, artifact, cell, as_of=None):
    if panel_df is None:
        sys.exit(f"No usable panel found at '{PANEL_PATH}'. Use interactive manual entry instead.")
    sub = panel_df[panel_df["cell"] == cell].sort_values("ym")
    if sub.empty:
        available = sorted(panel_df["cell"].unique())
        sys.exit(
            f"No data for cell '{cell}'.\n"
            f"Try --list-cells to see all {len(available)} available ids, "
            f"e.g.: {', '.join(available[:10])}"
        )
    if as_of is not None:
        sub = sub[sub["ym"] <= pd.Timestamp(as_of)]

    seq_len = artifact["seq_len"]
    if len(sub) < seq_len:
        sys.exit(f"Only {len(sub)} month(s) on file for cell '{cell}', need {seq_len}.")

    pred_count, prob_big = predict_from_features(artifact, sub)
    print_forecast(f"cell {cell}", sub["ym"].iloc[-1], artifact, pred_count, prob_big)



def prompt_int(msg, default=None, minimum=0):
    while True:
        raw = input(msg).strip()
        if raw == "" and default is not None:
            return default
        try:
            val = int(raw)
            if val < minimum:
                print(f"  please enter a number >= {minimum}.")
                continue
            return val
        except ValueError:
            print("  please enter a whole number.")


def prompt_float(msg, default=None):
    while True:
        raw = input(msg).strip()
        if raw == "" and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            print("  please enter a number.")


def prompt_month(msg):
    while True:
        raw = input(msg).strip()
        try:
            return pd.Timestamp(raw + "-01")
        except (ValueError, TypeError):
            print("  please enter a month as YYYY-MM, e.g. 2026-09")


def collect_manual_months(seq_len, min_mag):
    print(f"\nEnter monthly earthquake activity for the region you want to forecast.")
    print(f"You'll need at least {seq_len} consecutive months, oldest reported first.")
    print(f"(Counts should be for events of magnitude >= {min_mag}; if a month had none, enter 0 "
          f"and press Enter through the magnitude/depth prompts to skip them.)\n")

    last_month = prompt_month("Most recent month you have data for (YYYY-MM): ")
    n_months = prompt_int(
        f"How many consecutive months back do you want to enter? [default {seq_len}]: ",
        default=seq_len, minimum=seq_len,
    )
    months = pd.date_range(end=last_month, periods=n_months, freq="MS")

    rows = []
    for m in months:
        print(f"\n-- {m.strftime('%Y-%m')} --")
        count = prompt_int(f"  Number of M>={min_mag} events this month [0]: ", default=0)
        if count > 0:
            mag_mean = prompt_float("  Mean magnitude this month: ")
            mag_max = prompt_float("  Max magnitude this month: ")
            depth_mean = prompt_float("  Mean depth this month (km) [10]: ", default=10.0)
        else:
            mag_mean, mag_max, depth_mean = 0.0, 0.0, 10.0
        rows.append({"ym": m, "count": count, "mag_mean": mag_mean,
                     "mag_max": mag_max, "depth_mean": depth_mean})
    return pd.DataFrame(rows)


def run_interactive(panel_df, artifact):
    print("Earthquake seismicity forecaster")
    print("---------------------------------")
    print("1) Look up a region already in the historical panel (fast)")
    print("2) Enter your own recent monthly earthquake data (e.g. from USGS)")
    choice = input("Choose 1 or 2: ").strip()

    if choice == "1":
        if panel_df is None:
            print(f"\nNo panel file found at '{PANEL_PATH}' -- switching to manual entry.\n")
        else:
            cells = sorted(panel_df["cell"].unique())
            print(f"\n{len(cells)} region(s) available, e.g.: {', '.join(cells[:15])}")
            cell = input("Enter a cell id from that list: ").strip()
            run_panel_lookup(panel_df, artifact, cell)
            return

    label = input("\nLabel for this region (e.g. 'Off the coast of Japan'): ").strip() or "your region"
    monthly_df = collect_manual_months(artifact["seq_len"], artifact["min_mag"])
    feature_df = engineer_features(monthly_df, artifact["min_mag"], artifact["big_mag"])
    pred_count, prob_big = predict_from_features(artifact, feature_df)
    print_forecast(label, monthly_df["ym"].iloc[-1], artifact, pred_count, prob_big)


# --------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Forecast next-month seismicity using the saved LSTM-notebook baseline."
    )
    p.add_argument("--cell", help="Grid cell id, e.g. '35_140' - quick lookup against the historical panel.")
    p.add_argument("--as-of", default=None,
                   help="Optional YYYY-MM-DD, used with --cell. Forecast as of this month instead of the latest.")
    p.add_argument("--artifact-path", type=Path, default=ARTIFACT_PATH)
    p.add_argument("--panel-path", type=Path, default=PANEL_PATH)
    p.add_argument("--list-cells", action="store_true", help="Print available cell ids and exit.")
    return p


def main():
    args = build_parser().parse_args()
    panel_df = load_panel(args.panel_path)

    if args.list_cells:
        if panel_df is None:
            sys.exit(f"No usable panel found at '{args.panel_path}'.")
        cells = sorted(panel_df["cell"].unique())
        print(f"{len(cells)} cell(s) available:")
        print(", ".join(cells))
        return

    artifact = load_artifact(args.artifact_path)

    if args.cell:
        run_panel_lookup(panel_df, artifact, args.cell, args.as_of)
        return

 
    run_interactive(panel_df, artifact)


if __name__ == "__main__":
    main()
