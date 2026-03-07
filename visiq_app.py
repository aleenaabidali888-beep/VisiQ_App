"""
VisiQ — AI Data Intelligence (Python Edition)
=============================================
Anomaly detection · NaN/missing value identification · Data quality scoring
Supports: CSV, JSON, Excel (.xlsx/.xls), TSV

Usage:
    pip install pandas numpy openpyxl rich
    python visiq_app.py                         # interactive menu
    python visiq_app.py --file data.csv         # analyze a file directly
    python visiq_app.py --file data.csv --label Name --value Sales
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

console = Console() if HAS_DEPS else None

# ── MATH HELPERS ─────────────────────────────────────────────────────────────

def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0

def _std(vals):
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))

def _median(vals):
    s = sorted(vals)
    n = len(s)
    return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2) if s else 0.0

def _linreg(xs, ys):
    """Simple linear regression — returns slope, intercept, r2."""
    n = len(xs)
    if n < 2:
        return 0.0, _mean(ys), 0.0
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0, _mean(ys), 0.0
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    y_mean = sy / n
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, intercept, max(0.0, r2)

def _fmt(v, decimals=2):
    try:
        f = float(v)
        if abs(f) >= 1_000_000:
            return f"{f/1_000_000:.1f}M"
        if abs(f) >= 1_000:
            return f"{f/1_000:.1f}K"
        return f"{f:.{decimals}f}"
    except Exception:
        return str(v)

# ── FILE LOADING ──────────────────────────────────────────────────────────────

def load_file(path: str) -> pd.DataFrame:
    """Load CSV / Excel / JSON / TSV into a DataFrame."""
    ext = Path(path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext == ".json":
        with open(path) as f:
            raw = json.load(f)
        if isinstance(raw, list):
            df = pd.DataFrame(raw)
        elif isinstance(raw, dict):
            # find first list value
            key = next((k for k, v in raw.items() if isinstance(v, list)), None)
            if key:
                df = pd.DataFrame(raw[key])
            else:
                df = pd.DataFrame([raw])
        else:
            raise ValueError("JSON must be an array or object containing an array.")
    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\t")
    else:
        # Try comma first, then auto-detect
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, sep=None, engine="python")
    return df

# ── MISSING VALUE ANALYSIS ────────────────────────────────────────────────────

def analyze_missing(df: pd.DataFrame) -> dict:
    """
    Return a full missing-value report:
    - total_cells, empty_cells, missing_pct
    - per_column breakdown: {col: {count, pct, examples_of_nan_rows}}
    - rows_with_any_missing: count and index list
    """
    total_cells = df.size
    empty_cells = int(df.isnull().sum().sum())
    # Also count literal empty-string cells (common in CSV)
    str_empty = int((df == "").sum().sum())
    total_missing = empty_cells + str_empty
    missing_pct = round(total_missing / max(1, total_cells) * 100, 2)

    per_col = {}
    for col in df.columns:
        null_mask = df[col].isnull() | (df[col].astype(str).str.strip() == "")
        cnt = int(null_mask.sum())
        if cnt > 0:
            pct = round(cnt / len(df) * 100, 1)
            row_idxs = list(df.index[null_mask][:10])  # first 10 row indices
            per_col[col] = {"count": cnt, "pct": pct, "row_indices": row_idxs}

    rows_with_any = df[df.isnull().any(axis=1) | (df == "").any(axis=1)]
    return {
        "total_cells": total_cells,
        "total_missing": total_missing,
        "missing_pct": missing_pct,
        "per_column": per_col,
        "rows_with_any_missing": {
            "count": len(rows_with_any),
            "indices": list(rows_with_any.index[:20]),
        },
    }

# ── NaN / UNPARSEABLE ROW ANALYSIS ───────────────────────────────────────────

def analyze_nan_rows(df: pd.DataFrame, value_col: str) -> dict:
    """
    For a specific value column, find rows that cannot be parsed as numbers
    (i.e. NaN rows when coerced to numeric).
    Returns count, pct, and a list of {row_index, raw_value} for each bad row.
    """
    series = pd.to_numeric(df[value_col].astype(str).str.replace(r"[$,%\s]", "", regex=True), errors="coerce")
    nan_mask = series.isnull()
    nan_count = int(nan_mask.sum())
    nan_pct = round(nan_count / max(1, len(df)) * 100, 1)
    bad_rows = []
    for idx in df.index[nan_mask][:20]:
        bad_rows.append({"row_index": int(idx), "raw_value": str(df.loc[idx, value_col])})
    return {
        "nan_count": nan_count,
        "nan_pct": nan_pct,
        "total_rows": len(df),
        "bad_rows": bad_rows,
    }

# ── ANOMALY DETECTION ─────────────────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame, value_col: str, label_col: str = None, z_thresh: float = 2.0) -> dict:
    """
    Z-score anomaly detection on the value column.
    Returns anomaly rows with label, value, z-score, and direction.
    """
    series = pd.to_numeric(
        df[value_col].astype(str).str.replace(r"[$,%\s]", "", regex=True),
        errors="coerce"
    ).dropna()

    if len(series) < 3:
        return {"anomalies": [], "anomaly_count": 0, "anomaly_pct": 0.0,
                "mean": 0.0, "std": 0.0, "z_threshold": z_thresh}

    mu = float(series.mean())
    sigma = float(series.std()) or 1.0
    z_scores = (series - mu) / sigma
    anomaly_mask = z_scores.abs() > z_thresh

    anomalies = []
    for idx in series.index[anomaly_mask]:
        val = float(series.loc[idx])
        z = float(z_scores.loc[idx])
        label = str(df.loc[idx, label_col]) if label_col and label_col in df.columns else f"Row {idx}"
        anomalies.append({
            "row_index": int(idx),
            "label": label,
            "value": round(val, 4),
            "z_score": round(z, 3),
            "direction": "HIGH" if z > 0 else "LOW",
        })

    anomaly_count = len(anomalies)
    anomaly_pct = round(anomaly_count / max(1, len(series)) * 100, 1)

    return {
        "anomalies": sorted(anomalies, key=lambda x: abs(x["z_score"]), reverse=True),
        "anomaly_count": anomaly_count,
        "anomaly_pct": anomaly_pct,
        "mean": round(mu, 4),
        "std": round(sigma, 4),
        "z_threshold": z_thresh,
    }

# ── DESCRIPTIVE STATS ─────────────────────────────────────────────────────────

def descriptive_stats(df: pd.DataFrame, value_col: str) -> dict:
    series = pd.to_numeric(
        df[value_col].astype(str).str.replace(r"[$,%\s]", "", regex=True),
        errors="coerce"
    ).dropna()
    if series.empty:
        return {}
    vals = series.tolist()
    xs = list(range(len(vals)))
    slope, intercept, r2 = _linreg(xs, vals)
    n = len(vals)
    mae = sum(abs(v - (slope * i + intercept)) for i, v in enumerate(vals)) / n
    rmse = math.sqrt(sum((v - (slope * i + intercept)) ** 2 for i, v in enumerate(vals)) / n)
    mape = sum(abs((v - (slope * i + intercept)) / v) for i, v in enumerate(vals) if v != 0) / n * 100
    forecast_acc = max(0.0, round(min(75.0, 100 - mape), 1))
    return {
        "count": n,
        "total": round(float(series.sum()), 4),
        "mean": round(float(series.mean()), 4),
        "median": round(float(series.median()), 4),
        "std": round(float(series.std()), 4),
        "min": round(float(series.min()), 4),
        "max": round(float(series.max()), 4),
        "cv_pct": round(float(series.std() / series.mean() * 100) if series.mean() != 0 else 0, 1),
        "r2": round(r2, 4),
        "slope": round(slope, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 1),
        "forecast_accuracy_pct": forecast_acc,
        "forecast_next_3": [round(slope * (n + i) + intercept, 2) for i in range(1, 4)],
    }

# ── DATA QUALITY SCORE ────────────────────────────────────────────────────────

def data_quality_score(missing_report: dict, nan_report: dict, anomaly_report: dict, total_rows: int) -> dict:
    nan_pct = nan_report["nan_pct"]
    anomaly_pct = anomaly_report["anomaly_pct"]
    missing_pct = missing_report["missing_pct"]
    cv = 0.0  # placeholder — computed separately if needed

    completeness = max(0.0, 100 - nan_pct)
    consistency  = max(0.0, 100 - anomaly_pct * 3)
    data_fill    = max(0.0, 100 - missing_pct)
    score = round(completeness * 0.4 + consistency * 0.35 + data_fill * 0.25, 1)

    if score >= 90:
        grade, verdict = "Excellent", "Very clean data. Low noise, no significant anomalies."
    elif score >= 75:
        grade, verdict = "Good", "Good quality with minor issues."
    elif score >= 55:
        grade, verdict = "Fair", "Notable errors or outliers present — review recommended."
    else:
        grade, verdict = "Poor", "Significant errors, NaNs, or anomalies. Data needs cleaning."

    return {"score": score, "grade": grade, "verdict": verdict,
            "completeness_pct": round(completeness, 1),
            "consistency_pct": round(consistency, 1),
            "data_fill_pct": round(data_fill, 1)}

# ── RICH DISPLAY ─────────────────────────────────────────────────────────────

def _color_pct(pct: float, reverse=False) -> str:
    """Return rich color tag based on severity."""
    if reverse:
        return "green" if pct >= 90 else "yellow" if pct >= 55 else "red"
    return "green" if pct < 5 else "yellow" if pct < 20 else "red"

def print_report(df, label_col, value_col, missing, nan_rep, anomalies, stats, quality):
    console.print()
    console.print(Panel.fit(
        f"[bold white]VisiQ[/bold white] [dim]— AI Data Intelligence (Python)[/dim]\n"
        f"[dim]File: {getattr(df, '_source', 'DataFrame')} · {len(df)} rows · {len(df.columns)} cols[/dim]",
        border_style="blue"
    ))

    # ── DESCRIPTIVE STATS ──
    if stats:
        t = Table(title="📊 Descriptive Statistics", box=box.SIMPLE_HEAD, show_header=True)
        t.add_column("Metric", style="bold")
        t.add_column("Value", justify="right")
        rows_s = [
            ("Count", str(stats["count"])),
            ("Total", _fmt(stats["total"])),
            ("Mean", _fmt(stats["mean"])),
            ("Median", _fmt(stats["median"])),
            ("Std Dev", _fmt(stats["std"])),
            ("Min", _fmt(stats["min"])),
            ("Max", _fmt(stats["max"])),
            ("Coeff of Variation", f"{stats['cv_pct']}%"),
        ]
        for k, v in rows_s:
            t.add_row(k, v)
        console.print(t)

    # ── NaN / UNPARSEABLE ROWS ──
    console.print()
    nan_color = _color_pct(nan_rep["nan_pct"])
    console.print(Panel(
        f"[bold]NaN / Unparseable Rows in [cyan]'{value_col}'[/cyan][/bold]\n"
        f"  [{nan_color}]{nan_rep['nan_count']} rows ({nan_rep['nan_pct']}%)[/{nan_color}] "
        f"could not be parsed as numbers out of {nan_rep['total_rows']} total rows.\n"
        + (("\n  Bad rows (first 20):\n" +
           "\n".join(f"  · Row {r['row_index']}: raw value = [red]'{r['raw_value']}'[/red]"
                     for r in nan_rep["bad_rows"])) if nan_rep["bad_rows"] else "  ✅ All rows parsed successfully."),
        title="🚫 NaN Detection", border_style=nan_color
    ))

    # ── MISSING VALUES ──
    console.print()
    miss_color = _color_pct(missing["missing_pct"])
    miss_lines = [
        f"  [{miss_color}]{missing['total_missing']} missing cells ({missing['missing_pct']}%)[/{miss_color}] "
        f"across {missing['total_cells']} total cells.",
        f"  Rows with any missing value: [bold]{missing['rows_with_any_missing']['count']}[/bold]"
    ]
    if missing["per_column"]:
        miss_lines.append("\n  Missing by column:")
        for col, info in missing["per_column"].items():
            miss_lines.append(
                f"  · [cyan]{col}[/cyan]: [red]{info['count']}[/red] missing ({info['pct']}%) "
                f"— row indices: {info['row_indices'][:5]}" + ("…" if len(info['row_indices']) > 5 else "")
            )
    else:
        miss_lines.append("  ✅ No missing values found.")
    console.print(Panel(
        "\n".join(miss_lines),
        title="❌ Missing Value Identification", border_style=miss_color
    ))

    # ── ANOMALY DETECTION ──
    console.print()
    anom_color = _color_pct(anomalies["anomaly_pct"])
    anom_lines = [
        f"  [{anom_color}]{anomalies['anomaly_count']} anomalies ({anomalies['anomaly_pct']}%)[/{anom_color}] "
        f"detected (|z| > {anomalies['z_threshold']}σ) in [cyan]'{value_col}'[/cyan].",
        f"  Mean = [bold]{anomalies['mean']}[/bold], Std Dev = [bold]{anomalies['std']}[/bold]",
    ]
    if anomalies["anomalies"]:
        anom_lines.append("\n  Anomaly rows (sorted by |z-score|):")
        for a in anomalies["anomalies"][:15]:
            direction_color = "red" if a["direction"] == "HIGH" else "blue"
            anom_lines.append(
                f"  · Row {a['row_index']} | [bold]{a['label']}[/bold] = "
                f"[{direction_color}]{a['value']}[/{direction_color}] "
                f"(z = {a['z_score']}σ, [{direction_color}]{a['direction']}[/{direction_color}])"
            )
        if len(anomalies["anomalies"]) > 15:
            anom_lines.append(f"  … and {len(anomalies['anomalies']) - 15} more")
    else:
        anom_lines.append("  ✅ No anomalies detected. All values within normal range.")
    console.print(Panel(
        "\n".join(anom_lines),
        title="⚠️  Anomaly Detection (Z-Score > 2σ)", border_style=anom_color
    ))

    # ── MODEL ACCURACY ──
    if stats:
        console.print()
        r2_color = "green" if stats["r2"] >= 0.75 else "yellow" if stats["r2"] >= 0.5 else "red"
        acc_color = "green" if stats["forecast_accuracy_pct"] >= 65 else "yellow" if stats["forecast_accuracy_pct"] >= 45 else "red"
        console.print(Panel(
            f"  R² Score: [{r2_color}]{stats['r2']}[/{r2_color}]  |  "
            f"MAE: [bold]{_fmt(stats['mae'])}[/bold]  |  "
            f"RMSE: [bold]{_fmt(stats['rmse'])}[/bold]  |  "
            f"MAPE: [bold]{stats['mape']}%[/bold]\n"
            f"  Forecast Accuracy: [{acc_color}]{stats['forecast_accuracy_pct']}%[/{acc_color}]\n"
            f"  Next 3 Forecast Values: [cyan]{stats['forecast_next_3']}[/cyan]",
            title="🎯 Model Accuracy & Forecast", border_style=acc_color
        ))

    # ── DATA QUALITY SCORE ──
    console.print()
    q_color = "green" if quality["score"] >= 75 else "yellow" if quality["score"] >= 55 else "red"
    console.print(Panel(
        f"  Overall Score: [{q_color}]{quality['score']}% — {quality['grade']}[/{q_color}]\n"
        f"  {quality['verdict']}\n\n"
        f"  Completeness : [bold]{quality['completeness_pct']}%[/bold]  (based on NaN rows)\n"
        f"  Consistency  : [bold]{quality['consistency_pct']}%[/bold]  (based on anomalies)\n"
        f"  Data Fill    : [bold]{quality['data_fill_pct']}%[/bold]  (based on missing cells)",
        title="✅ Data Quality Score", border_style=q_color
    ))
    console.print()

# ── FALLBACK PLAIN-TEXT REPORT (no rich) ────────────────────────────────────

def print_report_plain(df, label_col, value_col, missing, nan_rep, anomalies, stats, quality):
    sep = "=" * 60
    print(f"\n{sep}")
    print("VisiQ — AI Data Intelligence (Python)")
    print(f"{len(df)} rows · {len(df.columns)} cols")
    print(sep)

    print(f"\n[NaN Detection — '{value_col}']")
    print(f"  NaN/unparseable rows: {nan_rep['nan_count']} ({nan_rep['nan_pct']}%) of {nan_rep['total_rows']}")
    for r in nan_rep["bad_rows"]:
        print(f"  · Row {r['row_index']}: raw = '{r['raw_value']}'")

    print(f"\n[Missing Values]")
    print(f"  Total missing cells: {missing['total_missing']} ({missing['missing_pct']}%)")
    if missing["per_column"]:
        for col, info in missing["per_column"].items():
            print(f"  · {col}: {info['count']} missing ({info['pct']}%) at rows {info['row_indices'][:5]}")
    else:
        print("  No missing values found.")

    print(f"\n[Anomaly Detection (z > {anomalies['z_threshold']})]")
    print(f"  Anomalies: {anomalies['anomaly_count']} ({anomalies['anomaly_pct']}%)")
    for a in anomalies["anomalies"][:15]:
        print(f"  · Row {a['row_index']} | {a['label']} = {a['value']} (z={a['z_score']}, {a['direction']})")

    if stats:
        print(f"\n[Model Accuracy]")
        print(f"  R2={stats['r2']}  MAE={_fmt(stats['mae'])}  RMSE={_fmt(stats['rmse'])}  MAPE={stats['mape']}%")
        print(f"  Forecast Accuracy: {stats['forecast_accuracy_pct']}%")
        print(f"  Next 3 Forecast: {stats['forecast_next_3']}")

    print(f"\n[Data Quality Score]")
    print(f"  Score: {quality['score']}% — {quality['grade']}")
    print(f"  {quality['verdict']}")
    print(sep + "\n")

# ── INTERACTIVE COLUMN SELECTION ─────────────────────────────────────────────

def pick_column(df: pd.DataFrame, prompt: str, prefer_numeric=False) -> str:
    cols = list(df.columns)
    print(f"\n{prompt}")
    for i, c in enumerate(cols):
        dtype = str(df[c].dtype)
        print(f"  [{i}] {c}  ({dtype})")
    while True:
        raw = input("Enter column number or name: ").strip()
        if raw.isdigit() and int(raw) < len(cols):
            return cols[int(raw)]
        if raw in cols:
            return raw
        print("  Invalid choice, try again.")

# ── MAIN ──────────────────────────────────────────────────────────────────────

def run(file_path: str, label_col: str = None, value_col: str = None, z_thresh: float = 2.0):
    if not HAS_DEPS:
        print("ERROR: Required libraries not found.")
        print("Install them with:  pip install pandas numpy openpyxl rich")
        sys.exit(1)

    console.print(f"\n[bold blue]Loading:[/bold blue] {file_path}")
    df = load_file(file_path)
    df._source = Path(file_path).name  # attach filename for display
    console.print(f"[green]✓ Loaded {len(df)} rows × {len(df.columns)} columns[/green]")

    if label_col is None:
        label_col = pick_column(df, "Select the LABEL column (e.g. names, categories):")
    if value_col is None:
        value_col = pick_column(df, "Select the VALUE column (numeric data to analyze):", prefer_numeric=True)

    console.print(f"\n[dim]Analyzing · label=[cyan]{label_col}[/cyan] · value=[cyan]{value_col}[/cyan] · z-threshold={z_thresh}σ[/dim]")

    missing   = analyze_missing(df)
    nan_rep   = analyze_nan_rows(df, value_col)
    anomalies = detect_anomalies(df, value_col, label_col=label_col, z_thresh=z_thresh)
    stats     = descriptive_stats(df, value_col)
    quality   = data_quality_score(missing, nan_rep, anomalies, len(df))

    print_report(df, label_col, value_col, missing, nan_rep, anomalies, stats, quality)

    # Optionally export JSON report
    export = input("Export full report to JSON? (y/n): ").strip().lower()
    if export == "y":
        out = {
            "file": Path(file_path).name,
            "rows": len(df),
            "columns": len(df.columns),
            "label_col": label_col,
            "value_col": value_col,
            "missing_values": missing,
            "nan_rows": nan_rep,
            "anomalies": anomalies,
            "descriptive_stats": stats,
            "data_quality": quality,
        }
        out_path = Path(file_path).stem + "_visiq_report.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        console.print(f"\n[green]✓ Report saved to:[/green] [bold]{out_path}[/bold]\n")


def main():
    parser = argparse.ArgumentParser(description="VisiQ — AI Data Intelligence (Python)")
    parser.add_argument("--file",  "-f", help="Path to CSV / Excel / JSON / TSV file")
    parser.add_argument("--label", "-l", help="Label column name")
    parser.add_argument("--value", "-v", help="Value column name")
    parser.add_argument("--z",     "-z", type=float, default=2.0, help="Z-score threshold for anomalies (default: 2.0)")
    args = parser.parse_args()

    if not HAS_DEPS:
        print("Missing dependencies. Run:  pip install pandas numpy openpyxl rich")
        sys.exit(1)

    if args.file:
        run(args.file, label_col=args.label, value_col=args.value, z_thresh=args.z)
    else:
        # Interactive mode
        console.print("\n[bold blue]VisiQ — Python Data Intelligence[/bold blue]")
        console.print("[dim]Anomaly detection · NaN identification · Missing value analysis[/dim]\n")
        file_path = input("Enter path to your file (CSV / Excel / JSON / TSV): ").strip().strip('"')
        if not os.path.exists(file_path):
            console.print(f"[red]File not found: {file_path}[/red]")
            sys.exit(1)
        run(file_path, z_thresh=args.z)


if __name__ == "__main__":
    main()
