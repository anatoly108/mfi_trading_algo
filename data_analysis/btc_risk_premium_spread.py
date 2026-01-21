from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px


DATA_FILE = Path(__file__).parent / "data" / "vrp_daily_lines_histogram.csv"
NUMERIC_COLUMNS = ["implied_vol_1m_pct", "realized_vol_1m_pct", "spread_bps"]


def extend_spread_through_noise(
    spread_types: pd.Series, max_noise_duration: int = 2
) -> pd.Series:
    """Replace short opposite streaks with surrounding trend to treat them as noise."""
    if spread_types.empty:
        return spread_types

    types = spread_types.to_list()
    run_bounds = []
    start = 0
    for idx in range(1, len(types) + 1):
        if idx == len(types) or types[idx] != types[start]:
            run_bounds.append((start, idx))
            start = idx

    for i, (run_start, run_end) in enumerate(run_bounds):
        run_length = run_end - run_start
        if run_length > max_noise_duration:
            continue

        prev_type = types[run_bounds[i - 1][0]] if i > 0 else None
        next_type = types[run_bounds[i + 1][0]] if i + 1 < len(run_bounds) else None

        target_type = None
        if prev_type is not None and next_type is not None and prev_type == next_type:
            target_type = prev_type
        elif prev_type is None and next_type is not None:
            target_type = next_type
        elif next_type is None and prev_type is not None:
            target_type = prev_type

        if target_type is None:
            continue

        for pos in range(run_start, run_end):
            types[pos] = target_type

    return pd.Series(types, index=spread_types.index)


def load_vrp_data(path: Path) -> pd.DataFrame:
    """Load the VRP CSV and keep rows with valid spread data."""
    df = pd.read_csv(path, parse_dates=["date"])
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("date").set_index("date")
    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].replace(0, np.nan)
    df[NUMERIC_COLUMNS] = df[NUMERIC_COLUMNS].interpolate(
        method="time", limit_direction="both"
    )
    df = df.reset_index()
    df = df.dropna(subset=["spread_bps"]).sort_values("date")
    return df


def compute_spread_durations(df: pd.DataFrame) -> pd.DataFrame:
    """Return duration (in days) of consecutive positive/negative spread streaks."""
    valid = df[df["spread_bps"] != 0].copy()
    if valid.empty:
        return pd.DataFrame(columns=["spread_type", "duration", "start_date"])

    valid["spread_type"] = valid["spread_bps"].gt(0).map(
        {True: "positive", False: "negative"}
    )
    valid["spread_type"] = extend_spread_through_noise(valid["spread_type"])
    valid["group_id"] = (valid["spread_type"] != valid["spread_type"].shift()).cumsum()
    durations = (
        valid.groupby(["spread_type", "group_id"])
        .agg(duration=("date", "count"), start_date=("date", "min"))
        .reset_index()
        .drop(columns="group_id")
    )
    durations = durations[durations["duration"] > 2].reset_index(drop=True)
    return durations


def display_summary(durations: pd.DataFrame) -> None:
    averages = durations.groupby("spread_type")["duration"].mean()
    for spread_type in ("positive", "negative"):
        if spread_type in averages:
            print(f"Average {spread_type} spread duration: {averages[spread_type]:.2f} days")
        else:
            print(f"No {spread_type} spread sequences found.")


def display_top_sequences(durations: pd.DataFrame, limit: int = 5) -> None:
    for spread_type in ("negative", "positive"):
        subset = (
            durations[durations["spread_type"] == spread_type]
            .sort_values("duration", ascending=False)
            .head(limit)
        )
        if subset.empty:
            print(f"No {spread_type} sequences longer than two days.")
            continue

        print(f"Top {len(subset)} {spread_type} periods (>2 days):")
        for _, row in subset.iterrows():
            start = row["start_date"]
            start_display = start.date().isoformat() if pd.notna(start) else "N/A"
            print(f"  start {start_display}, duration {int(row['duration'])} days")


def plot_duration_boxplots(durations: pd.DataFrame) -> None:
    fig = px.box(
        durations,
        x="spread_type",
        y="duration",
        points="all",
        color="spread_type",
        labels={"spread_type": "Spread sign", "duration": "Sequence duration (days)"},
        title="Positive vs Negative VRP Spread Durations",
    )
    fig.update_layout(showlegend=False)
    fig.show()


df = load_vrp_data(DATA_FILE)
durations = compute_spread_durations(df)
if durations.empty:
    print("No spread sequences available for duration analysis.")
else:
    display_summary(durations)
    display_top_sequences(durations)
    plot_duration_boxplots(durations)
