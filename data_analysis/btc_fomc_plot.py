"""
Script to visualize BTC price paths around FOMC meetings.

Adjust the variables at the bottom of the file before running in an
interactive session to change inputs or output paths.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objs as go


FOMC_DECISION_TIME = "18:00"
FOMC_PRESS_CONF_TIME = "18:30"


def load_minute_candles(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Timestamp" not in df.columns:
        raise ValueError("Input CSV must include a 'Timestamp' column with epoch seconds")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s", utc=True)
    df = df.set_index("Timestamp").sort_index()
    return df


def load_daily_candles(minute_df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        minute_df.resample("1D")
        .agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        })
        .dropna(subset=["Open", "High", "Low", "Close"])
    )
    daily.index = daily.index.normalize()
    return daily


def load_fomc_dates(path: Path) -> List[pd.Timestamp]:
    if not path.exists():
        raise FileNotFoundError(f"FOMC dates file not found: {path}")

    dates: List[pd.Timestamp] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        dates.append(pd.to_datetime(line, format="%Y.%m.%d", utc=True).normalize())
    return sorted(dates)


def extract_daily_price_path(
    daily: pd.DataFrame,
    event_date: pd.Timestamp,
    days_before: int,
    days_after: int,
) -> Optional[Tuple[pd.Series, float]]:
    window = pd.date_range(
        event_date - pd.Timedelta(days=days_before),
        event_date + pd.Timedelta(days=days_after),
        freq="1D",
    )
    window_df = daily.reindex(window)
    base_price = window_df.loc[event_date, "Close"]
    if pd.isna(base_price) or base_price == 0:
        return None

    pct_move = (window_df["Close"] / base_price - 1) * 100
    pct_move.index = range(-days_before, days_after + 1)
    return pct_move, float(base_price)


def extract_minute_price_path(
    minute_df: pd.DataFrame,
    event_time: pd.Timestamp,
    minutes_before: int,
    minutes_after: int,
) -> Optional[Tuple[pd.Series, float]]:
    window = pd.date_range(
        event_time - pd.Timedelta(minutes=minutes_before),
        event_time + pd.Timedelta(minutes=minutes_after),
        freq="1min",
    )
    window_df = minute_df.reindex(window)
    base_price = window_df.loc[event_time, "Close"]
    if pd.isna(base_price):
        base_price = minute_df["Close"].asof(event_time)
    if pd.isna(base_price) or base_price == 0:
        return None

    pct_move = (window_df["Close"] / base_price - 1) * 100
    pct_move.index = range(-minutes_before, minutes_after + 1)
    return pct_move, float(base_price)


def build_daily_paths(
    daily: pd.DataFrame,
    events: List[pd.Timestamp],
    days_before: int,
    days_after: int,
) -> Dict[pd.Timestamp, Tuple[pd.Series, float]]:
    paths: Dict[pd.Timestamp, Tuple[pd.Series, float]] = {}
    for event_date in events:
        series = extract_daily_price_path(daily, event_date, days_before, days_after)
        if series is not None:
            paths[event_date] = series
    return paths


def build_minute_paths(
    minute_df: pd.DataFrame,
    events: List[pd.Timestamp],
    minutes_before: int,
    minutes_after: int,
) -> Dict[pd.Timestamp, Tuple[pd.Series, float]]:
    paths: Dict[pd.Timestamp, Tuple[pd.Series, float]] = {}
    for event_time in events:
        series = extract_minute_price_path(
            minute_df, event_time, minutes_before, minutes_after
        )
        if series is not None:
            paths[event_time] = series
    return paths


def select_recent_events(
    event_dates: List[pd.Timestamp], last_n_meetings: int
) -> Tuple[pd.Timestamp, List[pd.Timestamp]]:
    if not event_dates:
        raise ValueError("No events available to select")

    last_n_meetings = min(last_n_meetings, len(event_dates))
    recent_events = event_dates[-last_n_meetings:]
    if len(recent_events) < 2:
        raise ValueError("Need at least two events to build average path")
    return recent_events[-1], recent_events[:-1]


def average_series_from_paths(
    paths: Dict[pd.Timestamp, Tuple[pd.Series, float]],
    events: List[pd.Timestamp],
) -> pd.Series:
    return pd.concat([paths[e][0] for e in events], axis=1).mean(axis=1)


def compute_window_slope(series: pd.Series, start: int, end: int) -> float:
    window = series.loc[start:end].dropna()
    if len(window) < 2:
        return float("nan")
    x = window.index.to_numpy(dtype=float)
    y = window.to_numpy(dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def compute_pre_post_metrics(
    avg_series: pd.Series, minutes_before: int, minutes_after: int
) -> Dict[str, float]:
    pre_slope = compute_window_slope(avg_series, -minutes_before, -1)
    post_slope = compute_window_slope(avg_series, 1, minutes_after)
    delta = pre_slope - post_slope
    return {
        "pre_slope": pre_slope,
        "post_slope": post_slope,
        "delta": delta,
    }


def build_candidate_press_conf_times(
    minute_df: pd.DataFrame,
    minutes_before: int,
    minutes_after: int,
    exclude_dates: List[pd.Timestamp],
    min_coverage: float,
    valid_weekdays: List[int],
) -> List[pd.Timestamp]:
    unique_dates = pd.Index(minute_df.index.normalize().unique()).sort_values()
    exclude = set(exclude_dates)
    candidates: List[pd.Timestamp] = []
    for date in unique_dates:
        if date in exclude:
            continue
        if valid_weekdays and date.dayofweek not in valid_weekdays:
            continue
        event_time = date + pd.Timedelta(hours=18, minutes=30)
        series_tuple = extract_minute_price_path(
            minute_df, event_time, minutes_before, minutes_after
        )
        if series_tuple is None:
            continue
        series, _ = series_tuple
        coverage = float(series.notna().mean())
        if coverage >= min_coverage:
            candidates.append(event_time)
    return candidates


def permutation_test_press_conf(
    minute_df: pd.DataFrame,
    observed_events: List[pd.Timestamp],
    minutes_before: int,
    minutes_after: int,
    n_samples: int,
    seed: int,
    min_coverage: float,
) -> Dict[str, float]:
    observed_paths = build_minute_paths(
        minute_df, observed_events, minutes_before, minutes_after
    )
    avg_series = average_series_from_paths(observed_paths, observed_events)
    observed_metrics = compute_pre_post_metrics(
        avg_series, minutes_before, minutes_after
    )

    valid_weekdays = sorted({dt.dayofweek for dt in observed_events})
    candidates = build_candidate_press_conf_times(
        minute_df,
        minutes_before,
        minutes_after,
        exclude_dates=[dt.normalize() for dt in observed_events],
        min_coverage=min_coverage,
        valid_weekdays=valid_weekdays,
    )
    if len(candidates) < len(observed_events):
        raise ValueError(
            "Not enough candidate timestamps to run permutation test"
        )

    rng = np.random.default_rng(seed)
    metrics: List[Dict[str, float]] = []
    attempts = 0
    max_attempts = n_samples * 30
    while len(metrics) < n_samples and attempts < max_attempts:
        attempts += 1
        sample_times = rng.choice(
            candidates, size=len(observed_events), replace=False
        )
        sample_times = sorted(sample_times)
        sample_paths = build_minute_paths(
            minute_df, sample_times, minutes_before, minutes_after
        )
        if len(sample_paths) < len(observed_events):
            continue
        sample_avg = average_series_from_paths(sample_paths, sample_times)
        sample_metrics = compute_pre_post_metrics(
            sample_avg, minutes_before, minutes_after
        )
        if np.isnan(list(sample_metrics.values())).any():
            continue
        metrics.append(sample_metrics)

    if not metrics:
        raise ValueError("Permutation test failed to generate valid samples")

    metrics_df = pd.DataFrame(metrics)
    return {
        "observed_pre_slope": observed_metrics["pre_slope"],
        "observed_post_slope": observed_metrics["post_slope"],
        "observed_delta": observed_metrics["delta"],
        "samples_used": len(metrics_df),
        "delta_p_two_sided": float(
            (np.sum(np.abs(metrics_df["delta"]) >= abs(observed_metrics["delta"])) + 1)
            / (len(metrics_df) + 1)
        ),
        "delta_p_one_sided": float(
            (np.sum(metrics_df["delta"] >= observed_metrics["delta"]) + 1)
            / (len(metrics_df) + 1)
        )
        if observed_metrics["delta"] >= 0
        else float(
            (np.sum(metrics_df["delta"] <= observed_metrics["delta"]) + 1)
            / (len(metrics_df) + 1)
        ),
        "pre_p_one_sided": float(
            (np.sum(metrics_df["pre_slope"] >= observed_metrics["pre_slope"]) + 1)
            / (len(metrics_df) + 1)
        )
        if observed_metrics["pre_slope"] >= 0
        else float(
            (np.sum(metrics_df["pre_slope"] <= observed_metrics["pre_slope"]) + 1)
            / (len(metrics_df) + 1)
        ),
        "post_p_one_sided": float(
            (np.sum(metrics_df["post_slope"] <= observed_metrics["post_slope"]) + 1)
            / (len(metrics_df) + 1)
        )
        if observed_metrics["post_slope"] <= 0
        else float(
            (np.sum(metrics_df["post_slope"] >= observed_metrics["post_slope"]) + 1)
            / (len(metrics_df) + 1)
        ),
    }


def make_figure(
    paths: Dict[pd.Timestamp, Tuple[pd.Series, float]],
    last_n_meetings: int,
    show_individual: bool,
    title: str,
    xaxis_title: str,
    hover_label: str,
    annotation_label: str,
) -> go.Figure:
    if not paths:
        raise ValueError("No FOMC events with complete data were found")

    event_dates = sorted(paths.keys())
    last_event, previous_events = select_recent_events(event_dates, last_n_meetings)

    x_axis = paths[last_event][0].index
    fig = go.Figure()

    if show_individual:
        for event in previous_events:
            series, base_price = paths[event]
            base_price_text = f"${base_price:,.2f}"
            fig.add_trace(
                go.Scatter(
                    x=x_axis,
                    y=series,
                    mode="lines",
                    name=event.strftime("%Y-%m-%d"),
                    line=dict(color="lightgray", width=1),
                    text=[base_price_text] * len(x_axis),
                    hovertemplate=(
                        "%{x}: %{y:.2f}%<br>"
                        "Base price: %{text}<extra>%{fullData.name}</extra>"
                    ),
                    showlegend=False,
                )
            )

    avg_series = average_series_from_paths(paths, previous_events)
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=avg_series,
            mode="lines",
            name=f"Average of last {len(previous_events)}",
            line=dict(color="#e91e63", width=3),
            hovertemplate="%{x}: %{y:.2f}%<extra>Average</extra>",
        )
    )

    last_series, last_base_price = paths[last_event]
    last_base_price_text = f"${last_base_price:,.2f}"
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=last_series,
            mode="lines",
            name=(
                f"Most recent ({last_event.strftime('%Y-%m-%d')} |"
                f" base {last_base_price_text})"
            ),
            line=dict(color="black", width=3),
            text=[last_base_price_text] * len(x_axis),
            hovertemplate=(
                "%{x}: %{y:.2f}%<br>"
                "Base price: %{text}<extra>Most recent</extra>"
            ),
        )
    )

    min_y = min(avg_series)
    max_y = max(avg_series)

    fig.add_shape(
        type="line",
        x0=0,
        x1=0,
        y0=min_y,
        y1=max_y,
        line=dict(color="#888", width=2, dash="dot"),
    )

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title="Price change (%)",
        template="plotly_white",
        hovermode="x unified",
        xaxis=dict(
            showgrid=True,
            gridcolor="lightgrey",
            spikemode="across",
            spikesnap="cursor",
            spikecolor="grey",
            spikethickness=1,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#f2f2f2",
            zeroline=True,
            zerolinecolor="#e0e0e0",
            dtick=5,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="grey",
            spikethickness=1,
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=40, r=40, t=60, b=60),
        height=900,
    )

    fig.add_annotation(
        x=0,
        y=min_y,
        yanchor="bottom",
        text=annotation_label,
        showarrow=False,
        font=dict(color="#888"),
    )

    return fig


csv_path = Path("data_analysis/data/btcusd_1-min_data.csv")
fomc_path = Path("data_analysis/data/fomc_last_100_dates.txt")
last_n_meetings = 30
show_individual = True

days_before = days_after = 30
minutes_before = minutes_after = 180
output_daily_path = Path("btc_fomc_days_paths.html")
output_minute_path = Path("btc_fomc_minutes_paths.html")
permutation_samples = 2000
permutation_seed = 7
min_coverage = 0.8

minute_df = load_minute_candles(csv_path)
daily = load_daily_candles(minute_df)

fomc_dates = load_fomc_dates(fomc_path)
if len(fomc_dates) < 2:
    raise ValueError("Not enough FOMC dates found to build the plots")

daily_events = [dt for dt in fomc_dates if dt in daily.index]
press_conf_events = [
    dt + pd.Timedelta(hours=18, minutes=30) for dt in fomc_dates
]

daily_paths = build_daily_paths(daily, daily_events, days_before, days_after)
daily_fig = make_figure(
    daily_paths,
    last_n_meetings,
    show_individual,
    title="BTC price path around FOMC decision days",
    xaxis_title="Days relative to FOMC decision day",
    hover_label="Day",
    annotation_label="FOMC decision day (18:00 UTC)",
)
daily_fig.write_html(output_daily_path, include_plotlyjs="cdn")
print(f"Saved daily plot to {output_daily_path.resolve()}")

minute_paths = build_minute_paths(
    minute_df, press_conf_events, minutes_before, minutes_after
)
minute_fig = make_figure(
    minute_paths,
    last_n_meetings,
    show_individual,
    title="BTC price path around FOMC press conference",
    xaxis_title="Minutes relative to 18:30 UTC press conference start",
    hover_label="Minute",
    annotation_label="Press conference start (18:30 UTC)",
)
minute_fig.write_html(output_minute_path, include_plotlyjs="cdn")
print(f"Saved minute plot to {output_minute_path.resolve()}")

minute_event_dates = sorted(minute_paths.keys())
_, avg_events = select_recent_events(minute_event_dates, last_n_meetings)
if len(avg_events) >= 2:
    stats = permutation_test_press_conf(
        minute_df=minute_df,
        observed_events=avg_events,
        minutes_before=minutes_before,
        minutes_after=minutes_after,
        n_samples=permutation_samples,
        seed=permutation_seed,
        min_coverage=min_coverage,
    )
    print("Press conference average path significance:")
    print(
        "Observed slopes (pct/min): "
        f"pre={stats['observed_pre_slope']:.6f}, "
        f"post={stats['observed_post_slope']:.6f}, "
        f"delta={stats['observed_delta']:.6f}"
    )
    print(
        "Permutation p-values: "
        f"delta one-sided={stats['delta_p_one_sided']:.4f}, "
        f"delta two-sided={stats['delta_p_two_sided']:.4f}, "
        f"pre>0 one-sided={stats['pre_p_one_sided']:.4f}, "
        f"post<0 one-sided={stats['post_p_one_sided']:.4f}, "
        f"samples={int(stats['samples_used'])}"
    )
