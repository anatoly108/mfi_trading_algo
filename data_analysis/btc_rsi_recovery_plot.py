"""
Script to visualize BTC price paths around RSI < 30 events.

Adjust the variables at the bottom of the file before running in an
interactive session to change inputs or output paths.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objs as go
import talib


DEFAULT_THRESHOLD = 30
DEFAULT_TIMEPERIOD = 14


def load_daily_candles(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Timestamp" not in df.columns:
        raise ValueError("Input CSV must include a 'Timestamp' column with epoch seconds")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s", utc=True)
    df = df.set_index("Timestamp")

    daily = (
        df.resample("1D")
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


def find_rsi_events(daily: pd.DataFrame, days_before: int, days_after: int) -> List[pd.Timestamp]:
    rsi = talib.RSI(daily["Close"], timeperiod=DEFAULT_TIMEPERIOD)
    daily["RSI"] = rsi

    crosses = (daily["RSI"].shift(1) >= DEFAULT_THRESHOLD) & (daily["RSI"] < DEFAULT_THRESHOLD)
    event_dates = daily.index[crosses].to_list()
    print("Last event: " + str(event_dates[0]))

    filtered_events: List[pd.Timestamp] = []
    min_spacing = days_before + days_after
    for event in event_dates:
        if not filtered_events:
            filtered_events.append(event)
            continue
        if (event - filtered_events[-1]).days > min_spacing:
            filtered_events.append(event)

    return filtered_events


def extract_price_path(
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


def build_paths(
    daily: pd.DataFrame,
    events: List[pd.Timestamp],
    days_before: int,
    days_after: int,
) -> Dict[pd.Timestamp, Tuple[pd.Series, float]]:
    paths: Dict[pd.Timestamp, Tuple[pd.Series, float]] = {}
    # event_date = events[-1]
    for event_date in events:
        series = extract_price_path(daily, event_date, days_before, days_after)
        if series is not None:
            paths[event_date] = series
    return paths


def make_figure(
    paths: Dict[pd.Timestamp, Tuple[pd.Series, float]],
    last_n: int,
    show_individual: bool,
) -> go.Figure:
    if not paths:
        raise ValueError("No RSI < 30 events with complete data were found")

    event_dates = sorted(paths.keys())
    last_event = event_dates[-1]
    previous_events = event_dates[:-1]
    last_n_events = previous_events[-last_n:] if last_n > 0 else previous_events

    x_axis = paths[last_event][0].index
    fig = go.Figure()

    if show_individual:
        for event in last_n_events:
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
                        "Day %{x}: %{y:.2f}%<br>"
                        "Base price: %{text}<extra>%{fullData.name}</extra>"
                    ),
                    showlegend=False,
                )
            )

    avg_series = pd.concat([paths[e][0] for e in last_n_events], axis=1).mean(axis=1)
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=avg_series,
            mode="lines",
            name=f"Average of last {len(last_n_events)}", 
            line=dict(color="#e91e63", width=3),
            hovertemplate="Day %{x}: %{y:.2f}%<extra>Average</extra>",
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
                "Day %{x}: %{y:.2f}%<br>"
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
        title=f"BTC price path around RSI < {DEFAULT_THRESHOLD}",
        xaxis_title="Days relative to RSI 30 cross",
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
        text=f"Oversold (RSI < {DEFAULT_THRESHOLD})",
        showarrow=False,
        font=dict(color="#888"),
    )

    return fig


csv_path = Path("/Users/anatoly/projects/trading/data_analysis/data/btcusd_1-min_data.csv")
days = 120
events_to_average = 8
output_path = Path("btc_rsi_paths_individual.html")
show_individual = True

days_before = days_after = days

daily = load_daily_candles(csv_path)
daily = daily.sort_values("Timestamp")
events = find_rsi_events(daily, days_before, days_after)
if len(events) < 2:
    raise ValueError("Not enough RSI < 30 events found to build the plot")

paths = build_paths(daily, events, days_before, days_after)
fig = make_figure(
    paths, events_to_average, show_individual
)
fig.write_html(output_path, include_plotlyjs="cdn")
print(f"Saved plot to {output_path.resolve()}")
