"""Price-based features for the unified minute feature pipeline."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import exchange_calendars as xcals


RETURN_LAGS = (1, 2, 3, 5, 10, 15, 30, 60)
CUMULATIVE_RETURN_WINDOWS = (3, 5, 10, 20, 30, 60)
ACCELERATION_WINDOWS = (3, 5, 10, 20, 30)
KAUFMAN_WINDOWS = (5, 10, 20, 30, 60, 90)
CLOSE_ZSCORE_WINDOWS = (5, 10, 20, 30, 60, 90)
VOLATILITY_WINDOWS = (5, 10, 20, 30, 60)
CANDLE_WINDOWS = (5, 10, 20, 30, 60)
VOLUME_WINDOWS = (10, 20, 30, 60)
TRADE_COUNT_WINDOWS = (5, 10, 20, 30)
TRADE_FLOW_WINDOWS = (5, 10, 20, 30)
LARGE_TRADE_QUANTILE = 0.75
TRADE_FLOW_METRICS = (
    "aggression_delta_norm",
    "large_trade_aggression_delta_norm",
    "volume_entropy_norm",
    "inverse_simpson_norm",
    "average_trade_size_quote",
    "pressure_concentration",
)
BTC_RETURN_WINDOWS = (5, 10, 20, 30, 60)
BTC_VOLATILITY_WINDOWS = (10, 20, 30, 60)
BTC_KAUFMAN_WINDOWS = (30, 60)
TIME_FEATURE_COLUMNS = (
    "time_of_day_sin",
    "time_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "is_weekend",
    "is_asia_open",
    "is_cboe_europe_open",
    "is_europe_open",
    "is_us_open",
)
EXCHANGE_GROUPS = {
    "is_asia_open": ("XTKS", "XSHG", "XHKG"),
    "is_europe_open": ("XIST", "XPAR", "XETR", "XSWX", "XSTO"),
    "is_us_open": ("XNYS", "XNAS"),
}
EXCHANGE_CALENDAR_START = "2020-01-01"
EXCHANGE_CALENDAR_END = "2026-12-31"
TARGET_HORIZONS = (10, 20, 30)
TARGET_COLUMNS = tuple(f"target_log_return_{horizon}m" for horizon in TARGET_HORIZONS)

FEATURE_COLUMNS = tuple(
    [f"ada_log_return_lag_{lag}" for lag in RETURN_LAGS]
    + [f"ada_cum_return_{window}m" for window in CUMULATIVE_RETURN_WINDOWS]
    + [f"ada_return_acceleration_{window}m" for window in ACCELERATION_WINDOWS]
    + [f"ada_kaufman_efficiency_{window}m" for window in KAUFMAN_WINDOWS]
    + [f"ada_close_zscore_{window}m" for window in CLOSE_ZSCORE_WINDOWS]
    + [f"ada_return_std_{window}m" for window in VOLATILITY_WINDOWS]
    + [f"ada_realized_volatility_{window}m" for window in VOLATILITY_WINDOWS]
    + [
        f"ada_{metric}_{window}m"
        for window in CANDLE_WINDOWS
        for metric in (
            "upper_wick",
            "lower_wick",
            "log_range",
            "close_position",
            "high_close_ratio",
        )
    ]
    + [
        f"ada_{metric}_{window}m"
        for window in VOLUME_WINDOWS
        for metric in ("volume_sum", "volume_zscore")
    ]
    + [f"ada_trades_per_minute_{window}m" for window in TRADE_COUNT_WINDOWS]
    + [
        f"ada_{metric}_{window}m"
        for window in TRADE_FLOW_WINDOWS
        for metric in TRADE_FLOW_METRICS
    ]
    + [f"btc_log_return_{window}m" for window in BTC_RETURN_WINDOWS]
    + [f"btc_return_std_{window}m" for window in BTC_VOLATILITY_WINDOWS]
    + [f"btc_kaufman_efficiency_{window}m" for window in BTC_KAUFMAN_WINDOWS]
    + list(TIME_FEATURE_COLUMNS)
)


def _prepare_minute_frame(klines: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", "volume", "trades"}
    missing = required.difference(klines.columns)
    if missing:
        raise ValueError(f"Missing required kline columns: {sorted(missing)}")

    frame = klines.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce").astype("float64")
    frame["trades"] = pd.to_numeric(frame["trades"], errors="coerce").astype("float64")

    if frame["timestamp"].isna().any():
        raise ValueError("Klines contain invalid timestamps")
    if frame["timestamp"].duplicated().any():
        duplicates = frame.loc[frame["timestamp"].duplicated(), "timestamp"].head().tolist()
        raise ValueError(f"Klines contain duplicate timestamps: {duplicates}")
    if (frame[["open", "high", "low", "close"]].dropna() <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    if (frame["volume"].dropna() < 0).any():
        raise ValueError("Volume must be non-negative")
    if (frame["trades"].dropna() < 0).any():
        raise ValueError("Trade count must be non-negative")

    valid_ohlc = frame[["open", "high", "low", "close"]].dropna()
    invalid = (
        valid_ohlc["high"].lt(valid_ohlc[["open", "close"]].max(axis=1))
        | valid_ohlc["low"].gt(valid_ohlc[["open", "close"]].min(axis=1))
        | valid_ohlc["high"].lt(valid_ohlc["low"])
    )
    if invalid.any():
        raise ValueError("Klines contain inconsistent OHLC prices")

    frame = frame.sort_values("timestamp").set_index("timestamp")
    if frame.empty:
        return frame

    # A shift must mean a calendar-minute shift, not merely the previous row.
    # Reindexing makes gaps explicit and prevents returns from jumping over them.
    complete_index = pd.date_range(
        start=frame.index.min().floor("min"),
        end=frame.index.max().floor("min"),
        freq="1min",
        tz="UTC",
        name="timestamp",
    )
    return frame.reindex(complete_index)


def _prepare_trades(raw_trades: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "qty", "quote_qty", "is_buyer_maker"}
    missing = required.difference(raw_trades.columns)
    if missing:
        raise ValueError(f"Missing required trades columns: {sorted(missing)}")

    trades = raw_trades.copy()
    trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True, errors="coerce")
    trades["qty"] = pd.to_numeric(trades["qty"], errors="coerce").astype("float64")
    trades["quote_qty"] = pd.to_numeric(
        trades["quote_qty"], errors="coerce"
    ).astype("float64")
    trades["is_buyer_maker"] = trades["is_buyer_maker"].astype("boolean")
    required_values = trades[["timestamp", "qty", "quote_qty", "is_buyer_maker"]]
    if required_values.isna().any().any():
        raise ValueError("trades contain invalid required values")
    if (trades[["qty", "quote_qty"]] <= 0).any().any():
        raise ValueError("trade base and quote quantities must be positive")
    return trades.sort_values("timestamp").reset_index(drop=True)


def _trade_window_statistics(
    base_quantity: np.ndarray,
    quote_quantity: np.ndarray,
    is_buyer_maker: np.ndarray,
    event_ns: np.ndarray,
    pressure_split_ns: int,
) -> tuple[float, float, float, float, float, float]:
    if base_quantity.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    buy_volume = base_quantity[~is_buyer_maker].sum()
    sell_volume = base_quantity[is_buyer_maker].sum()
    total_volume = buy_volume + sell_volume
    aggression_delta = (buy_volume - sell_volume) / total_volume

    large_threshold = np.quantile(base_quantity, LARGE_TRADE_QUANTILE)
    large = base_quantity >= large_threshold
    large_quantity = base_quantity[large]
    large_side = is_buyer_maker[large]
    large_buy = large_quantity[~large_side].sum()
    large_sell = large_quantity[large_side].sum()
    large_total = large_buy + large_sell
    large_delta = (large_buy - large_sell) / large_total

    weights = base_quantity / base_quantity.sum()
    if base_quantity.size == 1:
        entropy_norm = 0.0
    else:
        entropy = -(weights * np.log(weights)).sum()
        entropy_norm = entropy / np.log(base_quantity.size)
    inverse_simpson_norm = 1.0 / (base_quantity.size * np.square(weights).sum())
    average_trade_size_quote = np.mean(quote_quantity)

    signed_quantity = np.where(is_buyer_maker, -base_quantity, base_quantity)
    start_delta = signed_quantity[event_ns < pressure_split_ns].sum()
    end_delta = signed_quantity[event_ns >= pressure_split_ns].sum()
    total_delta = start_delta + end_delta
    pressure_concentration = (
        0.0 if np.isclose(total_delta, 0.0) else (end_delta - start_delta) / total_delta
    )
    return (
        aggression_delta,
        large_delta,
        entropy_norm,
        inverse_simpson_norm,
        average_trade_size_quote,
        pressure_concentration,
    )


def _build_trade_flow_features(
    raw_trades: pd.DataFrame,
    minute_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    trades = _prepare_trades(raw_trades)
    columns = [
        f"ada_{metric}_{window}m"
        for window in TRADE_FLOW_WINDOWS
        for metric in TRADE_FLOW_METRICS
    ]
    result = pd.DataFrame(np.nan, index=minute_index, columns=columns, dtype="float64")

    if trades.empty or minute_index.empty:
        return result

    # Loaded partitions define how much causal history is actually available.
    history_start = trades["timestamp"].min().normalize()
    # Timestamp.value is always nanoseconds; pandas may otherwise expose the
    # underlying datetime array in microseconds depending on its version.
    event_ns = trades["timestamp"].map(lambda timestamp: timestamp.value).to_numpy(
        dtype="int64"
    )
    base_quantity = trades["qty"].to_numpy(dtype="float64")
    quote_quantity = trades["quote_qty"].to_numpy(dtype="float64")
    maker = trades["is_buyer_maker"].to_numpy(dtype="bool")

    for window in TRADE_FLOW_WINDOWS:
        values = np.full(
            (len(minute_index), len(TRADE_FLOW_METRICS)), np.nan, dtype="float64"
        )
        for row, minute in enumerate(minute_index):
            start = minute - pd.Timedelta(minutes=window - 1)
            if start < history_start:
                continue
            end = minute + pd.Timedelta(minutes=1)
            left = np.searchsorted(event_ns, start.value, side="left")
            right = np.searchsorted(event_ns, end.value, side="left")
            pressure_split = end - pd.Timedelta(minutes=window / 5)
            values[row] = _trade_window_statistics(
                base_quantity[left:right],
                quote_quantity[left:right],
                maker[left:right],
                event_ns[left:right],
                pressure_split.value,
            )

        for metric_index, metric in enumerate(TRADE_FLOW_METRICS):
            result[f"ada_{metric}_{window}m"] = values[:, metric_index]
    return result


def _build_btc_features(
    btc_klines: pd.DataFrame,
    minute_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    btc = _prepare_minute_frame(btc_klines)
    result = pd.DataFrame(index=btc.index)
    log_close = np.log(btc["close"])
    minute_return = log_close.diff()

    for window in BTC_RETURN_WINDOWS:
        result[f"btc_log_return_{window}m"] = log_close - log_close.shift(window)

    for window in BTC_VOLATILITY_WINDOWS:
        result[f"btc_return_std_{window}m"] = minute_return.rolling(
            window, min_periods=window
        ).std(ddof=1)

    absolute_price_change = btc["close"].diff().abs()
    for window in BTC_KAUFMAN_WINDOWS:
        direction = (btc["close"] - btc["close"].shift(window)).abs()
        path = absolute_price_change.rolling(window, min_periods=window).sum()
        efficiency = (direction / path).mask(path.eq(0), 0.0)
        result[f"btc_kaufman_efficiency_{window}m"] = efficiency

    columns = [
        *[f"btc_log_return_{window}m" for window in BTC_RETURN_WINDOWS],
        *[f"btc_return_std_{window}m" for window in BTC_VOLATILITY_WINDOWS],
        *[f"btc_kaufman_efficiency_{window}m" for window in BTC_KAUFMAN_WINDOWS],
    ]
    return result.reindex(minute_index)[columns]


def _calendar_open_mask(
    minute_index: pd.DatetimeIndex,
    calendar_names: tuple[str, ...],
) -> np.ndarray:
    mask = np.zeros(len(minute_index), dtype="bool")
    if minute_index.empty:
        return mask

    start = (minute_index.min() - pd.Timedelta(days=1)).date().isoformat()
    end = (minute_index.max() + pd.Timedelta(days=1)).date().isoformat()
    for calendar_name in calendar_names:
        calendar = _get_exchange_calendar(calendar_name)
        schedule = calendar.schedule.loc[start:end]
        for _, session in schedule.iterrows():
            session_mask = (
                (minute_index >= session["open"])
                & (minute_index < session["close"])
            )
            break_start = session.get("break_start")
            break_end = session.get("break_end")
            if pd.notna(break_start) and pd.notna(break_end):
                session_mask &= ~(
                    (minute_index >= break_start) & (minute_index < break_end)
                )
            mask |= np.asarray(session_mask)
    return mask


def _cboe_europe_open_mask(minute_index: pd.DatetimeIndex) -> np.ndarray:
    """Cboe Europe proxy: 08:00-22:00 Paris time on XPAR session days."""

    mask = np.zeros(len(minute_index), dtype="bool")
    if minute_index.empty:
        return mask

    start = (minute_index.min() - pd.Timedelta(days=1)).date().isoformat()
    end = (minute_index.max() + pd.Timedelta(days=1)).date().isoformat()
    sessions = _get_exchange_calendar("XPAR").schedule.loc[start:end].index
    for session_label in sessions:
        local_midnight = pd.Timestamp(session_label).tz_localize("Europe/Paris")
        session_open = (local_midnight + pd.Timedelta(hours=8)).tz_convert("UTC")
        session_close = (local_midnight + pd.Timedelta(hours=22)).tz_convert("UTC")
        mask |= np.asarray(
            (minute_index >= session_open) & (minute_index < session_close)
        )
    return mask


@lru_cache(maxsize=None)
def _get_exchange_calendar(calendar_name: str):
    """Create each calendar once with bounds safe for this dataset range."""

    return xcals.get_calendar(
        calendar_name,
        start=EXCHANGE_CALENDAR_START,
        end=EXCHANGE_CALENDAR_END,
    )


def _build_exchange_session_features(
    minute_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    result = pd.DataFrame(index=minute_index)
    for feature_name, calendar_names in EXCHANGE_GROUPS.items():
        result[feature_name] = _calendar_open_mask(minute_index, calendar_names)
    result["is_cboe_europe_open"] = _cboe_europe_open_mask(minute_index)
    return result[
        ["is_asia_open", "is_cboe_europe_open", "is_europe_open", "is_us_open"]
    ]


def build_price_features(
    klines: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    btc_klines: pd.DataFrame | None = None,
    future_klines: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build causal ADA return features from ordered one-minute klines.

    ``ada_log_return_lag_1`` is the return of the most recently completed
    minute: ``log(C_t / C_{t-1})``. For lag ``k``, the minute return is shifted
    by ``k - 1`` rows. Cumulative return over ``k`` minutes is
    ``log(C_t / C_{t-k})``. Return acceleration is ``r_t - r_{t-k}``.
    Kaufman's efficiency ratio is the absolute net price movement divided by
    the sum of absolute one-minute movements. Close-price z-scores use the
    population standard deviation of the trailing window, including ``C_t``.
    Return standard deviation uses the sample definition (``ddof=1``) from the
    source document. Realized volatility is the
    unannualized ``sqrt(sum(r_i ** 2))`` over the trailing return window.
    Candle geometry contains normalized upper/lower wicks, ``log(high/low)``,
    close position inside the candle, and the share of one-minute candles whose
    close position is at least 0.8. For every window, geometry is calculated
    from a synthetic rolling candle: first open, maximum high, minimum low,
    and last close. It is not an average of one-minute geometry values.
    Volume features contain the trailing sum of base volume and the z-score of
    the current minute's volume relative to the same trailing window.
    Trade-rate features average the Binance kline ``trades`` field over each
    trailing window, producing the mean number of trades per minute.
    Trade-flow features use raw non-aggregated trades. Buyer-taker volume is where
    ``is_buyer_maker`` is false. Large trades are the upper quartile by base
    quantity inside each causal rolling window.
    Average trade size is measured in quote currency. Pressure concentration
    compares signed base-volume delta in the last 20% versus the first 80% of
    the window and divides the difference by total signed delta.
    BTC features are aligned to the ADA minute backbone and contain trailing
    log returns, return standard deviations, and Kaufman efficiency ratios.
    Calendar features use UTC and encode minute-of-day and Monday-based
    day-of-week cyclically; ``is_weekend`` is true on Saturday and Sunday.
    Exchange-session flags use exchange-calendars schedules. Group flags are
    true when at least one selected exchange is trading. Cboe Europe uses a
    documented 08:00-22:00 Europe/Paris proxy on Euronext Paris session days.
    Forward targets are ``log(C_{t+h} / C_t)`` for 10, 20, and 30 minutes.
    """

    frame = _prepare_minute_frame(klines)
    result = frame.reset_index()[["timestamp"]]
    if frame.empty:
        for column in FEATURE_COLUMNS:
            result[column] = pd.Series(dtype="float64")
        return result

    log_close = np.log(frame["close"])
    minute_return = log_close.diff()

    features = pd.DataFrame(index=frame.index)

    minute_of_day = frame.index.hour * 60 + frame.index.minute
    day_of_week = frame.index.dayofweek
    features["time_of_day_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    features["time_of_day_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    features["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    features["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    features["is_weekend"] = day_of_week >= 5
    exchange_features = _build_exchange_session_features(frame.index)
    features[exchange_features.columns] = exchange_features

    for lag in RETURN_LAGS:
        features[f"ada_log_return_lag_{lag}"] = minute_return.shift(lag - 1)

    for window in CUMULATIVE_RETURN_WINDOWS:
        features[f"ada_cum_return_{window}m"] = log_close - log_close.shift(window)

    for window in ACCELERATION_WINDOWS:
        features[f"ada_return_acceleration_{window}m"] = (
            minute_return - minute_return.shift(window)
        )

    absolute_price_change = frame["close"].diff().abs()
    for window in KAUFMAN_WINDOWS:
        direction = (frame["close"] - frame["close"].shift(window)).abs()
        path = absolute_price_change.rolling(window, min_periods=window).sum()
        efficiency = direction / path
        # A fully flat, otherwise valid window has no directional movement.
        efficiency = efficiency.mask(path.eq(0), 0.0)
        features[f"ada_kaufman_efficiency_{window}m"] = efficiency

    for window in CLOSE_ZSCORE_WINDOWS:
        rolling = frame["close"].rolling(window, min_periods=window)
        mean = rolling.mean()
        std = rolling.std(ddof=0)
        zscore = (frame["close"] - mean) / std
        # The current price equals the mean in a constant valid window.
        zscore = zscore.mask(std.eq(0), 0.0)
        features[f"ada_close_zscore_{window}m"] = zscore

    squared_return = minute_return.pow(2)
    for window in VOLATILITY_WINDOWS:
        rolling_return = minute_return.rolling(window, min_periods=window)
        features[f"ada_return_std_{window}m"] = rolling_return.std(ddof=1)
        features[f"ada_realized_volatility_{window}m"] = np.sqrt(
            squared_return.rolling(window, min_periods=window).sum()
        )

    minute_range = frame["high"] - frame["low"]
    minute_close_position = (frame["close"] - frame["low"]) / minute_range
    minute_close_position = minute_close_position.mask(minute_range.eq(0), 0.5)
    high_close = minute_close_position.ge(0.8).where(
        minute_close_position.notna()
    ).astype("float64")

    for window in CANDLE_WINDOWS:
        rolling_open = frame["open"].shift(window - 1)
        rolling_high = frame["high"].rolling(window, min_periods=window).max()
        rolling_low = frame["low"].rolling(window, min_periods=window).min()
        rolling_close = frame["close"]
        rolling_range = rolling_high - rolling_low

        upper_wick = (
            rolling_high - pd.concat([rolling_open, rolling_close], axis=1).max(axis=1)
        ) / rolling_range
        lower_wick = (
            pd.concat([rolling_open, rolling_close], axis=1).min(axis=1) - rolling_low
        ) / rolling_range
        close_position = (rolling_close - rolling_low) / rolling_range

        flat_candle = rolling_range.eq(0)
        upper_wick = upper_wick.mask(flat_candle, 0.0)
        lower_wick = lower_wick.mask(flat_candle, 0.0)
        close_position = close_position.mask(flat_candle, 0.5)

        features[f"ada_upper_wick_{window}m"] = upper_wick
        features[f"ada_lower_wick_{window}m"] = lower_wick
        features[f"ada_log_range_{window}m"] = np.log(rolling_high / rolling_low)
        features[f"ada_close_position_{window}m"] = close_position
        features[f"ada_high_close_ratio_{window}m"] = high_close.rolling(
            window, min_periods=window
        ).mean()

    for window in VOLUME_WINDOWS:
        rolling_volume = frame["volume"].rolling(window, min_periods=window)
        volume_mean = rolling_volume.mean()
        volume_std = rolling_volume.std(ddof=0)
        volume_zscore = (frame["volume"] - volume_mean) / volume_std
        volume_zscore = volume_zscore.mask(volume_std.eq(0), 0.0)

        features[f"ada_volume_sum_{window}m"] = rolling_volume.sum()
        features[f"ada_volume_zscore_{window}m"] = volume_zscore

    for window in TRADE_COUNT_WINDOWS:
        features[f"ada_trades_per_minute_{window}m"] = frame["trades"].rolling(
            window, min_periods=window
        ).mean()

    trade_feature_columns = [
        f"ada_{metric}_{window}m"
        for window in TRADE_FLOW_WINDOWS
        for metric in TRADE_FLOW_METRICS
    ]
    if trades is None:
        trade_features = pd.DataFrame(
            np.nan,
            index=features.index,
            columns=trade_feature_columns,
            dtype="float64",
        )
    else:
        trade_features = _build_trade_flow_features(trades, frame.index)

    btc_feature_columns = [column for column in FEATURE_COLUMNS if column.startswith("btc_")]
    if btc_klines is None:
        btc_features = pd.DataFrame(
            np.nan,
            index=features.index,
            columns=btc_feature_columns,
            dtype="float64",
        )
    else:
        btc_features = _build_btc_features(btc_klines, frame.index)

    features = pd.concat(
        [
            features,
            trade_features[trade_feature_columns],
            btc_features[btc_feature_columns],
        ],
        axis=1,
    ).copy()

    target_close = frame[["close"]]
    if future_klines is not None:
        future = _prepare_minute_frame(future_klines)[["close"]]
        overlap = target_close.index.intersection(future.index)
        if not overlap.empty:
            raise ValueError(f"future_klines overlap feature klines: {overlap[:5].tolist()}")
        target_close = pd.concat([target_close, future]).sort_index()

    target_log_close = np.log(target_close["close"])
    targets = pd.DataFrame(index=frame.index)
    for horizon in TARGET_HORIZONS:
        forward_return = target_log_close.shift(-horizon) - target_log_close
        targets[f"target_log_return_{horizon}m"] = forward_return.reindex(frame.index)
    features = pd.concat([features, targets], axis=1)

    return features.reset_index()[["timestamp", *FEATURE_COLUMNS, *TARGET_COLUMNS]]
