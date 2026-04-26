import argparse
from pathlib import Path
import pandas as pd
import pyarrow.dataset as ds


def load_first_klines_row(klines_path: Path) -> pd.Series:
    df = pd.read_parquet(klines_path, columns=["open_time"])
    if df.empty:
        raise ValueError(f"Klines file is empty: {klines_path}")
    first_row = df.sort_values("open_time").iloc[0]
    return first_row


def load_trades_minute(trades_path: Path, minute_start_ms: int) -> pd.DataFrame:
    # Сначала прочитаем схему файла, чтобы определить правильное имя поля времени
    import pyarrow.parquet as pq
    parquet_file = pq.ParquetFile(trades_path)
    schema = parquet_file.schema_arrow

    # Определим поле времени (может быть 'time' или 'transact_time')
    time_field = None
    if 'time' in schema.names:
        time_field = 'time'
    elif 'transact_time' in schema.names:
        time_field = 'transact_time'
    else:
        raise ValueError(f"Could not find time field in {trades_path}. Available fields: {schema.names}")

    # Определим поле количества (может быть 'qty' или 'quantity')
    qty_field = None
    if 'qty' in schema.names:
        qty_field = 'qty'
    elif 'quantity' in schema.names:
        qty_field = 'quantity'
    else:
        raise ValueError(f"Could not find quantity field in {trades_path}. Available fields: {schema.names}")

    # Определим поле quote_qty (может отсутствовать в aggTrades)
    quote_field = None
    columns = [time_field, "price", qty_field, "is_buyer_maker"]
    if 'quote_qty' in schema.names:
        quote_field = 'quote_qty'
        columns.append(quote_field)

    dataset = ds.dataset(str(trades_path), format="parquet")
    predicate = (
        (ds.field(time_field) >= minute_start_ms)
        & (ds.field(time_field) < minute_start_ms + 60_000)
    )
    table = dataset.to_table(
        columns=columns,
        filter=predicate,
    )
    df = table.to_pandas()

    # Если quote_qty отсутствует, вычислим его как price * quantity
    if quote_field is None:
        df['quote_qty'] = df['price'] * df[qty_field]

    if df.empty:
        raise ValueError(
            f"No trades found for minute starting at {minute_start_ms} (UTC)."
        )
    df = df.sort_values([time_field]).reset_index(drop=True)
    return df


def build_klines_row_from_trades(trades_df: pd.DataFrame, minute_start_ms: int) -> pd.DataFrame:
    # Определим правильные имена полей
    qty_field = 'qty' if 'qty' in trades_df.columns else 'quantity'

    open_price = float(trades_df.iloc[0]["price"])
    close_price = float(trades_df.iloc[-1]["price"])
    high_price = float(trades_df["price"].max())
    low_price = float(trades_df["price"].min())
    volume = float(trades_df[qty_field].sum())
    quote_volume = float(trades_df["quote_qty"].sum())
    trades_count = int(len(trades_df))
    taker_buy_mask = ~trades_df["is_buyer_maker"].astype(bool)
    taker_buy_base = float(trades_df.loc[taker_buy_mask, qty_field].sum())
    taker_buy_quote = float(trades_df.loc[taker_buy_mask, "quote_qty"].sum())

    row = {
        "timestamp": pd.to_datetime(minute_start_ms, unit="ms", utc=True),
        "open_time": minute_start_ms,
        "close_time": minute_start_ms + 59_999,
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "volume": volume,
        "quote_volume": quote_volume,
        "trades": trades_count,
        "taker_buy_base": taker_buy_base,
        "taker_buy_quote": taker_buy_quote,
    }
    return pd.DataFrame([row])


def restore_first_minute(klines_path: Path, trades_path: Path, output_path: Path) -> pd.DataFrame:
    first_klines = load_first_klines_row(klines_path)
    missing_minute_start = int(first_klines["open_time"] - 60_000)

    trades_minute = load_trades_minute(trades_path, missing_minute_start)
    restored_row = build_klines_row_from_trades(trades_minute, missing_minute_start)

    klines_df = pd.read_parquet(klines_path)
    if (klines_df["open_time"] == missing_minute_start).any():
        raise ValueError("Klines already contain the first minute row.")

    result = pd.concat([restored_row, klines_df], ignore_index=True)
    result = result.sort_values("open_time").reset_index(drop=True)

    result = result.astype(
        {
            "open_time": "int64",
            "close_time": "int64",
            "open": "float32",
            "high": "float32",
            "low": "float32",
            "close": "float32",
            "volume": "float32",
            "quote_volume": "float32",
            "trades": "int64",
            "taker_buy_base": "float32",
            "taker_buy_quote": "float32",
        }
    )

    result.to_parquet(output_path, index=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore the missing first 1-minute kline from trade data."
    )
    parser.add_argument(
        "--klines",
        type=Path,
        default=Path("klines_ada.parquet"),
        help="Path to the klines parquet file with the missing first minute.",
    )
    parser.add_argument(
        "--trades",
        type=Path,
        default=Path("trades_ada.parquet"),
        help="Path to the trades parquet file for the same day.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("klines_ada_restored.parquet"),
        help="Path to write the restored klines parquet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = restore_first_minute(args.klines, args.trades, args.output)
    print(f"Restored first minute and saved to: {args.output}")
    print(output.head())


if __name__ == "__main__":
    main()
