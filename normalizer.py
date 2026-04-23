import pandas as pd


class KlinesNormalizer:
    COLUMNS = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
    ]

    FLOAT_COLS = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base",
        "taker_buy_quote",
    ]

    INT_COLS = [
        "open_time",
        "close_time",
        "trades",
    ]

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # =========================
        # 1. Обрезаем лишние колонки
        # =========================
        df = df.iloc[:, :len(self.COLUMNS)]
        df.columns = self.COLUMNS

        # =========================
        # 2. Приведение типов
        # =========================
        for col in self.FLOAT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

        for col in self.INT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("int64")

        # =========================
        # 3. Timestamp
        # =========================
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

        # =========================
        # 4. Чистка
        # =========================
        df = df.dropna(subset=["timestamp"])
        df = df.drop_duplicates(subset=["open_time"])
        df = df.sort_values("open_time")

        # =========================
        # 5. Финальный порядок колонок
        # =========================
        return df[
            [
                "timestamp",
                "open_time",
                "close_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
            ]
        ]


class AggTradesNormalizer:
    COLUMNS = [
        "agg_trade_id",
        "price",
        "quantity",
        "first_trade_id",
        "last_trade_id",
        "transact_time",
        "is_buyer_maker",
    ]

    FLOAT_COLS = [
        "price",
        "quantity",
    ]

    INT_COLS = [
        "agg_trade_id",
        "first_trade_id",
        "last_trade_id",
        "transact_time",
    ]

    BOOL_COLS = [
        "is_buyer_maker",
    ]

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.iloc[:, :len(self.COLUMNS)].copy()
        df.columns = self.COLUMNS

        for col in self.FLOAT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

        for col in self.INT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in self.BOOL_COLS:
            df[col] = df[col].astype("boolean")

        df = df.dropna(subset=["agg_trade_id", "transact_time"])

        for col in self.INT_COLS:
            df[col] = df[col].astype("int64")

        df["timestamp"] = pd.to_datetime(df["transact_time"], unit="ms", utc=True)

        df = df.dropna(subset=["timestamp"])
        df = df.drop_duplicates(subset=["agg_trade_id"])
        df = df.sort_values(["transact_time", "agg_trade_id"])

        return df[
            [
                "timestamp",
                "transact_time",
                "agg_trade_id",
                "first_trade_id",
                "last_trade_id",
                "price",
                "quantity",
                "is_buyer_maker",
            ]
        ]


class TradesNormalizer:
    COLUMNS = [
        "id",
        "price",
        "qty",
        "quote_qty",
        "time",
        "is_buyer_maker",
    ]

    FLOAT_COLS = [
        "price",
        "qty",
        "quote_qty",
    ]

    INT_COLS = [
        "id",
        "time",
    ]

    BOOL_COLS = [
        "is_buyer_maker",
    ]

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.iloc[:, :len(self.COLUMNS)].copy()
        df.columns = self.COLUMNS

        for col in self.FLOAT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

        for col in self.INT_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in self.BOOL_COLS:
            df[col] = df[col].astype("boolean")

        df = df.dropna(subset=["id", "time"])

        for col in self.INT_COLS:
            df[col] = df[col].astype("int64")

        df["timestamp"] = pd.to_datetime(df["time"], unit="ms", utc=True)

        df = df.dropna(subset=["timestamp"])
        df = df.drop_duplicates(subset=["id"])
        df = df.sort_values(["time", "id"])

        return df[
            [
                "timestamp",
                "time",
                "id",
                "price",
                "qty",
                "quote_qty",
                "is_buyer_maker",
            ]
        ]


def get_normalizer(source: str):
    if source == "klines":
        return KlinesNormalizer()

    if source == "aggTrades":
        return AggTradesNormalizer()

    if source == "trades":
        return TradesNormalizer()

    raise ValueError(f"Unsupported source: {source}")
