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