import io
import zipfile
import time
from typing import Optional

import requests
import pandas as pd


class BinanceClient:
    def __init__(
        self,
        base_root: str,
        timeout: tuple[int, int] = (10, 60),
        max_retries: int = 3,
        backoff: int = 2,
    ):
        """
        base_root: https://data.binance.vision/data/futures/um/daily
        """
        self.base_root = base_root.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff

    # =========================
    # URL BUILDER
    # =========================
    def build_url(
        self,
        source: str,
        symbol: str,
        date_str: str,
        interval: Optional[str] = None,
    ) -> str:
        """
        Примеры:
        klines:
        /klines/ADAUSDT/1m/ADAUSDT-1m-2020-01-01.zip

        aggTrades:
        /aggTrades/ADAUSDT/ADAUSDT-aggTrades-2020-01-01.zip
        """
        if interval:
            return (
                f"{self.base_root}/{source}/{symbol}/{interval}/"
                f"{symbol}-{interval}-{date_str}.zip"
            )
        else:
            return (
                f"{self.base_root}/{source}/{symbol}/"
                f"{symbol}-{source}-{date_str}.zip"
            )

    # =========================
    # DOWNLOAD WITH RETRY
    # =========================
    def download_zip(self, url: str) -> Optional[bytes]:
        for attempt in range(1, self.max_retries + 1):
            try:
                r = requests.get(url, timeout=self.timeout)

                if r.status_code == 200:
                    return r.content

                if r.status_code == 404:
                    # данных нет — это нормально
                    return None

                # другие коды — пробуем ретрай
                raise RuntimeError(f"HTTP {r.status_code}")

            except Exception as e:
                if attempt == self.max_retries:
                    raise

                sleep_time = self.backoff ** attempt
                time.sleep(sleep_time)

        return None

    # =========================
    # ZIP → DATAFRAME
    # =========================
    def unzip_to_df(self, content: bytes) -> pd.DataFrame:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            file_name = z.namelist()[0]

            with z.open(file_name) as f:
                df = pd.read_csv(f, header=0, low_memory=False)

        df = df.dropna(how="all")
        return df

    # =========================
    # HIGH LEVEL API
    # =========================
    def load_day(
        self,
        source: str,
        symbol: str,
        date_str: str,
        interval: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Главный метод:
        скачать → распаковать → вернуть df
        """
        url = self.build_url(source, symbol, date_str, interval)

        content = self.download_zip(url)
        # print(url)
        # print(content[:200])

        if content is None:
            return None

        df = self.unzip_to_df(content)

        return df