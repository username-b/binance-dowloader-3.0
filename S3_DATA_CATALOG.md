# Каталог данных S3

Дата инвентаризации: 8 июня 2026 года.

Бакет: `s3://binance-data-downloader`

Все времена в наборах считаются UTC. Основная гранулярность признаков и
таргетов — одна минута. Стандартный путь дневной партиции:

```text
<layer>/<dataset>/symbol=<SYMBOL>/interval=1m/date=YYYY-MM-DD/data.parquet
```

## Обозначения

- `p_i`, `q_i` — цена и количество сделки `i`.
- `s_i = +1` для агрессивной покупки и `-1` для агрессивной продажи.
- В Binance `is_buyer_maker = false` означает агрессивную покупку.
- `O, H, L, C` — open, high, low, close минутной свечи.
- `V_buy`, `V_sell` — объём агрессивных покупок и продаж.
- При нулевом знаменателе функции построения записывают `0`.

## Фактическая структура

| Слой / датасет | Период | Файлов | Размер | Состояние |
|---|---:|---:|---:|---|
| `raw/klines/ADAUSDT/1m` | 2020-02-01 — 2026-02-02 | 2194 | 0.16 GiB | основной |
| `raw/klines/BTCUSDT/1m` | 2020-02-01 — 2026-02-01 | 2193 | 0.19 GiB | основной |
| `raw/aggTrades/ADAUSDT` | 2020-02-01 — 2026-02-01 | 2193 | 7.13 GiB | основной |
| `raw/trades/ADAUSDT` | 2020-02-01 — 2026-02-01 | 2193 | 16.20 GiB | основной |
| `raw/trades/BTCUSDT` | 2020-02-01 — 2022-08-10 | 922 | 23.39 GiB | неполный |
| `raw/sinthetic_data` | 2020-02-01 — 2026-02-01 | 2193 | 0.08 GiB | опечатка в имени |
| `features/aggression_features` | 2020-02-01 — 2026-02-01 | 2193 | 0.14 GiB | полный |
| `features/btc_features` | 2020-02-01 — 2026-02-01 | 2193 | 0.11 GiB | полный |
| `features/intraminute_features` | 2020-02-01 — 2026-02-01 | 2193 | 0.13 GiB | полный |
| `features/intraminute_segments` | 2020-02-01 — 2026-02-01 | 2193 | 0.25 GiB | полный |
| `features/intraminute_dynamics` | 2020-02-01 — 2026-02-01 | 2193 | 0.15 GiB | полный |
| `features/price_pressure` | 2020-02-01 — 2026-02-01 | 2193 | 0.14 GiB | полный, пересобран |
| `features/trade_distribution` | 2020-02-01 — 2026-02-01 | 2193 | 0.17 GiB | полный |
| `features/trades_minute_level` | 2020-02-01 — 2026-02-01 | 2193 | 0.14 GiB | полный |
| `features/return_{1,15,20,25}m_forward` | 2020-02-01 — 2026-02-01 | 2193 каждый | 0.05 GiB каждый | по смыслу таргеты |
| `targets/future_returns` | 2020-02-01 — 2026-02-01 | 2193 | 1.79 GiB | горизонты 1–120 мин |
| `raw_backup/*` | разные | 9 232 | 30.7 GiB | резервные дубли |
| parquet в корне бакета | без партиций | 2 | 1.65 GiB | модельные витрины |

## Сырые данные

### `raw/klines`

Источник: минутные свечи Binance.

Поля: `timestamp`, `open_time`, `close_time`, `open`, `high`, `low`, `close`,
`volume`, `quote_volume`, `trades`, `taker_buy_base`, `taker_buy_quote`.

Это опорная временная сетка для всех минутных наборов.

### `raw/aggTrades`

Агрегированные сделки Binance: `transact_time`, `agg_trade_id`,
`first_trade_id`, `last_trade_id`, `price`, `quantity`, `is_buyer_maker`.

Используются в `aggression_features`, `intraminute_segments`,
`intraminute_dynamics` и `price_pressure`.

### `raw/trades`

Неагрегированные сделки: `time`, `id`, `price`, `qty`, `quote_qty`,
`is_buyer_maker`. Используются в `trades_minute_level` и
`trade_distribution`.

### `raw/sinthetic_data`

Календарные признаки: `hour`, `minute`, `weekday`, `t`, `cos_hour`,
`sin_hour`, `t_week`, `cos_weekday`, `sin_weekday`, `is_weekend`,
`asia_open`, `europe_open`, `us_open`.

Точная функция-генератор в доступном коде не найдена. Имя следует исправить
на `synthetic_data` только через версионированную миграцию.

## Признаки

### `aggression_features` и `trades_minute_level`

Наборы имеют одинаковую схему, но первый рассчитан по `aggTrades`, второй по
обычным `trades`.

```text
V_buy_base  = Σ q_i, где s_i = +1
V_sell_base = Σ q_i, где s_i = -1
delta_base  = V_buy_base - V_sell_base
delta_base_norm = delta_base / (V_buy_base + V_sell_base)

V_buy_quote  = Σ (p_i q_i), где s_i = +1
V_sell_quote = Σ (p_i q_i), где s_i = -1
delta_quote  = V_buy_quote - V_sell_quote
delta_quote_norm = delta_quote / (V_buy_quote + V_sell_quote)
```

### `intraminute_features`

```text
log_close       = ln(C)
candle_range    = H - L
body            = C - O
upper_wick      = H - max(O, C)
lower_wick      = min(O, C) - L
body_norm       = body / candle_range
wick_upper_norm = upper_wick / candle_range
wick_lower_norm = lower_wick / candle_range
volume_log      = ln(1 + volume)
quote_volume_log = ln(1 + quote_volume)
taker_buy_ratio = taker_buy_base / volume
```

### `intraminute_segments`

Минута делится на три сегмента: `[0,20)`, `[20,40)`, `[40,60)` секунд.

```text
VWAP_minute  = Σ(p_i q_i) / Σq_i
VWAP_segment_j = Σ_j(p_i q_i) / Σ_j q_i
V_buy_segment_j  = Σ_j q_i, где s_i = +1
V_sell_segment_j = Σ_j q_i, где s_i = -1
delta_segment_j  = V_buy_segment_j - V_sell_segment_j
pressure_segment_j = Σ_j s_i q_i (p_i - VWAP_minute)
```

### `intraminute_dynamics`

```text
delta_total = delta_segment_1 + delta_segment_2 + delta_segment_3
L   = delta_segment_3 / delta_total
r_1 = ln(VWAP_segment_2 / VWAP_segment_1)
r_2 = ln(VWAP_segment_3 / VWAP_segment_2)
r_3 = ln(VWAP_segment_3 / VWAP_segment_1)
T   = |r_1 + r_2| / (|r_1| + |r_2|)
pressure_total = pressure_segment_1 + pressure_segment_2 + pressure_segment_3
F_concentration = pressure_segment_3 / pressure_total
RV = Σ_i (p_i - p_(i-1))²
```

`RV` здесь является суммой квадратов изменений цены сделки, а не
классической realized variance по лог-доходностям.

### `price_pressure`

```text
VWAP = Σ(p_i q_i) / Σq_i
delta_p_i = p_i - VWAP
PI_i = s_i q_i delta_p_i
PI_buy  = Σ PI_i для покупок
PI_sell = Σ PI_i для продаж
PI_total = PI_buy + PI_sell
F_PI = PI_total / (RV × Σq_i)
F_eff = (C - VWAP) / |V_buy_base - V_sell_base|
F_asymmetry = (PI_buy - |PI_sell|) / (PI_buy + |PI_sell|)
VWAP_pos = (C - VWAP) / (H - L)
```

В этой реализации `RV = Σ(p_i - p_(i-1))²`.

### `trade_distribution`

Формулы считаются отдельно для buy и sell сделок одной минуты. Пусть
`p_i = q_i / Σq_i`, `n` — число сделок, `k = max(1, floor(0.2n))`.

```text
Cp     = сумма объёма k крупнейших сделок / общий объём
H      = -Σ p_i ln(p_i)
H_norm = H / ln(n)
N_eff  = 1 / Σ p_i²
C_eff  = N_eff / n
```

### `btc_features`

Поля: `btc_log_return`, `btc_volatility`, `btc_volume_log`,
`btc_quote_volume_log`, `btc_zscore`, `btc_rolling_volatility`.

В доступных файлах проекта точный генератор не найден. До его обнаружения
эти названия нельзя считать достаточной спецификацией формул и окон.

## Таргеты

### `features/return_Nm_forward`

Для `N ∈ {1, 15, 20, 25}`:

```text
return_Nm(t) = C(t + N минут) / C(t) - 1
```

Строка сохраняется только при наличии свечи с точным временем `t + N`.

### `targets/future_returns`

Поля `return_1m` ... `return_120m`. По используемой семантике:

```text
return_hm(t) = C(t + h минут) / C(t) - 1,  h = 1..120
```

Рекомендуется считать этот набор каноническим источником forward returns, а
четыре каталога в `features/` — устаревшими производными.

## Модельные витрины в корне

В корне лежат два файла общим размером 1.65 GiB:

- `adausdt_minute_log_return_dataset.parquet`;
- `adausdt_minute_log_return_dataset.before_price_pressure_fix.parquet`.

Они объединяют таргет и признаки с префиксами групп. Суффикс
`before_price_pressure_fix` означает устаревшую версию.

Такие файлы лучше хранить под:

```text
datasets/model_input/<dataset_name>/version=<version>/data.parquet
```

## Рекомендуемый порядок

1. Оставить слои `raw/`, `features/`, `targets/`, добавить `datasets/` и
   `archive/`.
2. Перенести forward returns из `features/` в `targets/` или удалить после
   проверки эквивалентности `targets/future_returns`.
3. Переименовать `sinthetic_data` в `synthetic_data` через copy + проверку +
   удаление старого префикса.
4. Перенести корневые parquet в `datasets/model_input/` с явной версией.
5. Сверить `raw_backup/` с `raw/`; подтверждённые дубли перевести в
   lifecycle-архив или удалить.
6. Зафиксировать, является ли неполный `raw/trades/BTCUSDT` ожидаемым.
7. Добавить для каждого производного набора `metadata.json` с полями:
   `owner`, `description`, `grain`, `primary_key`, `inputs`, `formula_version`,
   `code_source`, `created_at`, `quality_checks`.

## Воспроизводимая инвентаризация

Скрипт `s3_inventory.py` только читает S3 и создаёт локальные
`s3_catalog/inventory.json` и `s3_catalog/inventory.csv`:

```powershell
.\.venv\Scripts\python.exe .\s3_inventory.py
```
