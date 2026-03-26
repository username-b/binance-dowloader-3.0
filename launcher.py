import datetime as dt
import subprocess
import yaml
import copy
import os


def split_date_range(start_date, end_date, n):
    total_days = (end_date - start_date).days + 1
    chunk_size = total_days // n
    remainder = total_days % n

    ranges = []
    current = start_date

    for i in range(n):
        extra = 1 if i < remainder else 0
        end = current + dt.timedelta(days=chunk_size + extra - 1)

        ranges.append((current, end))
        current = end + dt.timedelta(days=1)

    return ranges


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(cfg, path):
    with open(path, "w") as f:
        yaml.dump(cfg, f)


def run_container(config_path):
    cmd = [
        "docker",
        "run",
        "--env-file",
        ".env",
        "-v",
        f"{os.getcwd()}:/app",
        "binance-downloader",
        "python",
        "run.py",
        "--config",
        f"/app/{config_path}",
    ]

    return subprocess.Popen(cmd)


def main():
    base_config_path = "config.yaml"
    cfg = load_config(base_config_path)

    start = cfg["date_range"]["start"]
    end = cfg["date_range"]["end"]

    if isinstance(start, str):
        start = dt.datetime.strptime(start, "%Y-%m-%d").date()
    if isinstance(end, str):
        end = dt.datetime.strptime(end, "%Y-%m-%d").date()

    n = 20  # сколько контейнеров

    ranges = split_date_range(start, end, n)

    processes = []

    for i, (s, e) in enumerate(ranges):
        new_cfg = copy.deepcopy(cfg)

        new_cfg["date_range"]["start"] = s
        new_cfg["date_range"]["end"] = e

        config_name = f"config_{i}.yaml"
        save_config(new_cfg, config_name)

        print(f"Shard {i}: {s} → {e}")

        p = run_container(config_name)
        processes.append(p)

    # ждём завершения
    for p in processes:
        p.wait()


if __name__ == "__main__":
    main()