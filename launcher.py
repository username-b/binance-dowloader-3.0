import datetime as dt
import subprocess
import yaml
import copy
import tempfile
from pathlib import Path


# =========================
# SPLIT RANGE
# =========================
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


# =========================
# CONFIG
# =========================
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# =========================
# DOCKER RUN
# =========================
def run_container(config_path, mount_dir):
    cmd = [
        "docker",
        "run",
        "--env-file",
        ".env",
        "-v",
        f"{mount_dir}:/app",
        "binance-downloader",
        "python",
        "run.py",
        "--config",
        f"/app/{config_path}",
    ]

    print(" ".join(cmd))  # 🔥 полезно для дебага
    return subprocess.Popen(cmd)


# =========================
# MAIN
# =========================
def main():
    base_config_path = "config.yaml"
    cfg = load_config(base_config_path)

    start = cfg["date_range"]["start"]
    end = cfg["date_range"]["end"]

    if isinstance(start, str):
        start = dt.datetime.strptime(start, "%Y-%m-%d").date()
    if isinstance(end, str):
        end = dt.datetime.strptime(end, "%Y-%m-%d").date()

    # 🔥 сколько контейнеров
    n = 5

    ranges = split_date_range(start, end, n)

    processes = []
    temp_files = []

    # 🔥 нормальный путь (фикс бага)
    mount_dir = Path.cwd().as_posix()

    for i, (s, e) in enumerate(ranges):
        new_cfg = copy.deepcopy(cfg)

        new_cfg["date_range"]["start"] = s
        new_cfg["date_range"]["end"] = e

        # =========================
        # TEMP FILE
        # =========================
        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
            dir="."
        )

        yaml.dump(new_cfg, tmp)
        tmp.close()

        config_name = Path(tmp.name).name
        temp_files.append(tmp.name)

        print(f"Shard {i}: {s} → {e}")

        p = run_container(config_name, mount_dir)
        processes.append(p)

    # =========================
    # WAIT
    # =========================
    for p in processes:
        p.wait()

    # =========================
    # CLEANUP
    # =========================
    for f in temp_files:
        try:
            Path(f).unlink()
        except Exception as e:
            print(f"Failed to delete {f}: {e}")


if __name__ == "__main__":
    main()