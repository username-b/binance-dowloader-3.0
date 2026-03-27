import datetime as dt
import subprocess
import yaml
import copy
import tempfile
from pathlib import Path
import boto3
import os
from dotenv import load_dotenv
load_dotenv()

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

    print(" ".join(cmd))
    return subprocess.Popen(cmd)


# =========================
# S3 VALIDATION
# =========================
def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("YC_ENDPOINT"),
        region_name=os.getenv("YC_REGION"),
        aws_access_key_id=os.getenv("YC_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("YC_SECRET_ACCESS_KEY"),
    )


def list_existing_dates(s3, bucket, prefix):
    paginator = s3.get_paginator("list_objects_v2")
    dates = set()

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            for part in key.split("/"):
                if part.startswith("date="):
                    dates.add(part.replace("date=", ""))

    return dates


def generate_expected_dates(start, end):
    dates = set()
    current = start

    while current <= end:
        dates.add(current.strftime("%Y-%m-%d"))
        current += dt.timedelta(days=1)

    return dates


# =========================
# BACKFILL
# =========================
def run_backfill(cfg, missing_dates, mount_dir):
    print(f"\n🔥 BACKFILL: {len(missing_dates)} missing dates")

    processes = []
    temp_files = []

    for d in missing_dates:
        new_cfg = copy.deepcopy(cfg)

        new_cfg["date_range"]["start"] = d
        new_cfg["date_range"]["end"] = d

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

        p = run_container(config_name, mount_dir)
        processes.append(p)

    for p in processes:
        p.wait()

    for f in temp_files:
        try:
            Path(f).unlink()
        except:
            pass


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

    n = 5
    ranges = split_date_range(start, end, n)

    processes = []
    temp_files = []

    mount_dir = Path.cwd().as_posix()

    # =========================
    # BULK LOAD
    # =========================
    for i, (s, e) in enumerate(ranges):
        new_cfg = copy.deepcopy(cfg)

        new_cfg["date_range"]["start"] = s
        new_cfg["date_range"]["end"] = e

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

    for p in processes:
        p.wait()

    # =========================
    # CLEANUP TEMP
    # =========================
    for f in temp_files:
        try:
            Path(f).unlink()
        except:
            pass

    # =========================
    # VALIDATION
    # =========================
    print("\n🔍 VALIDATION...")

    s3 = get_s3_client()
    bucket = os.getenv("YC_BUCKET")

    symbol = cfg["symbols"][0]
    interval = cfg["interval"]

    prefix = f"raw/klines/symbol={symbol}/interval={interval}/"

    existing = list_existing_dates(s3, bucket, prefix)
    expected = generate_expected_dates(start, end)

    missing = sorted(expected - existing)

    print(f"Expected: {len(expected)}")
    print(f"Existing: {len(existing)}")
    print(f"Missing: {len(missing)}")

    # =========================
    # BACKFILL
    # =========================
    if missing:
        run_backfill(cfg, missing, mount_dir)
    else:
        print("✅ ALL DATA LOADED")


if __name__ == "__main__":
    main()