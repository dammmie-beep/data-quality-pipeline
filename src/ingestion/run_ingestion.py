"""
Ingestion stage entry-point.

Generates sample datasets — both structured (CSV) and semi-structured (JSON,
JSONL, log) — and writes them to ``data/raw/``.  The files serve as the
pipeline's canonical input fixtures for CI runs and local development.

Structured sample
-----------------
* ``data/raw/sample_orders.csv`` — 100 order rows with intentional data
  quality issues (nulls, duplicates, out-of-range prices).

Semi-structured samples
-----------------------
* ``data/raw/sample_users.json`` — 50 user records (nested address and
  preferences) with 5 missing emails, 3 duplicate records, and 2 records
  carrying an unexpected ``temp_debug_field``.
* ``data/raw/sample_events.jsonl`` — 50 event records, one per line, with 4
  null statuses and some sparse payload fields.
* ``data/raw/sample_app.log`` — 30 JSON log lines with 5 missing
  ``request_id`` values and 2 raw non-JSON lines.

Run directly::

    python src/ingestion/run_ingestion.py
"""

from __future__ import annotations

import json
import random
import string
from datetime import datetime, timedelta, timezone
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_DIR = Path("data/raw")

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _random_string(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase, k=length))


def _iso_days_ago(days: float) -> str:
    ts = datetime.now(timezone.utc) - timedelta(days=days)
    return ts.isoformat()


# ---------------------------------------------------------------------------
# Structured sample
# ---------------------------------------------------------------------------


def generate_structured_sample() -> None:
    """Write ``data/raw/sample_orders.csv`` with intentional quality issues.

    The CSV contains 100 rows of order data.  Issues introduced:

    * Rows 10, 25, 40: ``price`` is ``-1`` (invalid negative value).
    * Rows 5, 15, 30, 60: ``customer_email`` is empty.
    * Rows 0–4 are repeated as rows 95–99 (duplicate records).
    * Row 70: ``status`` is ``"UNKNOWN"`` (outside the known value set).
    """
    import csv

    out_path = RAW_DIR / "sample_orders.csv"
    headers = [
        "order_id", "customer_email", "product_name",
        "price", "quantity", "status", "created_at",
    ]

    statuses = ["pending", "shipped", "delivered", "cancelled"]
    rows: list[list] = []

    for i in range(95):
        price = round(random.uniform(5.0, 500.0), 2)
        if i in {10, 25, 40}:
            price = -1.0
        email = f"user{i}@example.com"
        if i in {5, 15, 30, 60}:
            email = ""
        status = random.choice(statuses)
        if i == 70:
            status = "UNKNOWN"
        rows.append([
            f"ORD-{i:05d}",
            email,
            f"Product-{_random_string(4).upper()}",
            price,
            random.randint(1, 20),
            status,
            _iso_days_ago(random.uniform(0, 60)),
        ])

    # Rows 95-99 duplicate rows 0-4
    for i in range(5):
        rows.append(list(rows[i]))

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)

    logger.info(f"generate_structured_sample: wrote {len(rows)} rows → {out_path}")


# ---------------------------------------------------------------------------
# Semi-structured samples
# ---------------------------------------------------------------------------


def generate_users_json() -> None:
    """Write ``data/raw/sample_users.json`` — 50 user records.

    Intentional quality issues:

    * Records 3, 12, 22, 33, 44: ``email`` key is omitted.
    * Records 47, 48, 49: exact duplicates of records 0, 1, 2.
    * Records 10, 20: contain an extra ``temp_debug_field`` key.
    """
    themes = ["light", "dark", "system"]
    countries = ["US", "UK", "CA", "AU", "DE"]

    users: list[dict] = []

    for i in range(47):
        record: dict = {
            "id": i + 1,
            "name": f"User {i + 1}",
            "address": {
                "street": f"{random.randint(1, 999)} Main St",
                "city": f"City{i % 10}",
                "country": random.choice(countries),
            },
            "preferences": {
                "theme": random.choice(themes),
                "notifications": random.choice([True, False]),
            },
            "created_at": _iso_days_ago(random.uniform(1, 365)),
            "is_active": random.choice([True, False]),
        }

        # 5 records missing email (indices 3, 12, 22, 33, 44)
        if i not in {3, 12, 22, 33, 44}:
            record["email"] = f"user{i + 1}@example.com"

        # 2 records with unexpected field
        if i in {10, 20}:
            record["temp_debug_field"] = f"debug_{_random_string(6)}"

        users.append(record)

    # 3 duplicate records (indices 47-49 duplicate 0-2)
    for i in range(3):
        users.append(dict(users[i]))

    out_path = RAW_DIR / "sample_users.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(users, fh, indent=2)

    logger.info(f"generate_users_json: wrote {len(users)} records → {out_path}")


def generate_events_jsonl() -> None:
    """Write ``data/raw/sample_events.jsonl`` — 50 event records (one per line).

    Intentional quality issues:

    * Records 5, 17, 29, 41: ``status`` is ``null``.
    * The ``payload`` dict is sparse — some fields are absent on a per-record
      basis to simulate a loosely-schemaed event stream.
    """
    event_types = ["click", "page_view", "purchase", "signup", "logout"]
    payload_keys = ["page", "item_id", "amount", "referrer", "session_id"]

    out_path = RAW_DIR / "sample_events.jsonl"
    null_status_indices = {5, 17, 29, 41}

    with out_path.open("w", encoding="utf-8") as fh:
        for i in range(50):
            # Build a sparse payload — randomly include 2-4 of the 5 keys
            n_keys = random.randint(2, 4)
            chosen = random.sample(payload_keys, n_keys)
            payload = {k: _random_string(6) for k in chosen}

            record = {
                "event_id": f"EVT-{i:05d}",
                "event_type": random.choice(event_types),
                "user_id": random.randint(1, 200),
                "timestamp": _iso_days_ago(random.uniform(0, 30)),
                "payload": payload,
                "status": None if i in null_status_indices else random.choice(
                    ["success", "failure", "pending"]
                ),
            }
            fh.write(json.dumps(record) + "\n")

    logger.info(f"generate_events_jsonl: wrote 50 records → {out_path}")


def generate_app_log() -> None:
    """Write ``data/raw/sample_app.log`` — 30 log lines.

    Intentional quality issues:

    * Lines 4, 10, 16, 22, 28: ``request_id`` key is omitted.
    * Lines 7, 14: raw non-JSON text (simulates log corruption or a
      non-structured logger writing to the same file).
    """
    levels = ["INFO", "DEBUG", "WARNING", "ERROR"]
    services = ["auth-service", "order-service", "payment-service", "api-gateway"]
    non_json_lines = {7, 14}
    missing_request_id = {4, 10, 16, 22, 28}

    out_path = RAW_DIR / "sample_app.log"

    with out_path.open("w", encoding="utf-8") as fh:
        for i in range(30):
            if i in non_json_lines:
                # Raw plain-text line — not valid JSON
                fh.write(f"[WARN] Unexpected restart detected at line {i}\n")
                continue

            record: dict = {
                "timestamp": _iso_days_ago(random.uniform(0, 7)),
                "level": random.choice(levels),
                "message": f"Processed request #{i} successfully",
                "service": random.choice(services),
            }

            if i not in missing_request_id:
                record["request_id"] = f"req-{_random_string(12)}"

            fh.write(json.dumps(record) + "\n")

    logger.info(f"generate_app_log: wrote 30 lines → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate all sample datasets and write them to ``data/raw/``."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("run_ingestion: starting structured sample generation")
    generate_structured_sample()

    logger.info("run_ingestion: starting semi-structured sample generation")
    generate_users_json()
    generate_events_jsonl()
    generate_app_log()

    logger.info("run_ingestion: all samples written to data/raw/")


if __name__ == "__main__":
    main()
