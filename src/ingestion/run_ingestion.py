"""
Ingestion stage entry-point.

Generates sample datasets — structured (CSV), semi-structured (JSON, JSONL, log),
and text (TXT, PDF, CSV) — and writes them to ``data/raw/``.  The files serve as
the pipeline's canonical input fixtures for CI runs and local development.

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

Text samples
------------
* ``data/raw/sample_report.txt`` — ~300-word business report with named
  entities and one ALL-CAPS section to trigger the uppercase accuracy check.
* ``data/raw/sample_document.pdf`` — 2-page PDF; page 1 has a title and three
  body paragraphs; page 2 has only ten words of sparse content.
* ``data/raw/sample_comments.csv`` — 50 social-media comment rows with
  intentional issues: 5 very short comments, 3 all-special-character comments,
  and 2 empty ``comment_text`` values.

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
# Text samples
# ---------------------------------------------------------------------------


def generate_report_txt() -> None:
    """Write ``data/raw/sample_report.txt`` — a ~300-word business report.

    The document includes:

    * Named entities: company names (Dangote Group, Access Bank, Zenith Bank,
      MTN Nigeria), people (Emeka Okafor, Funke Adeyemi, Chidi Nwachukwu),
      and Nigerian cities (Lagos, Abuja, Kano, Port Harcourt, Enugu).
    * An ALL-CAPS section to trigger the accuracy ``excessive_uppercase`` check.
    """
    report = """\
QUARTERLY BUSINESS REVIEW — Q1 2026

Submitted by: Emeka Okafor, Head of Operations, Dangote Group
Date: 29 April 2026, Lagos, Nigeria

Executive Summary

The first quarter of 2026 marked a period of robust expansion for Dangote Group
across its principal markets in Lagos, Abuja, and Kano. Under the strategic guidance
of Chief Executive Funke Adeyemi, the organisation successfully extended its logistics
network into Port Harcourt and Enugu, adding over two million new customer touchpoints.

Revenue Performance

Total group revenue for Q1 reached fourteen point seven billion naira, representing a
twelve per cent year-on-year increase. The cement division, directed by Chidi Nwachukwu,
contributed fifty-eight per cent of consolidated revenue. The food processing segment
recorded a seven per cent uplift in distribution volumes across northern states,
particularly in Kano and Kaduna.

Our continued partnership with Access Bank and Zenith Bank has supported working capital
requirements and ensured uninterrupted operations across regional depots. The fintech
subsidiary, operating in collaboration with MTN Nigeria, launched a payment integration
that processed over five hundred million naira in digital transactions since January.

Strategic Priorities for Q2

The board has approved capital expenditure of two point three billion naira for
infrastructure upgrades at the Lagos and Port Harcourt processing facilities. Human
Resources will recruit three hundred and fifty additional staff to support planned growth.

CRITICAL OPERATIONAL ALERT: SUPPLY CHAIN DISRUPTIONS HAVE BEEN IDENTIFIED IN THE
NORTHERN DISTRIBUTION CORRIDOR. ALL REGIONAL MANAGERS MUST SUBMIT INCIDENT REPORTS
BY END OF BUSINESS FRIDAY. COMPLIANCE IS MANDATORY ACROSS ALL DEPARTMENTS. THE
LOGISTICS TEAM IN KANO HAS BEEN FORMALLY NOTIFIED OF REQUIRED ESCALATION PROTOCOLS.
FAILURE TO RESPOND WITHIN TWENTY-FOUR HOURS WILL TRIGGER AUTOMATIC ESCALATION.

Conclusion

Despite the supply chain challenges in the north, the overall performance trajectory
for Dangote Group remains strongly positive entering the second quarter of 2026.
"""
    out_path = RAW_DIR / "sample_report.txt"
    out_path.write_text(report, encoding="utf-8")
    word_count = len(report.split())
    logger.info(
        f"generate_report_txt: wrote ~{word_count} words → {out_path}"
    )


def generate_document_pdf() -> None:
    """Write ``data/raw/sample_document.pdf`` — a 2-page PDF using PyMuPDF.

    * Page 1: title and three paragraphs of English prose (~120 words).
    * Page 2: sparse appendix section (~10 words) to demonstrate the
      contrast between rich and minimal page content.

    Requires ``PyMuPDF`` (``fitz``) to be installed.
    """
    import fitz  # noqa: PLC0415

    MARGIN_X: int = 72
    MARGIN_Y: int = 72
    RIGHT:    int = 523

    doc = fitz.open()

    # ---- Page 1: title and three body paragraphs ----
    page1 = doc.new_page(width=595, height=842)

    page1.insert_text(
        (MARGIN_X, MARGIN_Y),
        "Data Processing Technical Overview",
        fontname="helv",
        fontsize=16,
    )

    body_page1 = (
        "This document describes the core data ingestion and transformation pipeline "
        "used by the organisation to process large volumes of structured and unstructured "
        "data. The pipeline operates across multiple data centres located in Lagos and Abuja.\n\n"
        "Data arrives from upstream producers via secure API endpoints and message queues. "
        "Each batch undergoes schema validation, deduplication, and enrichment before being "
        "written to the central data warehouse. Throughput peaks at forty thousand records "
        "per second during business hours.\n\n"
        "Quality checks run automatically after each ingestion cycle. Any batch that falls "
        "below the configured pass score is quarantined for manual review by the data "
        "engineering team. Alerts are dispatched to the on-call engineer via the incident "
        "management system."
    )

    page1.insert_textbox(
        fitz.Rect(MARGIN_X, MARGIN_Y + 40, RIGHT, 770),
        body_page1,
        fontname="helv",
        fontsize=11,
    )

    # ---- Page 2: sparse appendix (≈10 words) ----
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text(
        (MARGIN_X, MARGIN_Y),
        "Appendix A: Preliminary notes only. Refer to main body.",
        fontname="helv",
        fontsize=11,
    )

    out_path = RAW_DIR / "sample_document.pdf"
    doc.save(str(out_path))
    doc.close()
    logger.info(f"generate_document_pdf: wrote 2-page PDF → {out_path}")


def generate_comments_csv() -> None:
    """Write ``data/raw/sample_comments.csv`` — 50 social-media comment rows.

    Columns: ``comment_id``, ``user_id``, ``comment_text``, ``platform``,
    ``created_at``.

    Intentional quality issues:

    * Rows at indices 10, 20, 30, 40, 45: very short comments (1–2 words),
      triggering a low word-count in the text completeness check.
    * Rows at indices 15, 25, 35: ``comment_text`` consists entirely of
      special characters, triggering the accuracy special-char-ratio check.
    * Rows at indices 47, 49: ``comment_text`` is empty.
    """
    import csv

    platforms = ["Twitter", "Facebook", "Instagram", "LinkedIn", "TikTok"]

    normal_comments = [
        "I really enjoyed this product. It has made my daily routine so much easier and I would definitely recommend it.",
        "The customer service team was incredibly helpful when I had an issue. They resolved everything within twenty-four hours.",
        "Outstanding quality. I have been using this brand for three years and have never been disappointed with any purchase.",
        "Delivery was faster than expected and the packaging was excellent. Will order again without hesitation.",
        "Good value for money. The features are well thought out and the interface is intuitive even for first-time users.",
        "I had a minor issue with my order but the support team sorted it out immediately. Very impressed with the response.",
        "Highly recommend this to anyone looking for a reliable and affordable option in this category.",
        "The product exceeded my expectations in every way. Build quality is superb and performance is consistent.",
        "Five stars from me. Prompt delivery, well packaged, and exactly as described on the website.",
        "Satisfied with my purchase. The product works as advertised and the price point is very competitive.",
        "Solid performance over the past two months. No issues to report and the battery life is surprisingly good.",
        "The app companion is easy to use and syncs reliably with the main device. Tech support was also very responsive.",
        "Would have given five stars but the instruction manual could be clearer. Otherwise a very good product.",
        "Great experience from order to delivery. The tracking updates were accurate and the courier was professional.",
        "Exactly what I needed. Simple to set up and works perfectly straight out of the box.",
        "I bought this as a gift and the recipient was very happy. Presentation and packaging are top notch.",
        "Reliable and durable. I use it every day and it shows no signs of wear after several months.",
        "The colour options available are attractive and the finish is premium. Definitely worth the price.",
        "Quick shipping, responsive support, and a product that does exactly what it promises. Very happy customer.",
        "I was initially sceptical about the price but the quality justifies every penny. Brilliant purchase.",
        "Good product but the checkout process on the website could be smoother. Delivery was on time though.",
        "Impressive performance for the price bracket. I compared several alternatives and this was clearly the best.",
        "The size is perfect and the weight is lighter than I expected. Very comfortable to use for extended periods.",
        "Excellent build quality and the design is sleek and modern. Gets lots of compliments from colleagues.",
        "Happy with this purchase. Arrived on time, well packaged, and the product quality matches the description.",
        "The features are comprehensive and the learning curve is minimal. I was up and running within minutes.",
        "Good after-sales support. They followed up proactively to make sure I was satisfied with my purchase.",
        "The product is stylish and practical. I especially appreciate the attention to detail in the design.",
        "Works exactly as described. I have recommended it to three friends already and they are all satisfied.",
        "Competitive pricing and reliable delivery. The product has held up well under daily use for two months.",
        "Sturdy construction and intuitive design. Setup took less than ten minutes and performance has been flawless.",
        "The upgrade from my previous model is significant. Much faster and the interface is cleaner.",
        "Packaging was a little bulky but the product itself is compact and well designed. Happy overall.",
        "Very pleased with this. The product is well made and the customer service team was easy to reach.",
        "Does what it says on the tin. Reliable, affordable, and backed by decent warranty support.",
        "I chose this over a competitor product based on the reviews and I am glad I did. No regrets.",
        "Prompt delivery and good communication throughout the order process. The product quality is as expected.",
        "Excellent value. I bought two units — one for home and one for the office — and both work perfectly.",
        "The design is ergonomic and the product is lightweight. Ideal for everyday carry without extra bulk.",
        "Good product, good service. Would not hesitate to order from this seller again in the future.",
        "Minor quibble about the packaging but the product itself is great. Arrived in perfect condition.",
        "I have been using this for six weeks now. Battery holds charge well and the performance is consistent.",
        "Straightforward to use and the results have been reliable. Exactly what I needed for my workflow.",
    ]

    short_comments  = ["Nice", "Good", "Okay", "Bad", "Hmm"]
    special_comments = ["!@#$%^&*()", ">>><<<|||###", "***@@@!!!~~~"]
    empty_comment    = ""

    rows: list[list] = []
    short_idx    = {10, 20, 30, 40, 45}
    special_idx  = {15, 25, 35}
    empty_idx    = {47, 49}

    short_pool   = list(short_comments)
    special_pool = list(special_comments)
    normal_pool  = list(normal_comments)
    normal_cycle = 0

    for i in range(50):
        if i in empty_idx:
            comment_text = empty_comment
        elif i in special_idx:
            comment_text = special_pool.pop(0)
        elif i in short_idx:
            comment_text = short_pool.pop(0)
        else:
            comment_text = normal_pool[normal_cycle % len(normal_pool)]
            normal_cycle += 1

        rows.append([
            i + 1,
            random.randint(1, 1000),
            comment_text,
            random.choice(platforms),
            _iso_days_ago(random.uniform(0, 30)),
        ])

    out_path = RAW_DIR / "sample_comments.csv"
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["comment_id", "user_id", "comment_text", "platform", "created_at"])
        writer.writerows(rows)

    logger.info(f"generate_comments_csv: wrote {len(rows)} rows → {out_path}")


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

    logger.info("run_ingestion: starting text sample generation")
    generate_report_txt()
    generate_document_pdf()
    generate_comments_csv()

    logger.info("run_ingestion: all samples written to data/raw/")


if __name__ == "__main__":
    main()
