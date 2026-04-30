"""
Tests for the text extractor and validator.

Covers:
  - TextExtractor.load_data     (.txt and .pdf formats)
  - TextExtractor.extract       (word count, language detection, feature keys)
  - TextValidator.check_completeness  (empty doc, full doc)
  - TextValidator.check_accuracy      (uppercase deduction)
  - TextValidator.check_uniqueness    (duplicate detection)
  - Full pipeline               (txt file → extract → validate → QualityResult)

All file-system operations use pytest's ``tmp_path`` fixture so nothing is
written to the project tree.  Tests that invoke the validator read
``dq_config.yaml`` from the project root via :func:`~src.utils.load_config`;
run pytest from the repository root (``pytest`` or ``python -m pytest``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.models import QualityResult
from src.modules.text.extractor import TextExtractor
from src.modules.text.validator import TextValidator
from src.utils import load_config

# ---------------------------------------------------------------------------
# Module-level sample strings
# ---------------------------------------------------------------------------

# Exactly 50 words when split on whitespace — used by feature extraction tests.
_FIFTY_WORDS: str = (
    "The annual report for the financial year two thousand and twenty-five shows "
    "strong growth across all major divisions. Revenue increased by fifteen percent "
    "compared to the previous year. Operating costs remained stable despite "
    "inflationary pressures in the supply chain. The board approved a new strategic "
    "plan for the coming decade."
)

# ~100 words — used as a realistic .txt payload.
_HUNDRED_WORDS: str = (
    "The annual report for the financial year two thousand and twenty-five shows "
    "strong growth across all major divisions. Revenue increased by fifteen percent "
    "compared to the previous year. Operating costs remained stable despite "
    "inflationary pressures in the supply chain. The board approved a new strategic "
    "plan for the coming decade. Our team in Lagos delivered exceptional results "
    "while the Abuja office expanded its client base significantly. The fintech "
    "division processed over one million transactions in the fourth quarter alone, "
    "demonstrating the strength of our digital payment infrastructure across Nigeria."
)

# ~200 words — used by the full-pipeline integration test.
_TWO_HUNDRED_WORDS: str = (
    "Dangote Group published its annual sustainability report for the fiscal year "
    "ending December two thousand and twenty-five. Chief Executive Funke Adeyemi "
    "highlighted record performance across cement, food processing, and logistics. "
    "Revenue grew by eighteen percent year-on-year, reaching forty-two billion naira. "
    "The cement division, led by Chidi Nwachukwu, contributed sixty percent of "
    "consolidated revenue and expanded production capacity at the Lagos and Kano "
    "facilities. The food processing segment recorded a twelve percent uplift in "
    "volumes across northern states, supported by improved distribution networks "
    "in Port Harcourt and Enugu. Capital expenditure of three point five billion "
    "naira was approved for infrastructure upgrades in the second half of the year. "
    "Partnerships with Access Bank and Zenith Bank ensured working capital "
    "requirements were met without disruption. The fintech subsidiary, operating in "
    "collaboration with MTN Nigeria, launched a new payment integration that processed "
    "over seven hundred million naira in digital transactions. Human Resources "
    "recruited four hundred and twenty additional staff across all departments. "
    "Looking ahead, the board expressed confidence in achieving continued growth "
    "targets through disciplined execution and investment in people and technology."
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> dict[str, Any]:
    """Return the pipeline configuration loaded from ``dq_config.yaml``."""
    return load_config()


@pytest.fixture
def extractor(config: dict[str, Any]) -> TextExtractor:
    """Return a fresh :class:`~TextExtractor` instance."""
    return TextExtractor(config)


@pytest.fixture
def validator(config: dict[str, Any]) -> TextValidator:
    """Return a fresh :class:`~TextValidator` instance."""
    return TextValidator(config)


@pytest.fixture
def txt_file(tmp_path: Path) -> Path:
    """Write :data:`_HUNDRED_WORDS` to a temporary ``.txt`` file."""
    path = tmp_path / "sample.txt"
    path.write_text(_HUNDRED_WORDS, encoding="utf-8")
    return path


@pytest.fixture
def pdf_file(tmp_path: Path) -> Path:
    """Create a minimal single-page PDF and return its path.

    Uses PyMuPDF (``fitz``) to construct the PDF in memory so the test is
    self-contained and does not rely on any pre-existing fixture file.
    """
    import fitz  # noqa: PLC0415

    pdf_path = tmp_path / "sample.pdf"
    doc  = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text(
        (72, 72),
        "This is a sample PDF document created for automated testing purposes.",
        fontname="helv",
        fontsize=11,
    )
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def base_features() -> dict[str, Any]:
    """Return a minimal, passing features dict for validator unit tests.

    Values are chosen to avoid triggering any quality check failures, so each
    test can selectively override only the field under examination.
    """
    return {
        "dataset_name":        "test_doc",
        "char_count":          500,
        "word_count":          80,
        "sentence_count":      5,
        "avg_word_length":     4.5,
        "language":            "en",
        "language_confidence": 0.99,
        "special_char_ratio":  0.05,
        "whitespace_ratio":    0.15,
        "uppercase_ratio":     0.08,
        "entities":            {"PERSON": 1},
        "is_empty":            False,
        "encoding_errors":     False,
        "extracted_at":        "2026-04-30T00:00:00+00:00",
    }


# ===========================================================================
# TextExtractor — load_data
# ===========================================================================


def test_extractor_loads_txt(
    extractor: TextExtractor,
    txt_file: Path,
) -> None:
    """load_data on a .txt file returns a non-empty string and type ``"txt"``.

    Verifies that:

    * The returned text is a non-empty ``str``.
    * The detected type string is exactly ``"txt"``.
    * The content contains at least some of the original text (smoke-check).
    """
    raw_text, detected_type = extractor.load_data(str(txt_file))

    assert isinstance(raw_text, str), "load_data should return a str as first element"
    assert len(raw_text) > 0,         "returned text must not be empty"
    assert detected_type == "txt",    f"expected type 'txt', got '{detected_type}'"
    assert "financial" in raw_text,   "content sanity check failed"


def test_extractor_loads_pdf(
    extractor: TextExtractor,
    pdf_file: Path,
) -> None:
    """load_data on a .pdf file returns non-empty extracted text and type ``"pdf"``.

    Verifies that:

    * PyMuPDF correctly extracts text from the PDF page.
    * The returned text is a non-empty string.
    * The detected type is ``"pdf"``.
    * A known word from the page content appears in the extracted text.
    """
    raw_text, detected_type = extractor.load_data(str(pdf_file))

    assert isinstance(raw_text, str), "load_data should return a str as first element"
    assert len(raw_text) > 0,         "extracted PDF text must not be empty"
    assert detected_type == "pdf",    f"expected type 'pdf', got '{detected_type}'"
    assert "sample" in raw_text.lower(), "expected 'sample' to appear in extracted PDF text"


# ===========================================================================
# TextExtractor — extract
# ===========================================================================


def test_extract_features(extractor: TextExtractor) -> None:
    """extract() on a 50-word string returns correct word_count and basic keys.

    Uses :data:`_FIFTY_WORDS`, which contains exactly 50 whitespace-delimited
    tokens.  Verifies:

    * ``word_count == 50``.
    * ``is_empty == False`` (50 >= 5 threshold).
    * ``char_count > 0``.
    * All required keys are present in the returned dict.
    """
    assert len(_FIFTY_WORDS.split()) == 50, "test constant must be exactly 50 words"

    features = extractor.extract(_FIFTY_WORDS, dataset_name="fifty_words")

    assert features["word_count"] == 50,   f"expected word_count=50, got {features['word_count']}"
    assert features["is_empty"]   is False, "50-word doc should not be marked is_empty"
    assert features["char_count"] > 0,      "char_count must be positive"

    required_keys = {
        "char_count", "word_count", "sentence_count", "avg_word_length",
        "language", "language_confidence", "special_char_ratio",
        "whitespace_ratio", "uppercase_ratio", "entities",
        "is_empty", "encoding_errors", "extracted_at",
    }
    assert required_keys.issubset(features.keys()), (
        f"Missing keys: {required_keys - set(features.keys())}"
    )


def test_extract_language_detection(extractor: TextExtractor) -> None:
    """extract() detects English as the dominant language of a clear English passage.

    Uses a plain English sentence to give ``langdetect`` a high-confidence
    signal.  Asserts that ``language == "en"``; does not assert on
    ``language_confidence`` because detection probability varies slightly
    across platforms.
    """
    english_text = (
        "The quick brown fox jumps over the lazy dog. "
        "This sentence is written entirely in the English language and should "
        "be detected reliably by any modern language detection library."
    )

    features = extractor.extract(english_text, dataset_name="lang_test")

    assert features["language"] == "en", (
        f"Expected language 'en', got '{features['language']}'"
    )


# ===========================================================================
# TextValidator — check_completeness
# ===========================================================================


def test_validator_completeness_empty(
    validator: TextValidator,
    base_features: dict[str, Any],
) -> None:
    """check_completeness returns score 0.0 when ``is_empty`` is True.

    An empty document (``word_count < 5``) is the most severe completeness
    failure.  The score must be exactly 0 and the failed_checks list must
    contain at least one entry whose ``"check"`` key is ``"is_empty"``.
    """
    base_features["is_empty"]   = True
    base_features["word_count"] = 0

    score, failed_checks = validator.check_completeness(base_features)

    assert score == 0.0, f"expected score 0.0 for empty doc, got {score}"
    is_empty_flags = [f for f in failed_checks if f["check"] == "is_empty"]
    assert len(is_empty_flags) >= 1, (
        "expected at least one 'is_empty' failure in failed_checks"
    )


def test_validator_completeness_full(
    validator: TextValidator,
    base_features: dict[str, Any],
) -> None:
    """check_completeness returns score 100.0 when ``word_count`` >= 50.

    A document with 100 words is well above the completeness threshold and
    should produce a perfect score with no failures.
    """
    base_features["is_empty"]   = False
    base_features["word_count"] = 100

    score, failed_checks = validator.check_completeness(base_features)

    assert score == pytest.approx(100.0), (
        f"expected score 100.0 for 100-word doc, got {score}"
    )
    assert failed_checks == [], (
        f"expected no failures for 100-word doc, got {failed_checks}"
    )


# ===========================================================================
# TextValidator — check_accuracy
# ===========================================================================


def test_validator_accuracy_uppercase(
    validator: TextValidator,
    base_features: dict[str, Any],
) -> None:
    """check_accuracy deducts points and flags a failure when uppercase_ratio > 0.5.

    Sets ``uppercase_ratio=0.8`` (well above the 0.5 threshold) while keeping
    other accuracy inputs clean.  Verifies:

    * The returned score is less than 100.
    * The ``failed_checks`` list is non-empty.
    * At least one failure has ``"check" == "excessive_uppercase"``.
    """
    base_features["uppercase_ratio"]   = 0.8
    base_features["encoding_errors"]   = False
    base_features["special_char_ratio"] = 0.05

    score, failed_checks = validator.check_accuracy(base_features)

    assert score < 100.0, f"expected score < 100 with uppercase_ratio=0.8, got {score}"
    assert len(failed_checks) > 0, "expected at least one accuracy failure"

    uppercase_flags = [f for f in failed_checks if f["check"] == "excessive_uppercase"]
    assert len(uppercase_flags) >= 1, (
        "expected an 'excessive_uppercase' entry in failed_checks"
    )


# ===========================================================================
# TextValidator — check_uniqueness
# ===========================================================================


def test_validator_uniqueness_duplicate(
    validator: TextValidator,
    base_features: dict[str, Any],
) -> None:
    """check_uniqueness returns score 0.0 when another doc shares identical metrics.

    Constructs two feature dicts with the same ``char_count`` and
    ``word_count`` but different ``dataset_name`` values (so self-comparison
    is excluded).  The check must:

    * Return score ``0.0`` for the current document.
    * Include a ``"likely_duplicate"`` entry in ``failed_checks``.
    """
    other_features = dict(base_features)
    other_features["dataset_name"] = "other_doc"
    # Both documents share char_count=500, word_count=80 (from base_features)
    assert base_features["char_count"] == other_features["char_count"]
    assert base_features["word_count"] == other_features["word_count"]

    all_features = [base_features, other_features]

    score, failed_checks = validator.check_uniqueness(base_features, all_features)

    assert score == 0.0, f"expected score 0.0 for duplicate doc, got {score}"
    dup_flags = [f for f in failed_checks if f["check"] == "likely_duplicate"]
    assert len(dup_flags) >= 1, (
        "expected a 'likely_duplicate' entry in failed_checks"
    )


# ===========================================================================
# Full pipeline integration test
# ===========================================================================


def test_full_text_pipeline(
    extractor: TextExtractor,
    validator: TextValidator,
    tmp_path: Path,
) -> None:
    """The full extract → validate chain produces a valid QualityResult.

    Writes :data:`_TWO_HUNDRED_WORDS` (~200 words) to a temporary ``.txt``
    file, runs the extractor, then runs the validator.  Verifies that:

    * The result is a :class:`~src.models.QualityResult` instance.
    * ``data_type`` is ``"text"``.
    * ``overall_score`` is in the range ``[0, 100]``.
    * All six quality dimensions are present in ``result.dimensions``.
    * The result is JSON-serialisable via ``to_dict()``.
    """
    txt_path = tmp_path / "report.txt"
    txt_path.write_text(_TWO_HUNDRED_WORDS, encoding="utf-8")

    raw_text, _ = extractor.load_data(str(txt_path))
    features    = extractor.extract(raw_text, dataset_name="report")

    result = validator.validate(
        features,
        dataset_name="report",
        source_path=str(txt_path),
        all_features=[features],
    )

    assert isinstance(result, QualityResult), (
        f"validate() should return a QualityResult, got {type(result)}"
    )
    assert result.data_type == "text", (
        f"expected data_type='text', got '{result.data_type}'"
    )
    assert 0.0 <= result.overall_score <= 100.0, (
        f"overall_score {result.overall_score} is outside [0, 100]"
    )

    expected_dims = {
        "completeness", "accuracy", "consistency",
        "freshness", "uniqueness", "validity",
    }
    assert set(result.dimensions.keys()) == expected_dims, (
        f"Missing dimensions: {expected_dims - set(result.dimensions.keys())}"
    )

    import json
    payload = result.to_dict()
    json.dumps(payload)  # must not raise TypeError
