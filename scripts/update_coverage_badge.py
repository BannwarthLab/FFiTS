#!/usr/bin/env python3
"""Update the coverage badge in README.md from a coverage.xml report.

Usage:
    pytest tests/ --cov=ffits --cov-report=xml
    python scripts/update_coverage_badge.py

Reads the Cobertura-style coverage.xml produced by pytest-cov and rewrites
the badge line between the COVERAGE-BADGE markers in README.md.
"""
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COVERAGE_XML = ROOT / "coverage.xml"
README = ROOT / "README.md"

BADGE_START = "<!-- COVERAGE-BADGE-START -->"
BADGE_END = "<!-- COVERAGE-BADGE-END -->"


def get_coverage_percent(xml_path: Path) -> float:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    line_rate = float(root.attrib["line-rate"])
    return round(line_rate * 100, 1)


def badge_color(pct: float) -> str:
    if pct >= 90:
        return "brightgreen"
    if pct >= 75:
        return "green"
    if pct >= 60:
        return "yellow"
    if pct >= 40:
        return "orange"
    return "red"


def badge_markdown(pct: float) -> str:
    color = badge_color(pct)
    label = f"{pct:g}%25"
    return f"![coverage](https://img.shields.io/badge/coverage-{label}-{color})"


def update_readme(pct: float) -> None:
    text = README.read_text()
    badge_block = f"{BADGE_START}\n{badge_markdown(pct)}\n{BADGE_END}"

    if BADGE_START in text and BADGE_END in text:
        text = re.sub(
            re.escape(BADGE_START) + r".*?" + re.escape(BADGE_END),
            badge_block,
            text,
            flags=re.DOTALL,
        )
    else:
        # No markers yet: insert the badge block right after the title.
        text = text.replace("# FFiTS\n", f"# FFiTS\n\n{badge_block}\n", 1)

    README.write_text(text)


def main() -> int:
    if not COVERAGE_XML.exists():
        print(
            f"Error: {COVERAGE_XML} not found. Run "
            "'pytest tests/ --cov=ffits --cov-report=xml' first "
            "(e.g. via 'make coverage').",
            file=sys.stderr,
        )
        return 1

    pct = get_coverage_percent(COVERAGE_XML)
    update_readme(pct)
    print(f"Updated coverage badge in README.md: {pct:g}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
