#!/usr/bin/env python3
"""Update the badge row in README.md from coverage.xml, pyproject.toml and
the CI matrix in .gitlab/.gitlab-ci.yml.

Usage:
    pytest tests/ --cov=ffits --cov-report=xml
    python scripts/update_coverage_badge.py

Generates a License / Python (tested versions) / Coverage badge row and
rewrites it between the BADGES markers in README.md.
"""

import re
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib
from pathlib import Path
from urllib.parse import quote
import xml.etree.ElementTree as ET

import yaml

ROOT = Path(__file__).resolve().parent.parent
COVERAGE_XML = ROOT / "coverage.xml"
PYPROJECT = ROOT / "pyproject.toml"
CI_CONFIG = ROOT / ".gitlab" / ".gitlab-ci.yml"
README = ROOT / "README.md"

BADGE_START = "<!-- BADGES-START -->"
BADGE_END = "<!-- BADGES-END -->"


def get_coverage_percent(xml_path: Path) -> float:
    root = ET.parse(xml_path).getroot()
    return round(float(root.attrib["line-rate"]) * 100, 1)


def coverage_color(pct: float) -> str:
    if pct >= 90:
        return "brightgreen"
    if pct >= 75:
        return "green"
    if pct >= 60:
        return "yellow"
    if pct >= 40:
        return "orange"
    return "red"


def get_license(pyproject_path: Path) -> str:
    data = tomllib.loads(pyproject_path.read_text())
    license_field = data.get("project", {}).get("license", "unknown")
    if isinstance(license_field, dict):
        return license_field.get("text", "unknown")
    return str(license_field)


def get_tested_python_versions(ci_config_path: Path) -> list[str]:
    data = yaml.safe_load(ci_config_path.read_text())
    matrix = data[".matrix"]["parallel"]["matrix"]
    versions = []
    for combo in matrix:
        for entry in combo.get("PYTHON", []):
            v = entry.replace("python", "")
            if v not in versions:
                versions.append(v)
    return sorted(versions, key=lambda v: [int(p) for p in v.split(".")])


def shield(label: str, message: str, color: str) -> str:
    label_enc = quote(label, safe="")
    message_enc = quote(message, safe="")
    return f"![{label}](https://img.shields.io/badge/{label_enc}-{message_enc}-{color})"


def build_badge_row(
    license_name: str, python_versions: list[str], coverage_pct: float
) -> str:
    license_badge = shield("License", license_name, "blue")
    python_badge = shield("python", " | ".join(python_versions), "blue")
    coverage_badge = shield(
        "coverage", f"{coverage_pct:g}%", coverage_color(coverage_pct)
    )
    return " ".join([license_badge, python_badge, coverage_badge])


def update_readme(badge_line: str) -> None:
    text = README.read_text()
    block = f"{BADGE_START}\n{badge_line}\n{BADGE_END}"

    if BADGE_START in text and BADGE_END in text:
        text = re.sub(
            re.escape(BADGE_START) + r".*?" + re.escape(BADGE_END),
            block,
            text,
            flags=re.DOTALL,
        )
    else:
        text = text.replace("# FFiTS\n", f"# FFiTS\n\n{block}\n", 1)

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

    coverage_pct = get_coverage_percent(COVERAGE_XML)
    license_name = get_license(PYPROJECT)
    python_versions = get_tested_python_versions(CI_CONFIG)

    badge_line = build_badge_row(license_name, python_versions, coverage_pct)
    update_readme(badge_line)

    print(
        f"Updated README badges: License={license_name}, "
        f"Python={', '.join(python_versions)}, Coverage={coverage_pct:g}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
