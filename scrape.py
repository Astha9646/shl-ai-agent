"""
Scrape SHL product catalog — Individual Test Solutions only.
Uses requests + BeautifulSoup. Writes data/assessments.json.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://www.shl.com"
CATALOG_URL = f"{BASE}/products/product-catalog/"
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "assessments.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Fallback if tooltip markup changes
DEFAULT_TYPE_LEGEND: dict[str, str] = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def fetch(session: requests.Session, url: str) -> BeautifulSoup:
    r = session.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def parse_type_legend(soup: BeautifulSoup) -> dict[str, str]:
    legend: dict[str, str] = dict(DEFAULT_TYPE_LEGEND)
    tip = soup.find("div", id="productCatalogueTooltip")
    if not tip:
        return legend
    for li in tip.select("li.custom__tooltip-item"):
        key_el = li.find("span", class_="product-catalogue__key")
        if not key_el:
            continue
        code = key_el.get_text(strip=True)
        label = li.get_text(separator=" ", strip=True)
        if code and label.startswith(code):
            label = label[len(code) :].strip()
        if code:
            legend[code] = label
    return legend


def find_individual_table(soup: BeautifulSoup):
    for table in soup.find_all("table"):
        th = table.find("th", class_="custom__table-heading__title")
        if th and "Individual Test Solutions" in th.get_text():
            return table
    return None


def parse_catalog_rows(
    soup: BeautifulSoup, legend: dict[str, str]
) -> list[dict[str, object]]:
    table = find_individual_table(soup)
    if not table:
        return []

    rows_out: list[dict[str, object]] = []
    for tr in table.find_all("tr", attrs={"data-entity-id": True}):
        a = tr.find("td", class_="custom__table-heading__title")
        if not a:
            continue
        link = a.find("a", href=True)
        if not link:
            continue
        href = link["href"].strip()
        name = link.get_text(strip=True)
        keys_td = tr.find("td", class_=lambda c: c and "product-catalogue__keys" in c)
        codes: list[str] = []
        if keys_td:
            for span in keys_td.find_all("span", class_="product-catalogue__key"):
                c = span.get_text(strip=True)
                if c:
                    codes.append(c)
        categories = [legend.get(c, c) for c in codes]
        rows_out.append(
            {
                "assessment_name": name,
                "assessment_url": urljoin(BASE, href),
                "test_type_codes": codes,
                "test_type_categories": categories,
            }
        )
    return rows_out


def extract_description(session: requests.Session, page_url: str) -> str:
    soup = fetch(session, page_url)
    for prop in ("og:description", "twitter:description"):
        m = soup.find("meta", property=prop)
        if m and m.get("content"):
            return _clean_desc(m["content"])
    m = soup.find("meta", attrs={"name": "description"})
    if m and m.get("content"):
        return _clean_desc(m["content"])
    return ""


def _clean_desc(text: str) -> str:
    t = text.replace("\u00a0", " ").strip()
    t = re.sub(r"\s+", " ", t)
    return t


def collect_catalog_rows(session: requests.Session) -> tuple[list[dict[str, object]], dict[str, str]]:
    """Paginate catalog until no new Individual rows (table empty or offset repeats)."""
    legend: dict[str, str] = {}
    seen_urls: set[str] = set()
    all_rows: list[dict[str, object]] = []
    start = 0
    page_size = 12

    while True:
        url = f"{CATALOG_URL}?start={start}"
        soup = fetch(session, url)
        if not legend:
            legend = parse_type_legend(soup)

        batch = parse_catalog_rows(soup, legend)
        if not batch:
            break

        new_for_page: list[dict[str, object]] = []
        for item in batch:
            u = str(item["assessment_url"])
            if u not in seen_urls:
                seen_urls.add(u)
                new_for_page.append(item)

        if not new_for_page:
            break

        all_rows.extend(new_for_page)
        print(
            f"Catalog offset start={start}: +{len(new_for_page)} new "
            f"(total {len(all_rows)} assessments)",
            flush=True,
        )
        start += page_size

    return all_rows, legend


def attach_descriptions(
    session: requests.Session, rows: list[dict[str, object]], delay_s: float
) -> None:
    total = len(rows)
    for i, item in enumerate(rows, start=1):
        time.sleep(delay_s)
        desc = extract_description(session, str(item["assessment_url"]))
        item["description"] = desc
        item["test_type"] = {
            "codes": item.pop("test_type_codes"),
            "categories": item.pop("test_type_categories"),
        }
        if i == 1 or i % 25 == 0 or i == total:
            print(f"Descriptions {i}/{total}", flush=True)


def scrape_assessments(
    session: requests.Session, delay_s: float = 0.15
) -> list[dict[str, object]]:
    rows, _legend = collect_catalog_rows(session)
    attach_descriptions(session, rows, delay_s=delay_s)
    return rows


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        assessments = scrape_assessments(session)

    payload = {
        "source": CATALOG_URL,
        "section": "Individual Test Solutions",
        "count": len(assessments),
        "assessments": assessments,
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(assessments)} assessments to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
