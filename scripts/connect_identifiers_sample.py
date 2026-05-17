#!/usr/bin/env python3
"""
Connect identifiers.json entries to sample.csv listings without AI.

Matching order:
  1. Exact Yahoo ticker after MIC suffix normalization.
  2. Root ticker match before venue suffixes such as ".", "_", or "/".
  3. ISIN match using identifiers_isin.json.

The ISIN step is what connects cross-listings like TLO.GY to TSLA.
"""

import csv
import json
import os
import re

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_CSV = os.path.join(SCRIPTS_DIR, "sample.csv")
IDENTIFIERS_JSON = os.path.join(SCRIPTS_DIR, "identifiers.json")
IDENTIFIERS_ISIN_JSON = os.path.join(SCRIPTS_DIR, "identifiers_isin.json")
OUTPUT_JSON = os.path.join(SCRIPTS_DIR, "sample_identifier_links.json")

YAHOO_SUFFIX = {
    "XNAS": "",
    "XNGS": "",
    "XNYS": "",
    "XASE": "",
    "XETR": ".DE",
    "XFRA": ".F",
    "XLON": ".L",
    "XTSE": ".TO",
    "XWBO": ".VI",
    "BVMF": ".SA",
    "XHKG": ".HK",
    "XTKS": ".T",
}


def ticker_root(ticker):
    return re.split(r"[._/]", (ticker or "").strip().upper())[0]


def resolve_yahoo_ticker(ticker, mic):
    base = ticker_root(ticker)
    return f"{base}{YAHOO_SUFFIX.get((mic or '').strip().upper(), '')}"


def load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    identifiers = load_json(IDENTIFIERS_JSON, {})
    identifier_isins = {
        ticker.upper(): isin.strip().upper()
        for ticker, isin in load_json(IDENTIFIERS_ISIN_JSON, {}).items()
        if isin
    }

    exact = {ticker.upper(): ticker for ticker in identifiers}
    roots = {}
    for ticker in identifiers:
        roots.setdefault(ticker_root(ticker), []).append(ticker)

    by_isin = {}
    for ticker, isin in identifier_isins.items():
        by_isin.setdefault(isin, []).append(ticker)

    links = []
    with open(SAMPLE_CSV, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sample_ticker = row["TICKER"].strip()
            sample_mic = row["MIC"].strip()
            sample_isin = row["ISIN"].strip().upper()
            yahoo_ticker = resolve_yahoo_ticker(sample_ticker, sample_mic)
            root = ticker_root(sample_ticker)

            matches = []
            if yahoo_ticker.upper() in exact:
                matches.append((exact[yahoo_ticker.upper()], "exact_yahoo_ticker"))
            for ticker in roots.get(root, []):
                matches.append((ticker, "root_ticker"))
            for ticker in by_isin.get(sample_isin, []):
                matches.append((ticker, "isin"))

            seen = set()
            unique_matches = []
            for ticker, method in matches:
                key = ticker.upper()
                if key in seen:
                    continue
                seen.add(key)
                unique_matches.append({
                    "identifier_ticker": ticker,
                    "method": method,
                    "identifier": identifiers.get(ticker, {}),
                })

            links.append({
                "sample_name": row["NAME"].strip(),
                "sample_ticker": sample_ticker,
                "sample_mic": sample_mic,
                "sample_isin": sample_isin,
                "sample_sedol": row["SEDOL"].strip(),
                "normalized_root": root,
                "resolved_yahoo_ticker": yahoo_ticker,
                "matches": unique_matches,
            })

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(links, f, indent=2)

    for link in links:
        if not link["matches"]:
            print(f"{link['sample_ticker']:10} -> no match")
            continue
        match_text = ", ".join(
            f"{m['identifier_ticker']} ({m['method']})" for m in link["matches"]
        )
        print(f"{link['sample_ticker']:10} -> {match_text}")

    print(f"\nWrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
