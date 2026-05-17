#!/usr/bin/env python3
"""
Merge MIC, ISIN, and SEDOL into a single mapping file.

Combines:
  identifiers_mic.json    → exchange MICs per ticker
  identifiers_isin.json   → ISIN per ticker
  identifiers_sedol.json  → SEDOL per ticker

Output: identifiers_mapping.json
  {"AAPL": {"mics": ["XNAS", ...], "isin": "US0378331005", "sedol": "0263494"}, ...}

Usage:
  python3 build_mapping.py
"""

import json, os

SCRIPT_DIR = os.path.dirname(__file__)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'identifiers_mapping.json')

SOURCES = {
    'mics': 'identifiers_mic.json',
    'isin': 'identifiers_isin.json',
    'sedol': 'identifiers_sedol.json',
}


def main():
    mapping = {}

    for key, fname in SOURCES.items():
        path = os.path.join(SCRIPT_DIR, fname)
        if not os.path.exists(path):
            print(f'SKIP {fname} — not found')
            continue
        with open(path) as f:
            data = json.load(f)
        print(f'{fname}: {len(data)} entries')
        for ticker, value in data.items():
            if ticker not in mapping:
                mapping[ticker] = {}
            mapping[ticker][key] = value

    # Clean up: remove tickers with no data
    mapping = {k: v for k, v in mapping.items() if v}

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(mapping, f, indent=2)
        f.write('\n')

    total = len(mapping)
    with_data = sum(1 for v in mapping.values() if len(v) == 3)
    print(f'\nWritten to {OUTPUT_PATH}')
    print(f'  Total tickers: {total}')
    print(f'  With all 3 identifiers: {with_data}')
    print(f'  With mics: {sum(1 for v in mapping.values() if "mics" in v)}')
    print(f'  With isin: {sum(1 for v in mapping.values() if "isin" in v)}')
    print(f'  With sedol: {sum(1 for v in mapping.values() if "sedol" in v)}')


if __name__ == '__main__':
    main()
