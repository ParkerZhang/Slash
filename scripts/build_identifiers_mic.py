#!/usr/bin/env python3
"""
Build ticker→MIC matrix from Yahoo Finance search.

Queries Yahoo for each company in identifiers.json, collects all exchange
MIC codes, and saves the mapping to identifiers_mic.json.

Output: {"AAPL": ["XNAS", "XFRA", ...], "BABA": ["XNYS", "XHKG", ...]}

Usage:
  python3 build_identifiers_mic.py
"""

import json
import os
import sys
import urllib.request
import urllib.error
import urllib.parse

SCRIPT_DIR = os.path.dirname(__file__)
IDENTIFIERS_PATH = os.path.join(SCRIPT_DIR, 'identifiers.json')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'identifiers_mic.json')

sys.path.insert(0, SCRIPT_DIR)
from lookup_mic import EXCHANGE_MIC


def yahoo_search(query):
    url = f'https://query1.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(query)}'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        print(f'  WARN: {e}', file=sys.stderr)
        return {}


def mic_for_exchange(exch):
    return EXCHANGE_MIC.get(exch, '')


def main():
    with open(IDENTIFIERS_PATH) as f:
        data = json.load(f)

    matrix = {}
    for ticker, entry in data.items():
        name = entry.get('longname') or entry.get('name', '')
        if not name:
            continue
        print(f'{ticker:8s} {name[:40]}')
        result = yahoo_search(name)
        quotes = result.get('quotes', [])
        mics = []
        seen = set()
        for q in quotes:
            sym = q.get('symbol')
            if not sym or sym in seen:
                continue
            seen.add(sym)
            exch = q.get('exchange', '?')
            mic = mic_for_exchange(exch)
            if mic and mic not in mics:
                mics.append(mic)
        if mics:
            matrix[ticker] = mics
            print(f'  -> {", ".join(mics)}')

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(matrix, f, indent=2)
        f.write('\n')
    print(f'\nSaved {len(matrix)} tickers to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
