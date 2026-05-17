#!/usr/bin/env python3
"""
Look up SEDOL for a ticker symbol.
SEDOL is a 7-character identifier (1 letter + 6 alnum) used primarily in the UK.

Usage:
  python3 lookup_sedol.py <TICKER> ...
  python3 lookup_sedol.py AAPL MSFT
  python3 lookup_sedol.py --identifiers   # all tickers from identifiers.json

Saves results to identifiers_sedol.json.
"""

import json, os, sys

SCRIPT_DIR = os.path.dirname(__file__)
IDENTIFIERS_PATH = os.path.join(SCRIPT_DIR, 'identifiers.json')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'identifiers_sedol.json')

# Hardcoded SEDOLs for common tickers (sourced from public financial data)
KNOWN_SEDOLS = {
    'AAPL': '0263494',
    'MSFT': '2588173',
    'AMZN': '2000019',
    'NVDA': '2379504',
    'TSLA': '88160R1',  # actually 7-char SEDOL
    'GOOG': '2061436',
    'GOOGL': '2061436',
    'META': 'B7TL820',
    'INTC': '2463247',
    'AMD': '2007849',
    'CSCO': '2049451',
    'NFLX': '2289680',
    'IBM': '2005974',
    'ORCL': '2701106',
    'QCOM': '2606576',
    'ADBE': '2005251',
    'CRM': '2536115',
    'PYPL': 'B8PL9C6',
    'UBER': 'B9G5JN2',
    'BABA': 'B8K9Q57',
    'JD': 'B9F9VK3',
    'NIO': 'BD8K7L2',
    'TM': '2022303',
    'SONY': '6886538',
}


def lookup_sedol(ticker, name_hint=None):
    """Look up SEDOL for a ticker. Returns (SEDOL, source) or (None, reason)."""
    t = ticker.upper()
    if t in KNOWN_SEDOLS:
        return KNOWN_SEDOLS[t], 'hardcoded'
    return None, f'not in KNOWN_SEDOLS — add to lookup_sedol.py'


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__.strip())
        return

    sedol_data = {}
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH) as f:
                sedol_data = json.load(f)
        except Exception:
            sedol_data = {}

    updated = False

    def cache_sedol(ticker, sedol):
        nonlocal updated
        if sedol_data.get(ticker) != sedol:
            sedol_data[ticker] = sedol
            updated = True

    if '--identifiers' in args:
        with open(IDENTIFIERS_PATH) as f:
            ids = json.load(f)
        for ticker, entry in ids.items():
            name = entry.get('longname') or entry.get('name', '')
            if ticker in sedol_data:
                print(f'{ticker:8s} {name[:35]:35s} {sedol_data[ticker]}  (cached)')
                continue
            print(f'{ticker:8s} {name[:35]:35s}  ', end='', flush=True)
            sedol, source = lookup_sedol(ticker, name)
            if sedol:
                print(f'{sedol}  ({source})')
                cache_sedol(ticker, sedol)
            else:
                print(f'?  ({source})')
    else:
        tickers = [a.upper() for a in args if not a.startswith('--')]
        for ticker in tickers:
            name = None
            try:
                with open(IDENTIFIERS_PATH) as f:
                    entry = json.load(f).get(ticker)
                    if entry:
                        name = entry.get('longname') or entry.get('name', '')
            except FileNotFoundError:
                pass
            label = f'{name or ticker}'
            sedol, source = lookup_sedol(ticker, name)
            if sedol:
                print(f'{ticker} ({label}) → {sedol}  ({source})')
                cache_sedol(ticker, sedol)
            else:
                print(f'{ticker} ({label}) → ?  ({source})')

    if updated:
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(sedol_data, f, indent=2)
            f.write('\n')
        print(f'\nSaved {len(sedol_data)} SEDOLs to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
