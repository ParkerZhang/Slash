#!/usr/bin/env python3
"""
Discover MIC codes from Yahoo Finance search and enrich known_mics.json.

Usage:
  python3 enrich_listings.py --all          # search by company name from identifiers.json
  python3 enrich_listings.py <name> ...     # search by given company name
  python3 enrich_listings.py --identifiers  # extract exchange codes from identifiers.json directly

Queries Yahoo's search API for each company name, collects all exchange codes
returned, maps them to MIC codes, and adds any new ones to known_mics.json.
Does NOT modify identifiers.json.
"""

import json
import os
import sys
import urllib.request
import urllib.error
import urllib.parse

SCRIPT_DIR = os.path.dirname(__file__)
IDENTIFIERS_PATH = os.path.join(SCRIPT_DIR, 'identifiers.json')
KNOWN_MICS_PATH = os.path.join(SCRIPT_DIR, 'known_mics.json')

sys.path.insert(0, SCRIPT_DIR)
from lookup_mic import EXCHANGE_MIC, NAME_MIC


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
    mic = EXCHANGE_MIC.get(exch)
    if mic is None:
        print(f'  INFO: no MIC mapping for exchange "{exch}"', file=sys.stderr)
    return mic or ''


def discover_from_names(names):
    """Query Yahoo for each name and collect exchange codes."""
    discovered = {}  # mic -> {name, yahoo_code, exchDisp}
    for name in names:
        print(f'\n"{name}"...')
        result = yahoo_search(name)
        quotes = result.get('quotes', [])
        seen = set()
        for q in quotes:
            sym = q.get('symbol')
            if not sym or sym in seen:
                continue
            seen.add(sym)
            exch = q.get('exchange', '?')
            mic = mic_for_exchange(exch)
            if mic and mic not in discovered:
                disp = q.get('exchDisp', '')
                discovered[mic] = {
                    'name': disp,
                    'yahoo_code': exch,
                    'exchDisp': disp,
                }
                print(f'  {sym:20s} exch={exch:6s} mic={mic:6s} disp="{disp}"')
    return discovered


def discover_from_identifiers():
    """Extract exchange codes from identifiers.json listings if any remain."""
    try:
        with open(IDENTIFIERS_PATH) as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    discovered = {}
    for ticker, entry in data.items():
        for li in entry.get('listings', []):
            mic = li.get('mic', '')
            exch = li.get('exchange', '')
            if mic and mic not in discovered:
                discovered[mic] = {
                    'name': li.get('exchDisp', ''),
                    'yahoo_code': exch,
                    'exchDisp': li.get('exchDisp', ''),
                }
    return discovered


def enrich_known_mics(discovered):
    if not discovered:
        return
    try:
        with open(KNOWN_MICS_PATH) as f:
            known = json.load(f)
    except FileNotFoundError:
        known = []
    existing = {entry['mic'] for entry in known}
    added = 0
    for mic in sorted(discovered):
        if mic in existing:
            continue
        info = discovered[mic]
        name = ''
        for dk, mv in NAME_MIC.items():
            if mv == mic:
                name = dk
                break
        if not name:
            name = info.get('exchDisp', '')
        if not name:
            name = info.get('yahoo_code', '')
        known.append({'mic': mic, 'name': name, 'yahoo_code': info.get('yahoo_code', '')})
        added += 1
    if added:
        with open(KNOWN_MICS_PATH, 'w') as f:
            json.dump(known, f, indent=2)
            f.write('\n')
        print(f'\nAdded {added} new MICs to known_mics.json:')
        for entry in known[-added:]:
            print(f'  {entry["mic"]:6s}  {entry["name"]}')
    else:
        print('\nNo new MICs to add.')


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__.strip())
        return

    discovered = {}

    if '--identifiers' in args:
        print('Extracting exchange codes from identifiers.json...')
        discovered = discover_from_identifiers()

    # Company names from identifiers
    if '--all' in args:
        try:
            with open(IDENTIFIERS_PATH) as f:
                data = json.load(f)
        except FileNotFoundError:
            print('identifiers.json not found')
            return
        names = []
        for entry in data.values():
            n = entry.get('longname') or entry.get('name', '')
            if n:
                names.append(n)
        discovered = discover_from_names(names)

    # Direct company name arguments
    name_args = [a for a in args if not a.startswith('--')]
    if name_args:
        discovered = discover_from_names(name_args)

    if discovered:
        enrich_known_mics(discovered)


if __name__ == '__main__':
    main()
