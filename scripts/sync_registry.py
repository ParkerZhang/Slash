#!/usr/bin/env python3
"""
Unified Identifier Registry Sync with Yahoo Ticker Resolution.
"""

import csv
import json
import os
import re

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_CSV = os.path.join(SCRIPTS_DIR, 'sample.csv')
IDENTIFIERS_JSON = os.path.join(SCRIPTS_DIR, 'identifiers.json')
REGISTRY_JSON = os.path.join(SCRIPTS_DIR, 'registry.json')

# MIC to Yahoo Suffix Mapping
YAHOO_SUFFIX = {
    'XNAS': '',     # Nasdaq
    'XNGS': '',     # Nasdaq GS
    'XNYS': '',     # NYSE
    'XASE': '',     # NYSE American
    'XETR': '.DE',  # Xetra
    'XFRA': '.F',   # Frankfurt
    'XLON': '.L',   # London
    'XTSE': '.TO',  # Toronto
    'XWBO': '.VI',  # Vienna
    'BVMF': '.SA',  # Brazil
    'XHKG': '.HK',  # Hong Kong
    'XTKS': '.T',   # Tokyo
}

def clean_company_name(name):
    if not name: return ""
    return re.split(r'\s+AT\s+', name, flags=re.IGNORECASE)[0].strip()

def resolve_yahoo_ticker(ticker, mic):
    """Maps SMCI_UW + XNGS -> SMCI, MS51_GY + XETR -> MS51.DE"""
    # Strip suffixes like _UW, _GY, .UN
    base = re.split(r'[._/]', ticker)[0]
    suffix = YAHOO_SUFFIX.get(mic, '')
    return f"{base}{suffix}"

def is_valid_ticker(t):
    """Filters out noise like 'INC', 'AT', 'SUPER'"""
    if t.upper() in {'AT', 'INC', 'SA', 'LTD', 'CORP', 'THE', 'AND', 'FOR', 'NEW', 'SUPER'}:
        return False
    # Valid tickers are usually 1-5 chars, or have dots/underscores
    if len(t) < 1: return False
    if len(t) > 12: return False
    return True

def main():
    registry = {'entities': {}}
    entities = registry['entities']

    # 1. Ingest sample.csv (Primary Truth)
    if os.path.exists(SAMPLE_CSV):
        print(f"Syncing from {SAMPLE_CSV}...")
        with open(SAMPLE_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                isin = row['ISIN'].strip()
                if not isin: continue
                
                if isin not in entities:
                    entities[isin] = {
                        'isin': isin,
                        'name': clean_company_name(row['NAME']),
                        'listings': []
                    }
                
                mic = row['MIC'].strip()
                ticker = row['TICKER'].strip()
                listing = {
                    'sedol': row['SEDOL'].strip(),
                    'ticker': ticker,
                    'mic': mic,
                    'yahoo_ticker': resolve_yahoo_ticker(ticker, mic),
                    'display_name': row['NAME'].strip()
                }
                
                # Update or Append listing
                found = False
                for i, existing in enumerate(entities[isin]['listings']):
                    if existing['mic'] == mic:
                        entities[isin]['listings'][i] = listing
                        found = True
                        break
                if not found:
                    entities[isin]['listings'].append(listing)

    # 2. Ingest identifiers.json (Discovery Buffer - Filtered)
    if os.path.exists(IDENTIFIERS_JSON):
        print(f"Checking {IDENTIFIERS_JSON} for new discoveries...")
        with open(IDENTIFIERS_JSON) as f:
            ids_data = json.load(f)
        
        known_tickers = {li['ticker'].upper() for ent in entities.values() for li in ent['listings']}
        
        for ticker, info in ids_data.items():
            if not is_valid_ticker(ticker): continue
            if ticker.upper() not in known_tickers:
                # Add as a discovery
                placeholder_id = f"YAHOO_{ticker.upper()}"
                if placeholder_id not in entities:
                    entities[placeholder_id] = {
                        'name': info.get('name', ticker),
                        'status': 'discovered',
                        'listings': [{
                            'ticker': ticker,
                            'mic': info.get('exchange', 'UNKNOWN'),
                            'yahoo_ticker': ticker, # Keys in identifiers.json are already Yahoo tickers
                            'sedol': ''
                        }]
                    }

    # 3. Save Registry
    registry['entities'] = entities
    with open(REGISTRY_JSON, 'w') as f:
        json.dump(registry, f, indent=2)
    
    print(f"Sync Complete. Total Entities: {len(entities)}")

if __name__ == '__main__':
    main()
