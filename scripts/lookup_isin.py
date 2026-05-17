#!/usr/bin/env python3
"""
Look up ISIN for a ticker symbol from Wikipedia infobox.

Usage:
  python3 lookup_isin.py <TICKER> ...
  python3 lookup_isin.py AAPL MSFT BABA
  python3 lookup_isin.py --identifiers  # all tickers from identifiers.json

Caches results to identifiers_isin.json.
"""

import json, os, re, sys, urllib.request, urllib.error

SCRIPT_DIR = os.path.dirname(__file__)
IDENTIFIERS_PATH = os.path.join(SCRIPT_DIR, 'identifiers.json')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'identifiers_isin.json')

# Known ticker → Wikipedia title overrides for ambiguous names
WIKI_TITLES = {
    'AAPL': 'Apple_Inc.',
    'MSFT': 'Microsoft',
    'GOOG': 'Google',
    'GOOGL': 'Google',
    'AMZN': 'Amazon_(company)',
    'META': 'Meta_Platforms',
    'NVDA': 'Nvidia',
    'TSLA': 'Tesla,_Inc.',
    'INTC': 'Intel',
    'AMD': 'Advanced_Micro_Devices',
    'CSCO': 'Cisco',
    'NFLX': 'Netflix',
    'IBM': 'IBM',
    'ORCL': 'Oracle_Corporation',
    'QCOM': 'Qualcomm',
    'TXN': 'Texas_Instruments',
    'ADBE': 'Adobe_Inc.',
    'CRM': 'Salesforce',
    'PYPL': 'PayPal',
    'UBER': 'Uber',
    'NIO': 'NIO_(car_company)',
    'BABA': 'Alibaba_Group',
    'JD': 'JD.com',
    'BIDU': 'Baidu',
    'TCEHY': 'Tencent',
    'SONY': 'Sony',
    'TM': 'Toyota',
    'HMC': 'Honda',
    'VWAGY': 'Volkswagen_Group',
}

# Hardcoded ISINs for tickers where Wikipedia doesn't provide them
KNOWN_ISINS = {
    'META': 'US30303M1027',
}


def fetch_wikipedia_company(text):
    """Search Wikipedia for a company name and return the page title."""
    url = 'https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=' + urllib.parse.quote(text) + '&format=json&srlimit=5'
    req = urllib.request.Request(url, headers={'User-Agent': 'lookup_isin/1.0'})
    try:
        data = json.loads(urllib.request.urlopen(req, timeout=10).read())
        results = data.get('query', {}).get('search', [])
        for r in results:
            title = r.get('title', '')
            # Prefer company/corporation pages
            if any(x in title.lower() for x in ['inc.', 'corporation', 'company', 'group', 'ltd', 'limited']):
                return title
        if results:
            return results[0]['title']
    except Exception:
        pass
    return None


def search_isin_in_page(html):
    """Search entire HTML page for an ISIN pattern."""
    # ISIN: 2 letters + 9 alphanumeric + 1 check digit = 12 chars
    m = re.search(r'\b([A-Z]{2}[A-Z0-9]{9}[0-9])\b', html)
    if m:
        return m.group(1)
    return None


def fetch_isin_from_wikipedia(title):
    """Scrape Wikipedia infobox for ISIN, fallback to full page search."""
    url = f'https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}'
    req = urllib.request.Request(url, headers={'User-Agent': 'lookup_isin/1.0'})
    try:
        html = urllib.request.urlopen(req, timeout=10).read().decode('utf-8', errors='replace')
        # Try infobox first
        ib = re.search(r'<table class=\"infobox[^\"]*\"[^>]*>.*?</table>', html, re.DOTALL)
        if ib:
            rows = re.findall(r'<th[^>]*>(.*?)</th>\s*<td[^>]*>(.*?)</td>', ib.group(), re.DOTALL)
            for th, td in rows:
                th_clean = re.sub(r'<[^>]+>', '', th).strip()
                if 'ISIN' in th_clean:
                    td_clean = re.sub(r'<[^>]+>', '', td).strip()
                    m = re.search(r'([A-Z]{2}[A-Z0-9]{9}[0-9])', td_clean)
                    if m:
                        return m.group(1)
        # Fallback: search full page
        isin = search_isin_in_page(html)
        if isin:
            return isin
    except Exception:
        pass
    return None


def lookup_isin(ticker, name_hint=None):
    """Look up ISIN for a ticker. Returns (ISIN, source_description) or (None, reason)."""
    single_ticker = ticker.upper()
    # 0. Check hardcoded ISINs
    if single_ticker in KNOWN_ISINS:
        return KNOWN_ISINS[single_ticker], 'hardcoded'
    # 1. Check known Wikipedia overrides
    title = WIKI_TITLES.get(single_ticker)
    if title:
        isin = fetch_isin_from_wikipedia(title)
        if isin:
            return isin, f'Wikipedia: {title}'
        return None, f'No ISIN found on Wikipedia page "{title}"'
    # 2. Fetch company name from identifiers.json
    if not name_hint:
        try:
            with open(IDENTIFIERS_PATH) as f:
                ids = json.load(f)
            entry = ids.get(ticker.upper())
            if entry:
                name_hint = entry.get('longname') or entry.get('name', '')
        except FileNotFoundError:
            pass
    if not name_hint:
        return None, 'No company name available'
    # 3. Search Wikipedia for company name
    title = fetch_wikipedia_company(name_hint)
    if title:
        isin = fetch_isin_from_wikipedia(title)
        if isin:
            return isin, f'Wikipedia: {title}'
        return None, f'No ISIN on Wikipedia page "{title}"'
    return None, f'Wikipedia page not found for "{name_hint}"'


def main():
    import urllib.parse  # noqa: F811

    args = sys.argv[1:]
    if not args:
        print(__doc__.strip())
        return

    # Load existing cache
    isin_data = {}
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH) as f:
                isin_data = json.load(f)
        except Exception:
            isin_data = {}

    updated = False

    def cache_isin(ticker, isin):
        nonlocal updated
        if isin_data.get(ticker) != isin:
            isin_data[ticker] = isin
            updated = True

    if '--identifiers' in args:
        with open(IDENTIFIERS_PATH) as f:
            ids = json.load(f)
        for ticker, entry in ids.items():
            name = entry.get('longname') or entry.get('name', '')
            if ticker in isin_data:
                print(f'{ticker:8s} {name[:35]:35s} {isin_data[ticker]}  (cached)')
                continue
            print(f'{ticker:8s} {name[:35]:35s}  ', end='', flush=True)
            isin, source = lookup_isin(ticker, name)
            if isin:
                print(f'{isin}  ({source})')
                cache_isin(ticker, isin)
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
            isin, source = lookup_isin(ticker, name)
            if isin:
                print(f'{ticker} ({label}) → {isin}  ({source})')
                cache_isin(ticker, isin)
            else:
                print(f'{ticker} ({label}) → ?  ({source})')

    if updated:
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(isin_data, f, indent=2)
            f.write('\n')
        print(f'\nSaved {len(isin_data)} ISINs to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
