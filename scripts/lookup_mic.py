#!/usr/bin/env python3
"""Discover MIC codes for Yahoo exchange codes.

Usage:
  python3 lookup_mic.py <YAHOO_EXCHANGE_CODE> ...
  python3 lookup_mic.py --unknowns     # scan identifiers.json for unmapped codes

Strategy:
  1. Look up exchange_code -> MIC in the built-in dictionary.
  2. If unknown, search Yahoo for a company from identifiers.json
     that trades on that exchange, read exchDisp (display name).
  3. Look up exchDisp name -> MIC in name dictionary.
  4. If still unknown, use the MIC model's ONNX embedding to find
     the closest known MIC by exchange-name text similarity.
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

# ── Yahoo exchange code -> MIC ──────────────────────────────────────────────
# Primary mapping: Yahoo's exchange field codes to ISO MIC
EXCHANGE_MIC = {
    # US
    'NYQ': 'XNYS', 'NYS': 'XNYS', 'NYE': 'XNYS',
    'NMS': 'XNAS', 'NCM': 'XNAS', 'NGM': 'XNAS', 'NAS': 'XNAS',
    'PCX': 'XASE', 'ASE': 'XASE', 'BTS': 'XBOS',
    'PNK': 'XOTC',
    # Americas
    'TOR': 'XTSE', 'TSE': 'XTSE', 'VAN': 'XVAN',
    'MEX': 'XMEX',
    'BUE': 'XBUE',
    'SGO': 'XSGO',
    'BSP': 'XBSP',
    # Europe
    'LSE': 'XLON', 'IOB': 'XLON',
    'FRA': 'XFRA', 'GER': 'XFRA', 'DUS': 'XFRA',
    'HAM': 'XHAM',  # Hamburg
    'STU': 'XSTU',  # Stuttgart
    'MUN': 'XMUN',  # Munich
    'BER': 'XBER',  # Berlin
    'PAR': 'XPAR',
    'SWX': 'XSWX',
    'WBO': 'XWBO', 'VIE': 'XWBO',
    'MIL': 'XMIL',
    'AMS': 'XAMS',
    'BRU': 'XBRU',
    'LIS': 'XLIS',
    'OSL': 'XOSL',
    'STO': 'XSTO',
    'CPH': 'XCSE',
    'HEL': 'XHEL',
    'ICE': 'XICE',
    'MAD': 'XMAD',
    'DUB': 'XDUB',
    # Asia
    'HKG': 'XHKG',
    'JPX': 'XTKS',  # Tokyo
    'TKS': 'XTKS',
    'SHG': 'XSHG', 'SZN': 'XSHE',
    'BOM': 'XBOM',
    'NSI': 'XNSE',
    'KOS': 'XKOS',
    'SES': 'XSES',
    'TAI': 'XTAI',
    'KLS': 'XKLS',
    'IDX': 'XIDX',
    'SET': 'XBKK',
    'PSE': 'XPHS',
    'HST': 'XSTC',
    # Oceania
    'ASX': 'XASX',
    'NZE': 'XNZE',
    # Africa
    'JNB': 'XJNB',
    # Middle East
    'SAU': 'XSAU',
    'DXB': 'XDFM',
    'ABU': 'XADS',
    'QAT': 'XDSM',
    'KUW': 'XKUW',
    # Other
    'DXE': 'XETR',
    'FGI': 'XSTM',
    'OQX': 'XOTC',
    'BUD': 'XBUD',
    # Indices / Futures
    'CME': 'XCME',
    'CGI': 'XCBO',   # CBOE Global Indices
    'DJI': 'XNYS',   # Dow Jones indices listed via NYSE
    'NEO': 'XNEO',   # NEO Exchange (Canada)
    'SAO': 'XBSP',   # São Paulo / B3 (Brazil)
    'HAN': 'XHAM',   # Hannover / Hamburg
}

# ── exchDisp (human-readable exchange display name) -> MIC ──────────────────
NAME_MIC = {
    'NYSE': 'XNYS',
    'New York Stock Exchange': 'XNYS',
    'NASDAQ': 'XNAS',
    'NASDAQ Stock Exchange': 'XNAS',
    'NYSE American': 'XASE',
    'NYSE American Stock Exchange': 'XASE',
    'NYSE Arca': 'XASE',
    'BATS': 'XBOS',
    'CBOE BZX': 'XBOS',
    'OTC': 'XOTC',
    'Toronto': 'XTSE',
    'Toronto Stock Exchange': 'XTSE',
    'TSX': 'XTSE',
    'TSX Venture': 'XVAN',
    'TSX Venture Exchange': 'XVAN',
    'Mexico': 'XMEX',
    'Mexican Stock Exchange': 'XMEX',
    'Buenos Aires': 'XBUE',
    'Buenos Aires Stock Exchange': 'XBUE',
    'Santiago': 'XSGO',
    'Santiago Stock Exchange': 'XSGO',
    'Brazil': 'XBSP',
    'B3 Brasil Bolsa Balcao': 'XBSP',
    'London': 'XLON',
    'London Stock Exchange': 'XLON',
    'Frankfurt': 'XFRA',
    'Frankfurt Stock Exchange': 'XFRA',
    'XETRA': 'XFRA',
    'Frankfurt Stock Exchange Xetra': 'XFRA',
    'Dusseldorf Stock Exchange': 'XFRA',
    'Hamburg': 'XHAM',
    'Hamburg Stock Exchange': 'XHAM',
    'Stuttgart': 'XSTU',
    'Stuttgart Stock Exchange': 'XSTU',
    'Munich': 'XMUN',
    'Munich Stock Exchange': 'XMUN',
    'Berlin': 'XBER',
    'Paris': 'XPAR',
    'Euronext Paris': 'XPAR',
    'Swiss': 'XSWX',
    'Swiss Exchange': 'XSWX',
    'Vienna': 'XWBO',
    'Vienna Stock Exchange': 'XWBO',
    'Milan': 'XMIL',
    'Borsa Italiana Milan': 'XMIL',
    'Euronext Amsterdam': 'XAMS',
    'Amsterdam': 'XAMS',
    'Brussels': 'XBRU',
    'Euronext Brussels': 'XBRU',
    'Lisbon': 'XLIS',
    'Euronext Lisbon': 'XLIS',
    'Oslo': 'XOSL',
    'Oslo Stock Exchange': 'XOSL',
    'Stockholm': 'XSTO',
    'Nasdaq Stockholm': 'XSTO',
    'Copenhagen': 'XCSE',
    'Nasdaq Copenhagen': 'XCSE',
    'Helsinki': 'XHEL',
    'Nasdaq Helsinki': 'XHEL',
    'Iceland': 'XICE',
    'Madrid': 'XMAD',
    'Dublin': 'XDUB',
    'Hong Kong': 'XHKG',
    'Hong Kong Stock Exchange': 'XHKG',
    'HKSE': 'XHKG',
    'Tokyo': 'XTKS',
    'Tokyo Stock Exchange': 'XTKS',
    'Shanghai': 'XSHG',
    'Shanghai Stock Exchange': 'XSHG',
    'Shenzhen': 'XSHE',
    'BSE': 'XBOM',
    'BSE India': 'XBOM',
    'NSE': 'XNSE',
    'National Stock Exchange of India': 'XNSE',
    'Korea': 'XKOS',
    'Korea Exchange': 'XKOS',
    'Singapore': 'XSES',
    'Singapore Stock Exchange': 'XSES',
    'SGX': 'XSES',
    'Taiwan': 'XTAI',
    'Taiwan Stock Exchange': 'XTAI',
    'KLSE': 'XKLS',
    'IDX': 'XIDX',
    'Stock Exchange of Thailand': 'XBKK',
    'SET': 'XBKK',
    'PSE': 'XPHS',
    'ASX': 'XASX',
    'Australian Securities Exchange': 'XASX',
    'Australia': 'XASX',
    'NZX': 'XNZE',
    'NZX New Zealand': 'XNZE',
    'New Zealand': 'XNZE',
    'New Zealand Exchange': 'XNZE',
    'Johannesburg': 'XJNB',
    'Saudi': 'XSAU',
    'Dubai': 'XDFM',
    'Abu Dhabi': 'XADS',
    'Qatar': 'XDSM',
    'Kuwait': 'XKUW',
    'NEO': 'XNEO',
    'NEO Exchange': 'XNEO',
    'Budapest': 'XBUD',
}

VAC_DIMS = [18, 62, 28, 245]


def load_embed_weight():
    """Load raw ONNX embedding weight for MIC-model similarity fallback."""
    import numpy as np
    base = os.path.join(SCRIPT_DIR, 'modelFiles/Xenova/all-MiniLM-L6-v2-mic')
    weight_path = os.path.join(base, 'embed_weight.bin')
    tokens_path = os.path.join(base, 'embed_weight_tokens.json')
    if not os.path.exists(weight_path):
        return None, None
    with open(tokens_path) as f:
        tokens = json.load(f)
    buf = np.fromfile(weight_path, dtype=np.float32)
    n = len(tokens)
    W = buf.reshape(n, 384)
    return W, tokens


def vac_vector(W, tokens, word):
    """Get 4-d vacuum vector for a token from the embedding weight."""
    idx = tokens.index(word) if word in tokens else -1
    if idx < 0:
        word_lower = word.lower()
        idx = tokens.index(word_lower) if word_lower in tokens else -1
    if idx < 0:
        return None
    return W[idx, VAC_DIMS]


def vac_cos(a, b):
    import numpy as np
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(dot / (na * nb))


def yahoo_search(query):
    url = f'https://query1.finance.yahoo.com/v1/finance/search?q={urllib.parse.quote(query)}'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        return None


def discover_exch_disp(exchange_code):
    """Search identifiers companies to find exchDisp for an exchange code."""
    with open(IDENTIFIERS_PATH) as f:
        data = json.load(f)
    # Try each company's longname
    for ticker, entry in list(data.items())[:20]:
        name = entry.get('longname') or entry.get('name', '')
        if not name:
            continue
        result = yahoo_search(name)
        if not result:
            continue
        for q in result.get('quotes', []):
            if q.get('exchange') == exchange_code:
                disp = q.get('exchDisp', '')
                if disp:
                    return disp, q.get('symbol', '')
    return None, None


def lookup_mic(exchange_code):
    """Return MIC for a Yahoo exchange code, with discovery fallback."""
    # Step 1: direct dictionary lookup
    mic = EXCHANGE_MIC.get(exchange_code)
    if mic:
        return mic, 'dictionary'

    # Step 2: discover exchDisp from Yahoo
    print(f'  Searching Yahoo for exchange code "{exchange_code}"...', file=sys.stderr)
    disp, example = discover_exch_disp(exchange_code)
    if disp:
        print(f'    exchDisp = "{disp}"  (example: {example})', file=sys.stderr)
        mic = NAME_MIC.get(disp)
        if mic:
            return mic, f'exchDisp match: "{disp}"'
        # Step 3: check normalized display name
        disp_upper = disp.upper()
        for name_key in sorted(NAME_MIC, key=len, reverse=True):
            if name_key.upper() in disp_upper or disp_upper in name_key.upper():
                mic = NAME_MIC[name_key]
                return mic, f'exchDisp fuzzy: "{disp}" ≈ "{name_key}"'
        # Step 4: try MIC-model embedding similarity
        mic = mic_model_fallback(disp, exchange_code)
        if mic:
            return mic, f'embedding match: "{disp}" → {mic}'
        return None, f'unknown (exchDisp="{disp}")'

    return None, f'no data found for exchange code "{exchange_code}"'


def mic_model_fallback(name, exchange_code):
    """Use MIC model ONNX embedding to find closest known MIC by name similarity."""
    W, tokens = load_embed_weight()
    if W is None or tokens is None:
        return None
    # Build list of known MIC tokens from known_mics.json + family_vac.json
    with open(KNOWN_MICS_PATH) as f:
        known_mics = json.load(f)
    mic_tokens = [m['mic'] for m in known_mics]
    # Get vac vector for the exchange name (fallback: use each word)
    best_mic = None
    best_sim = -1
    query_vac = vac_vector(W, tokens, name)
    if query_vac is None:
        # Try first word of name
        first_word = name.split()[0]
        query_vac = vac_vector(W, tokens, first_word)
    if query_vac is None:
        return None
    for mic in mic_tokens:
        mic_vac = vac_vector(W, tokens, mic)
        if mic_vac is None:
            mic_vac = vac_vector(W, tokens, mic.lower())
        if mic_vac is None:
            continue
        sim = vac_cos(query_vac, mic_vac)
        if sim > best_sim:
            best_sim = sim
            best_mic = mic
    if best_mic and best_sim > 0.3:
        print(f'    (embedding sim={best_sim:.3f} to {best_mic})', file=sys.stderr)
        return best_mic
    return None


def scan_unknowns():
    """Scan identifiers.json listings for unmapped exchange codes."""
    with open(IDENTIFIERS_PATH) as f:
        data = json.load(f)
    unknown = set()
    for ticker, entry in data.items():
        for li in entry.get('listings', []):
            exch = li.get('exchange', '')
            mic = li.get('mic', '')
            if exch and not mic:
                unknown.add(exch)
    return sorted(unknown)


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__.strip())
        return

    if '--unknowns' in args:
        codes = scan_unknowns()
        if not codes:
            print('No unmapped exchange codes found in identifiers.json.')
            return
        print(f'Found {len(codes)} unmapped exchange codes:')
    else:
        codes = [a for a in args if not a.startswith('--')]

    for code in codes:
        mic, source = lookup_mic(code)
        if mic:
            print(f'  {code:6s} → {mic:6s}  ({source})')
        else:
            print(f'  {code:6s} → ?  ({source})')

    # Also print the EXCHANGE_MIC mapping as JSON for easy copy-paste
    if '--json' in args:
        print()
        print('EXCHANGE_MIC = {')
        for k, v in sorted(EXCHANGE_MIC.items()):
            print(f'    {k!r}: {v!r},')
        print('}')


if __name__ == '__main__':
    main()
