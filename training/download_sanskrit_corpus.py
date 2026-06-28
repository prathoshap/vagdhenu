#!/usr/bin/env python3
"""Download Sanskrit text corpora for PL-BERT MLM pretraining.

Sources:
  - GRETIL — clean human-input Sanskrit (~250 MB). Scrape directly from gretil.sub.uni-goettingen.de.
  - Sangraha (ai4bharat/sangraha) Sanskrit split — large OCR'd corpus from HuggingFace.
  - DSBC — Digital Sanskrit Buddhist Canon. Smaller, clean.

Output: one directory per source with raw text files (UTF-8).
"""

import argparse
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser


GRETIL_INDEX = "https://gretil.sub.uni-goettingen.de/gretil.html"
GRETIL_BASE = "https://gretil.sub.uni-goettingen.de/"


class _LinkHarvester(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for k, v in attrs:
                if k == 'href' and v:
                    self.links.append(v)


def fetch_url(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={'User-Agent': 'sanskrit-tts-research/1.0'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def harvest_gretil_links(verbose: bool = True):
    """GRETIL is a multi-page hub. Crawl the index + per-section pages to find all .txt-like leaves."""
    seen = set()
    to_visit = [GRETIL_INDEX]
    txt_links = []
    while to_visit:
        url = to_visit.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            html = fetch_url(url).decode('utf-8', errors='replace')
        except Exception as e:
            if verbose: print(f'  [skip] {url}: {e}')
            continue
        h = _LinkHarvester()
        h.feed(html)
        for link in h.links:
            if link.startswith('#') or link.startswith('mailto:'):
                continue
            # Resolve relative URLs
            if link.startswith('http'):
                abs_url = link
            else:
                abs_url = urllib.request.urljoin(url, link)
            if 'gretil.sub.uni-goettingen.de' not in abs_url:
                continue
            # Plain-text Sanskrit files
            if re.search(r'\.(txt|htm|html)$', abs_url, re.I):
                if abs_url.endswith('.htm') or abs_url.endswith('.html'):
                    # Sub-page, recurse if it looks like a corpus page
                    if any(t in abs_url for t in ('gretil/', '/1_sanskr/', '/sa_')):
                        to_visit.append(abs_url)
                elif abs_url.endswith('.txt'):
                    if abs_url not in seen:
                        txt_links.append(abs_url)
                        seen.add(abs_url)
        if verbose and len(seen) % 50 == 0:
            print(f'  [crawl] visited {len(seen)} pages, found {len(txt_links)} txt files')
    return txt_links


def download_one(url: str, out_dir: str) -> tuple[str, int]:
    fname = url.rsplit('/', 1)[-1]
    safe = re.sub(r'[^a-zA-Z0-9._-]', '_', fname)
    out_path = os.path.join(out_dir, safe)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return out_path, os.path.getsize(out_path)
    try:
        data = fetch_url(url, timeout=60)
    except Exception as e:
        return f'FAIL: {url}: {e}', 0
    with open(out_path, 'wb') as f:
        f.write(data)
    return out_path, len(data)


def download_gretil(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    print('Harvesting GRETIL .txt links (may take a few minutes)...')
    links = harvest_gretil_links(verbose=True)
    print(f'Found {len(links)} .txt files. Downloading...')
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(download_one, url, out_dir): url for url in links}
        for i, fut in enumerate(as_completed(futures), 1):
            path, size = fut.result()
            total_bytes += size
            if i % 25 == 0 or i == len(links):
                print(f'  {i}/{len(links)} done, {total_bytes/1024/1024:.1f} MB so far')
    print(f'GRETIL done: {total_bytes/1024/1024:.1f} MB in {out_dir}')


def download_sangraha(out_dir: str, max_examples: int | None = None):
    """Sangraha is on HuggingFace: ai4bharat/sangraha. Sanskrit split: language 'san' or 'sa'."""
    os.makedirs(out_dir, exist_ok=True)
    print('Downloading Sangraha Sanskrit split via HF datasets (streaming)...')
    from datasets import load_dataset
    # Sangraha has multiple configs. Sanskrit data is under 'verified' config, language code 'san'
    try:
        ds = load_dataset('ai4bharat/sangraha', 'verified', split='san', streaming=True)
    except Exception as e1:
        print(f'  [info] verified/san failed ({e1}); trying alternate config...')
        try:
            ds = load_dataset('ai4bharat/sangraha', name='san', split='train', streaming=True)
        except Exception as e2:
            print(f'  [info] alternate failed ({e2}); trying default...')
            ds = load_dataset('ai4bharat/sangraha', streaming=True, split='train')

    n = 0
    chunk_bytes = 0
    chunk_idx = 0
    chunk_file = open(os.path.join(out_dir, f'sangraha_san_{chunk_idx:04d}.txt'), 'w', encoding='utf-8')
    for row in ds:
        text = row.get('text') or row.get('content') or ''
        if not text:
            continue
        chunk_file.write(text + '\n\n')
        chunk_bytes += len(text)
        n += 1
        if chunk_bytes > 256 * 1024 * 1024:  # 256 MB per file
            chunk_file.close()
            chunk_idx += 1
            print(f'  rotated to chunk {chunk_idx}, {n} examples so far')
            chunk_file = open(os.path.join(out_dir, f'sangraha_san_{chunk_idx:04d}.txt'), 'w', encoding='utf-8')
            chunk_bytes = 0
        if max_examples and n >= max_examples:
            break
        if n % 5000 == 0:
            print(f'  fetched {n} examples')
    chunk_file.close()
    print(f'Sangraha done: {n} examples across {chunk_idx + 1} chunk files')


def download_dsbc(out_dir: str):
    """DSBC files are small + dispersed; minimal grab.
    The main collection is at dsbcproject.org but files are individually linked.
    For now, optional — GRETIL+Sangraha is plenty.
    """
    print('DSBC: skipping for now (small corpus, optional). Manual download recommended.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', default='<PROD>/sanskrit-tts/corpora')
    ap.add_argument('--sangraha_max', type=int, default=None, help='cap on Sangraha rows for testing')
    ap.add_argument('--sources', default='gretil,sangraha', help='comma-separated: gretil,sangraha,dsbc')
    args = ap.parse_args()

    sources = [s.strip() for s in args.sources.split(',')]
    if 'gretil' in sources:
        download_gretil(os.path.join(args.out_root, 'gretil'))
    if 'sangraha' in sources:
        download_sangraha(os.path.join(args.out_root, 'sangraha'), max_examples=args.sangraha_max)
    if 'dsbc' in sources:
        download_dsbc(os.path.join(args.out_root, 'dsbc'))


if __name__ == '__main__':
    main()
