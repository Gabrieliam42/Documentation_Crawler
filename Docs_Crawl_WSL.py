# Script Developer: Gabriel Mihai Sandu
# GitHub Profile: https://github.com/Gabrieliam42

import sys
import os
import traceback
import asyncio
from urllib.parse import urlparse, urljoin, urlunparse
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig, CacheMode

import torch
from transformers import AutoTokenizer, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)

sys.setrecursionlimit(10000)

MAX_DEPTH = 124
ARUN_TIMEOUT = 30.0

def safe_filename(name: str) -> str:
    invalid = '<>:"/\\|?*'
    for c in invalid:
        name = name.replace(c, "_")
    return name

def url_to_filename(url: str) -> str:
    parsed = urlparse(url)
    netloc = parsed.netloc.replace(":", "_")
    path = parsed.path.strip("/").replace("/", "_") or "index"
    if parsed.query:
        qhash = str(abs(hash(parsed.query)))[:8]
        path = f"{path}_{qhash}"
    base = f"{netloc}_{path}"
    return safe_filename(base)

def normalize_url_no_fragment(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    cleaned = parsed._replace(fragment="")
    return urlunparse(cleaned)

def is_valid_http_link(href: str) -> bool:
    if not href:
        return False
    href = href.strip()
    if href.startswith("mailto:") or href.startswith("javascript:"):
        return False
    return href.startswith("http://") or href.startswith("https://")

def embed_text(text: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

async def crawl_and_save(url: str, crawler: AsyncWebCrawler, run_conf: CrawlerRunConfig,
                         data_dir: str, visited: set, base_domain: str, depth: int = 0):
    try:
        if depth > MAX_DEPTH:
            return
        norm = normalize_url_no_fragment(url)
        if norm in visited:
            return
        visited.add(norm)
        parsed_url = urlparse(norm)
        if parsed_url.netloc != base_domain:
            return

        try:
            result = await asyncio.wait_for(crawler.arun(url=norm, config=run_conf), timeout=ARUN_TIMEOUT)
        except asyncio.TimeoutError:
            with open(os.path.join(data_dir, "error_log.txt"), "a", encoding="utf-8") as f:
                f.write(f"Timeout fetching: {norm}\n")
            return
        except Exception:
            with open(os.path.join(data_dir, "error_log.txt"), "a", encoding="utf-8") as f:
                f.write(f"Fetch error: {norm}\n")
                f.write(traceback.format_exc() + "\n")
            return

        html_content = getattr(result, "html", "") or ""
        markdown_content = getattr(result, "markdown", "") or ""

        if markdown_content:
            text_content = markdown_content
        else:
            if html_content:
                soup = BeautifulSoup(html_content, "html.parser")
                text_content = soup.get_text(separator="\n", strip=True)
            else:
                text_content = ""

        filename_base = url_to_filename(norm)
        out_file_txt = os.path.join(data_dir, filename_base + ".txt")
        i = 1
        orig_txt = out_file_txt
        while os.path.exists(out_file_txt):
            out_file_txt = f"{os.path.splitext(orig_txt)[0]}_{i}.txt"
            i += 1
        with open(out_file_txt, "w", encoding="utf-8") as f:
            f.write(text_content)

        out_file_emb = out_file_txt.replace(".txt", "_embedding.pt")
        embeddings = embed_text(text_content)
        torch.save(embeddings.cpu(), out_file_emb)

        links = set()
        raw_links = getattr(result, "links", []) or []
        for l in raw_links:
            href = None
            if isinstance(l, str):
                href = l
            elif isinstance(l, dict):
                href = l.get("href") or l.get("url") or l.get("link")
            else:
                href = getattr(l, "href", None) or getattr(l, "url", None)
            if href:
                joined = urljoin(norm, href)
                joined = normalize_url_no_fragment(joined)
                if is_valid_http_link(joined):
                    links.add(joined)

        for a in BeautifulSoup(html_content, "html.parser").find_all("a", href=True):
            href = urljoin(norm, a["href"])
            href = normalize_url_no_fragment(href)
            if is_valid_http_link(href):
                links.add(href)

        for link in links:
            await crawl_and_save(link, crawler, run_conf, data_dir, visited, base_domain, depth + 1)

    except Exception:
        with open(os.path.join(data_dir, "error_log.txt"), "a", encoding="utf-8") as f:
            f.write("URL: " + url + "\n")
            f.write(traceback.format_exc() + "\n")

def main():
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, "data")
    os.makedirs(data_dir, exist_ok=True)

    browser_conf = BrowserConfig(headless=True, chrome_channel="chromium")
    run_conf = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    start_url = "https://docs.python.org/3/tutorial/index.html"
    base_domain = urlparse(start_url).netloc
    visited = set()

    async def run():
        async with AsyncWebCrawler(config=browser_conf) as crawler:
            await crawl_and_save(start_url, crawler, run_conf, data_dir, visited, base_domain, depth=0)

    try:
        asyncio.run(run())
    except Exception:
        with open(os.path.join(data_dir, "error_log.txt"), "a", encoding="utf-8") as f:
            f.write("Fatal error in main run\n")
            f.write(traceback.format_exc() + "\n")

if __name__ == "__main__":
    main()

# Script Developer: Gabriel Mihai Sandu
# GitHub Profile: https://github.com/Gabrieliam42