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

import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

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
        outputs = model(**inputs).last_hidden_state
        embeddings = outputs.mean(dim=1)
    return embeddings

def clean_html_and_extract_text(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    remove_tags = [
        "script", "style", "noscript", "iframe", "header", "footer", "nav", "aside",
        "form", "button", "input", "select", "option"
    ]
    for tag in soup.find_all(remove_tags):
        tag.decompose()
    for el in soup.find_all(attrs={"role": True}):
        role = el.get("role", "").lower()
        if "button" in role:
            el.decompose()
    for a in soup.find_all("a"):
        anchor_text = a.get_text(separator=" ", strip=True)
        if anchor_text:
            a.replace_with(anchor_text)
        else:
            a.decompose()
    for tag in soup.find_all("meta"):
        tag.decompose()
    text_parts = []
    for element in soup.descendants:
        if element.name in ("pre", "code"):
            try:
                block_text = element.get_text(separator="\n", strip=False)
            except Exception:
                block_text = str(element)
            if block_text:
                text_parts.append(block_text)
        elif getattr(element, "string", None) and element.parent.name not in ("pre", "code"):
            s = element.string.strip()
            if s:
                text_parts.append(s)
    combined = "\n\n".join(text_parts)
    lines = combined.splitlines()
    cleaned_lines = []
    prev_blank = False
    for ln in lines:
        stripped = ln.rstrip()
        if not stripped:
            if not prev_blank:
                cleaned_lines.append("")
            prev_blank = True
        else:
            cleaned_lines.append(stripped)
            prev_blank = False
    final_text = "\n\n".join([line for line in "\n".join(cleaned_lines).split("\n\n") if line.strip()])
    return final_text

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

        if html_content:
            text_content = clean_html_and_extract_text(html_content)
        else:
            text_content = markdown_content or ""

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
        try:
            embeddings = embed_text(text_content)
            torch.save(embeddings.cpu(), out_file_emb)
        except Exception:
            with open(os.path.join(data_dir, "error_log.txt"), "a", encoding="utf-8") as f:
                f.write(f"Embedding error for: {norm}\n")
                f.write(traceback.format_exc() + "\n")

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

        if html_content:
            soup_for_links = BeautifulSoup(html_content, "html.parser")
            for a in soup_for_links.find_all("a", href=True):
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

def process_text_files_in_folder(folder: str, data_dir: str):
    num_processed = 0
    for root, dirs, files in os.walk(folder):
        for fname in files:
            if fname.lower().endswith(".txt"):
                src_path = os.path.join(root, fname)
                try:
                    with open(src_path, "r", encoding="utf-8") as f:
                        text = f.read()
                except Exception:
                    try:
                        with open(src_path, "r", encoding="latin-1") as f:
                            text = f.read()
                    except Exception:
                        with open(os.path.join(data_dir, "error_log.txt"), "a", encoding="utf-8") as elog:
                            elog.write(f"Could not read file: {src_path}\n")
                            elog.write(traceback.format_exc() + "\n")
                        continue
                rel_path = os.path.relpath(src_path, folder)
                safe_base = safe_filename(rel_path).replace(os.sep, "_")
                out_file_txt = os.path.join(data_dir, safe_base)
                if not out_file_txt.lower().endswith(".txt"):
                    out_file_txt = out_file_txt + ".txt"
                i = 1
                orig_txt = out_file_txt
                while os.path.exists(out_file_txt):
                    out_file_txt = f"{os.path.splitext(orig_txt)[0]}_{i}.txt"
                    i += 1
                try:
                    with open(out_file_txt, "w", encoding="utf-8") as f:
                        f.write(text)
                except Exception:
                    with open(os.path.join(data_dir, "error_log.txt"), "a", encoding="utf-8") as elog:
                        elog.write(f"Could not write copy of file: {src_path} -> {out_file_txt}\n")
                        elog.write(traceback.format_exc() + "\n")
                    continue
                out_file_emb = out_file_txt.replace(".txt", "_embedding.pt")
                try:
                    embeddings = embed_text(text)
                    torch.save(embeddings.cpu(), out_file_emb)
                    num_processed += 1
                except Exception:
                    with open(os.path.join(data_dir, "error_log.txt"), "a", encoding="utf-8") as elog:
                        elog.write(f"Embedding error for file: {src_path}\n")
                        elog.write(traceback.format_exc() + "\n")
    return num_processed

def run_crawl_for_url(start_url: str, data_dir: str):
    browser_conf = BrowserConfig(headless=True, chrome_channel="chromium")
    run_conf = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    base_domain = urlparse(start_url).netloc
    visited = set()
    async def run():
        async with AsyncWebCrawler(config=browser_conf) as crawler:
            await crawl_and_save(start_url, crawler, run_conf, data_dir, visited, base_domain, depth=0)
    try:
        asyncio.run(run())
    except Exception:
        with open(os.path.join(data_dir, "error_log.txt"), "a", encoding="utf-8") as f:
            f.write("Fatal error in main crawl run\n")
            f.write(traceback.format_exc() + "\n")

def on_enter_url(root, data_dir):
    url = simpledialog.askstring("Enter URL", "Enter the starting URL to crawl:", parent=root)
    if not url or not url.strip():
        messagebox.showinfo("No URL", "No URL entered.", parent=root)
        return
    url = url.strip()
    try:
        run_crawl_for_url(url, data_dir)
        messagebox.showinfo("Done", f"Crawling finished. Results saved in: {data_dir}", parent=root)
    except Exception:
        with open(os.path.join(data_dir, "error_log.txt"), "a", encoding="utf-8") as f:
            f.write("Fatal error during URL processing\n")
            f.write(traceback.format_exc() + "\n")
        messagebox.showerror("Error", "An error occurred. Check error_log.txt in data folder.", parent=root)

def on_choose_folder(root, data_dir):
    folder = filedialog.askdirectory(title="Select folder containing .txt files", parent=root)
    if not folder:
        messagebox.showinfo("No folder", "No folder selected.", parent=root)
        return
    try:
        n = process_text_files_in_folder(folder, data_dir)
        messagebox.showinfo("Done", f"Processed {n} .txt files. Saved in: {data_dir}", parent=root)
    except Exception:
        with open(os.path.join(data_dir, "error_log.txt"), "a", encoding="utf-8") as f:
            f.write("Fatal error during folder processing\n")
            f.write(traceback.format_exc() + "\n")
        messagebox.showerror("Error", "An error occurred. Check error_log.txt in data folder.", parent=root)

def main():
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, "data")
    os.makedirs(data_dir, exist_ok=True)
    root = tk.Tk()
    root.title("Choose input")
    root.geometry("500x500")
    anthracite = "#2E2E2E"
    btn_bg = "#444444"
    root.configure(bg=anthracite)
    lbl = tk.Label(root, text="Select an action", bg=anthracite, fg="white", font=("Arial", 18))
    lbl.pack(pady=40)
    btn_url = tk.Button(root, text="Enter URL", width=20, height=2, bg=btn_bg, fg="white",
                        command=lambda: on_enter_url(root, data_dir))
    btn_url.pack(pady=20)
    btn_folder = tk.Button(root, text="Choose Folder", width=20, height=2, bg=btn_bg, fg="white",
                           command=lambda: on_choose_folder(root, data_dir))
    btn_folder.pack(pady=20)
    note = tk.Label(root, text="Folder processing will include .txt files in subdirectories.", bg=anthracite, fg="white")
    note.pack(side="bottom", pady=20)
    root.mainloop()

if __name__ == "__main__":
    main()
