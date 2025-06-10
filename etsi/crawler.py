import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
import time # Recommended for politeness, to add delays
import os
import sys

# --- Important Note on robots.txt ---
# The robots.txt file for www.etsi.org (https://www.etsi.org/robots.txt)
# contains "Disallow: /deliver/".
# Running this script against https://www.etsi.org/deliver/etsi_ts/
# would violate their robots.txt policy and terms of service.
# This code is provided for educational purposes to demonstrate web crawling and downloading.
# Always respect website terms of service and robots.txt files.
# Proceed with caution and at your own risk if you target etsi.org.

def compare_versions(v1_str, v2_str):
    """
    Compares two version strings.
    Version strings are expected to be like '010101'.
    They are split by '.' or '_' and compared numerically part by part.
    Returns:
        1 if v1_str > v2_str
       -1 if v1_str < v2_str
        0 if v1_str == v2_str
    """
    def normalize(v_str):
        return [int(x) for x in re.split(r'[._]', v_str)]

    n1 = normalize(v1_str)
    n2 = normalize(v2_str)

    if n1 > n2:
        return 1
    elif n1 < n2:
        return -1
    else:
        return 0

def crawl_and_find_latest_pdf(base_url, current_path=""):
    """
    Recursively crawls directories from the base_url + current_path,
    finds PDF files matching a specific pattern, and determines the latest version for each document.
    """
    full_url_to_crawl = urljoin(base_url, current_path)
    print(f"Crawling: {full_url_to_crawl}")

    latest_pdfs_found = {} # Holds {doc_num: (version_str, pdf_url)}

    try:
        # Consider adding a User-Agent header
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 ETSI_Crawler/1.0'}
        response = requests.get(full_url_to_crawl, timeout=15, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {full_url_to_crawl}: {e}")
        return latest_pdfs_found

    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a')

    for link in links:
        href = link.get('href')

        if href is None or href.startswith('?') or href == '../' or href == '/' or href.startswith("mailto:") or href.startswith("javascript:"):
            continue

        absolute_href_url = urljoin(full_url_to_crawl, href)

        if href.endswith('/'): # It's a subdirectory
            # time.sleep(0.1) # Optional delay
            new_relative_path = urljoin(current_path, href)
            sub_latest = crawl_and_find_latest_pdf(base_url, new_relative_path)
            for doc_num, (version_str, pdf_url) in sub_latest.items():
                if doc_num not in latest_pdfs_found or \
                   compare_versions(version_str, latest_pdfs_found[doc_num][0]) > 0:
                    latest_pdfs_found[doc_num] = (version_str, pdf_url)

        elif href.lower().endswith('.pdf'):
            match = re.match(r'ts_(\d+)v([\d]+)p\.pdf', href, re.IGNORECASE)
            if match:
                doc_num = match.group(1)
                version_str_from_filename = match.group(2)
                if doc_num not in latest_pdfs_found or \
                   compare_versions(version_str_from_filename, latest_pdfs_found[doc_num][0]) > 0:
                    latest_pdfs_found[doc_num] = (version_str_from_filename, absolute_href_url)
                    # print(f"  Found potential latest: Doc {doc_num}, Ver {version_str_from_filename}, URL {absolute_href_url}")
    return latest_pdfs_found

def download_pdf(pdf_url, output_directory):
    """Downloads a PDF from pdf_url into output_directory."""
    try:
        # Extract filename from URL
        filename = pdf_url.split('/')[-1]
        if not filename.lower().endswith(".pdf"): # Basic check
            print(f"  Skipping non-PDF or malformed URL for download: {pdf_url}")
            return False

        filepath = os.path.join(output_directory, filename)

        print(f"  Downloading {filename} to {filepath}...")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 ETSI_Downloader/1.0'}
        pdf_response = requests.get(pdf_url, stream=True, timeout=30, headers=headers)
        pdf_response.raise_for_status()

        with open(filepath, 'wb') as f:
            for chunk in pdf_response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  Successfully downloaded {filename}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"  Error downloading {pdf_url}: {e}")
        return False
    except IOError as e:
        print(f"  Error saving file {filepath}: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python crawler.py <output_directory>")
        sys.exit(1)

    output_dir = sys.argv[1]

    # Create the output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {os.path.abspath(output_dir)}")
    except OSError as e:
        print(f"Error creating directory {output_dir}: {e}")
        sys.exit(1)

    base_crawl_url = 'https://www.etsi.org/deliver/etsi_ts/' # As per user request

    print(f"Starting crawl from: {base_crawl_url}")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("WARNING: Crawling /deliver/ on etsi.org is DISALLOWED by their robots.txt.")
    print("This script will violate their policy and may result in your IP being blocked.")
    print("This code is for educational purposes. Use with extreme caution and awareness.")
    print("Consider testing on a permitted site or local server.")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    
    # Ask for explicit confirmation before proceeding if targeting etsi.org
    # For safety, this check is included.
    if "etsi.org/deliver/" in base_crawl_url:
        confirm = input(f"Are you sure you want to proceed with crawling {base_crawl_url}? (yes/no): ").lower()
        if confirm != 'yes':
            print("Crawling aborted by user.")
            sys.exit(0)

    latest_pdf_documents = crawl_and_find_latest_pdf(base_crawl_url)

    if latest_pdf_documents:
        print(f"\nFound {len(latest_pdf_documents)} unique documents with latest versions.")
        print("--- Starting Downloads ---")
        download_count = 0
        for doc_num, (version, pdf_url) in latest_pdf_documents.items():
            print(f"Document: {doc_num}, Latest Version: {version}, URL: {pdf_url}")
            if download_pdf(pdf_url, output_dir):
                download_count += 1
            # time.sleep(0.5) # Polite delay between downloads
        print(f"\n--- Download Summary ---")
        print(f"Successfully downloaded {download_count} of {len(latest_pdf_documents)} latest PDF documents.")
    else:
        print("No PDF documents matching the pattern were found, or an error occurred during crawling.")

