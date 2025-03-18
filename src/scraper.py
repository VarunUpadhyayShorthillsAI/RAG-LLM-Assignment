import os
import requests
from bs4 import BeautifulSoup

# Base URL for the Medline Plus medical encyclopedia
BASE_URL = "https://medlineplus.gov/ency/"

def fetch_page(url):
    """Fetch HTML content from a URL"""
    response = requests.get(url)
    return response.text if response.status_code == 200 else None

def extract_text(html):
    """Extract relevant text from HTML content"""
    soup = BeautifulSoup(html, "html.parser")
    
    # Extract article title
    title_tag = soup.find("h1", class_="with-also", itemprop="name")
    article_title = title_tag.get_text(strip=True) if title_tag else "Untitled"
    # Create a safe filename from the title
    safe_title = "".join(c for c in article_title if c.isalnum() or c in " _-").strip()

    extracted_text = [f"Title: {safe_title}"]

    # Process each section of the article
    for section in soup.find_all("div", class_="section"):
        title_div = section.find("div", class_="section-title")
        body_div = section.find("div", class_="section-body")

        if title_div and body_div:
            section_title = title_div.get_text(strip=True)
            section_content = body_div.get_text(" ", strip=True)
            
            # Skip unwanted sections
            if any(exclude in section_title.lower() for exclude in ["images", "references", "review date"]):
                continue

            extracted_text.append(f"\n{section_title}\n{section_content}")

    return safe_title, "\n".join(extracted_text)

def save_to_file(alphabet, title, content):
    """Save extracted content to a file organized by alphabet"""
    folder_path = os.path.join("articles", alphabet)
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f"{title}.txt")

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)

    print(f"Saved: {file_path}")

def get_article_links(alphabet):
    """Get all article links for a specific alphabet page"""
    url = f"{BASE_URL}encyclopedia_{alphabet}.htm"
    html = fetch_page(url)
    
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    article_links = []

    # Find all list items containing article links
    for li in soup.select("#mplus-content li"):
        if not li.get("class"):  # Skip non-article list items
            a_tag = li.find("a", href=True)
            if a_tag and a_tag["href"].startswith("article/"):
                article_links.append(BASE_URL + a_tag["href"])

    return article_links

def scrape_alphabets(alphabets):
    """Scrape all articles for each alphabet letter in the list"""
    for alphabet in alphabets:
        print(f"\nProcessing articles for: {alphabet}")
        article_links = get_article_links(alphabet)

        for link in article_links:
            print(f"Extracting from: {link}")
            html = fetch_page(link)

            if html:
                title, extracted_text = extract_text(html)
                save_to_file(alphabet, title, extracted_text)