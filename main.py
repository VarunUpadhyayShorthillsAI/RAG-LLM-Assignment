import requests
from bs4 import BeautifulSoup
import os

BASE_URL = "https://medlineplus.gov/ency/"

def fetch_page(url):
    response = requests.get(url)
    return response.text if response.status_code == 200 else None

def extract_text(html):
    soup = BeautifulSoup(html, "html.parser")

    # Extract article title
    title_tag = soup.find("h1", class_="with-also", itemprop="name")
    article_title = title_tag.get_text(strip=True) if title_tag else "Title not found"

    extracted_text = {"Title": article_title}

    # Extract all sections dynamically
    for section in soup.find_all("div", class_="section"):
        title_div = section.find("div", class_="section-title")
        body_div = section.find("div", class_="section-body")

        if title_div and body_div:
            section_title = title_div.get_text(strip=True)
            section_content = body_div.get_text(" ", strip=True)

            # Exclude unwanted sections
            if any(exclude in section_title.lower() for exclude in ["images", "references", "review date"]):
                continue

            extracted_text[section_title] = section_content

    return extracted_text

def save_to_txt(data, filename):
    """Save extracted data to a text file."""
    os.makedirs("articles", exist_ok=True)  # Ensure the folder exists
    filepath = os.path.join("articles", f"{filename}.txt")

    with open(filepath, "w", encoding="utf-8") as file:
        for section, text in data.items():
            file.write(f"{section}\n{text}\n\n")

    print(f"Saved: {filepath}")

def get_article_links(alphabet):
    url = f"{BASE_URL}encyclopedia_{alphabet}.htm"
    html = fetch_page(url)
    
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    article_links = []

    for li in soup.select("#mplus-content li"):
        if not li.get("class"):  # Ensure <li> has no class
            a_tag = li.find("a", href=True)
            if a_tag and a_tag["href"].startswith("article/"):  # Only take article links
                article_links.append(BASE_URL + a_tag["href"])

    return article_links

if __name__ == "__main__":
    alphabet = input("Enter the alphabet: ").strip().upper()
    article_links = get_article_links(alphabet)

    if not article_links:
        print("No articles found.")
    else:
        for link in article_links:
            print(f"\nExtracting from: {link}")
            html = fetch_page(link)

            if html:
                extracted_text = extract_text(html)
                article_name = extracted_text.get("Title", "Unknown_Article").replace(" ", "_")
                save_to_txt(extracted_text, article_name)
