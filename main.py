import os
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://medlineplus.gov/ency/"

def fetch_page(url):
    response = requests.get(url)
    return response.text if response.status_code == 200 else None

def extract_text(html):
    soup = BeautifulSoup(html, "html.parser")

    # Extract article title
    title_tag = soup.find("h1", class_="with-also", itemprop="name")
    article_title = title_tag.get_text(strip=True) if title_tag else "Untitled"

    # Remove any invalid filename characters
    safe_title = "".join(c for c in article_title if c.isalnum() or c in " _-").strip()

    extracted_text = {"Title": safe_title}

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

    return safe_title, extracted_text

def save_to_file(alphabet, title, content):
    folder_path = os.path.join("articles", alphabet)  # Create folder per alphabet
    os.makedirs(folder_path, exist_ok=True)  # Ensure folder exists

    file_path = os.path.join(folder_path, f"{title}.txt")

    with open(file_path, "w", encoding="utf-8") as file:
        for section, text in content.items():
            file.write(f"\n{section}\n{text}\n")  # Remove the underline

    print(f"Saved: {file_path}")


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
                title, extracted_text = extract_text(html)
                save_to_file(alphabet, title, extracted_text)
