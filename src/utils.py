def input_alphabet():
    """Prompts the user to input an alphabet for scraping."""
    alphabet = input("Enter the alphabet to scrape (e.g., A, B, C) or 'ALL' for all alphabets: ").strip().upper()
    return alphabet