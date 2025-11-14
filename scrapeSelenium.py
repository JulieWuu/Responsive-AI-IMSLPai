from bs4 import BeautifulSoup
import pandas as pd
import requests
import time
import os
from difflib import get_close_matches
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

MASTER_COLUMNS = [
    "Work Title", "Composer", "Opus/Catalogue NumberOp./Cat. No.",
    "Key", "Movements/SectionsMov'ts/Sec's",
    "Year/Date of CompositionY/D of Comp.",
    "Composer Time PeriodComp. Period", "Piece Style",
    "Instrumentation", "Average DurationAvg. Duration"
]

def get_soup(url):
    """A helper function to fetch and parse a URL, with error handling."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser')
    except requests.exceptions.RequestException as e:
        print(f"  [Error] Could not fetch {url}: {e}")
        return None

def normalize_label(label: str, master_columns: list[str]):
    """Match a raw IMSLP field label to the closest master column."""
    label = label.strip().replace("\n", " ").replace("  ", " ")
    # find the best fuzzy match (case-insensitive)
    matches = get_close_matches(label, master_columns, n=1, cutoff=0.6)
    return matches[0] if matches else None


def scrape_piece_info(piece_url) -> dict:
    soup = get_soup(piece_url)

    info_groups = soup.find_all('div', class_='wi_body', style='width:100%')
    if not info_groups:
        print("Could not find any information groups with class 'wi_body'. The HTML structure might have changed.")
        return {}

    print(f"Found a total of {len(info_groups)} information groups.")

    piece_info = {}
    ordered_info = {}

    for i, group in enumerate(info_groups):
        all_entries = group.find_all('tr')
        print(f"Found {len(all_entries)} entries.")

        for entry in all_entries:
            label = entry.find('th')
            content = entry.find('td')
            if not label or not content:
                continue

            raw_label = label.get_text(strip=True)
            normalized_label = normalize_label(raw_label, MASTER_COLUMNS)

            if normalized_label:
                piece_info[normalized_label] = content.get_text(strip=True)

        ordered_info = {col: piece_info.get(col, None) for col in MASTER_COLUMNS}

        print(f"Scraping of this info group completed. A total of {len(piece_info)} pieces of information collected.")
        # Let's add a small delay to be polite to the server
        time.sleep(0.05)

    return ordered_info

def scrape_imslp_composers(composer_url, output_csv="premium_data.csv"):
    """
    Scrapes all composer names and URLs from IMSLP using Selenium to handle
    JavaScript-based pagination ("next 200" button).
    """
    print(f"Starting scrape of: {composer_url} using Selenium.")

    # --- Step 1: Set up Selenium WebDriver ---
    # webdriver-manager will automatically download and manage the correct driver for your browser.
    try:
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
        driver.get(composer_url)
    except Exception as e:
        print(f"Error setting up Selenium WebDriver: {e}")
        print("Please ensure Google Chrome is installed on your system.")
        return

    all_info = []

    try:
        page_count = 1
        end = False
        while True:
            if end:
                break
            print(f"Scraping page {page_count}...")

            # Wait for the main content table to be present before scraping
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table[width='100%']"))
            )

            # --- Step 2: Parse the current page source with BeautifulSoup ---
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            pieces_links = soup.find('table', width='100%').find_all('a')

            if not pieces_links:
                print("Could not find composer links on the page.")
                break

            for i, link in enumerate(pieces_links):
                if i >= 300:
                    break

                piece_name = link.get('title')
                piece_url = "https://imslp.org" + link.get('href')

                # Let's add a small delay to be polite to the server
                time.sleep(0.05)

                # For this basic scraper, we will just get the name and URL.
                # A more advanced scraper would visit `composer_page_url` to get more details.
                piece = {'piece': piece_name, 'piece_url': piece_url}
                if scrape_piece_info(piece_url) is not None:
                    piece.update(scrape_piece_info(piece_url))

                all_info.append(piece)
                print(f"Scraped ({i + 1}/{len(pieces_links)}): {piece_name}")

            print(
                f"  > Found {len(pieces_links)} links on this page. Total unique pieces so far: {len(all_info)}")

            # --- Step 3: Find and click the "next 200" button ---
            if page_count >= 10:
                break

            try:
                # Use Selenium to find the link by its exact text
                next_button = driver.find_element(By.PARTIAL_LINK_TEXT, 'next')

                # Scroll the button into view and click it
                driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
                time.sleep(0.5)  # small delay before click
                next_button.click()

                page_count += 1
                # Wait a moment for the new content to load
                time.sleep(1)

            except NoSuchElementException:
                # This exception means the button wasn't found, so we're on the last page.
                print("\n'next' button not found. Assuming last page reached.")
                break


    except TimeoutException:
        print("Timed out waiting for page content to load. Aborting.")
    # except Exception as e:
        # print(f"An unexpected error occurred during scraping: {e}")
    finally:
        # --- Step 4: Important - close the browser ---
        driver.quit()

    if not all_info:
        print("No pieces were scraped.")
        return

    # --- Step 5: Convert to DataFrame and save to CSV ---
    print(f"\nScraping complete. Total unique pieces found: {len(all_info)}")

    df = pd.DataFrame(all_info)

    if os.path.exists(output_csv):
        old = pd.read_csv(output_csv, usecols=["piece_url"])
        df = df[~df["piece_url"].isin(old["piece_url"])]
        df.to_csv(output_csv, mode='a', index=False, header=False)
    else:
        df.to_csv(output_csv, mode='w', index=False, header=True)
    print(f"File '{output_csv}' saved successfully.")


if __name__ == '__main__':
    telemann = "https://imslp.org/wiki/Category:Telemann%2C_Georg_Philipp"
    scrape_imslp_composers(telemann)

