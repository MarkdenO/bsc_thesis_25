import requests
from bs4 import BeautifulSoup
from collections import defaultdict
import json
import re
import os
import time


def get_megathread_urls():
    url = "https://old.reddit.com/r/adventofcode/wiki/archives/solution_megathreads/"

    all_urls = defaultdict()

    # Define the user agent
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    }

    # Go to each url + year
    for i in range(2015, 2025):
        print(f'--- Fetching {i} ---\n')


        time.sleep(30)
        url = f"https://old.reddit.com/r/adventofcode/wiki/archives/solution_megathreads/{i}"
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        # Find div with class='content'
        content = soup.find("div", class_="content")

        # Search for 'md wiki' class inside 'wki-page-content md-container' class
        wiki_page_content = content.find("div", class_="wiki-page-content md-container")

        md_wiki = wiki_page_content.find("div", class_="md wiki")

        # Search for 'table' in 'md wiki' class
        table = md_wiki.find("table")

        # year 2023 has a slightly different format
        if i == 2023:
            all_urls[i] = get_urls_from_table(table, is_2023=True)
        else:
            all_urls[i] = get_urls_from_table(table)

    # Save all urls in a .json file
    with open("urls.json", "w") as file:
        json.dump(all_urls, file)

    return all_urls


def get_urls_from_table(table, is_2023=False):
    '''
        Get urls from table
        example: <td align="center"><a href="https://old.reddit.com/r/programming/comments/3uyl7s" rel="nofollow">1</a>*</td>
    '''
    urls = []
    for tr in table.find_all("tr"):
        for td in tr.find_all("td"):
            for a in td.find_all("a"):
                # Change URL to old.reddit.com
                if is_2023: # 2023 is somehow stored differently
                    url_end = a.get("href")
                    new_url = f"https://old.reddit.com/r/adventofcode/comments{url_end}"
                    urls.append(new_url)
                else:
                    urls.append(a.get("href").replace("www", "old"))
    return urls




def scrape_reddit():
    "Get solutions from r/adventofcode megathreads for all puzzles."

    # First, get the urls of the megathread webpages
    megathread_urls = get_megathread_urls()

    




