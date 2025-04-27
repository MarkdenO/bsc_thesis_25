import requests
from bs4 import BeautifulSoup
from collections import defaultdict
import json
import re
import os
import time
from pygments.lexers import guess_lexer
from pygments.lexers import PythonLexer
from pygments.util import ClassNotFound
from collections import defaultdict
from typing import List, Dict
from pathlib import Path



headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    }


def get_megathread_urls():
    url = "https://old.reddit.com/r/adventofcode/wiki/archives/solution_megathreads/"

    all_urls = defaultdict()

    # Define the user agent

    # Go to each url + year
    for i in range(2015, 2025):
        print(f'--- Fetching urls for {i} ---\n')


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
    os.makedirs('data', exist_ok=True)
    with open("data/megathread_urls.json", "w") as file:
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


def get_solutions(urls):
    all_solutions = defaultdict(dict)

    for year, days in urls.items():

        print(f'Searching for Python solutions in megathreads for {year}')

        for i, url in enumerate(days, start=1):
            print(f"Fetching comments for {year} day {i}...")
            scrape_url = f'{url}/{year}_day_{i}_solutions/?sort=top&limit=500'
            print(scrape_url)
            comments_data = scrape_comments(scrape_url)
            if comments_data:
                # Make sure directory exists
                if not os.path.exists("data/solution_threads"):
                    os.makedirs("data/solution_threads")
                
                with open(f"data/solution_threads/{year}_{i}_solution_thread.json", "w", encoding="utf-8") as file:
                    json.dump(comments_data, file, indent=2, ensure_ascii=False)
                print(f"Saved {year} day {i} solution thread to {year}_{i}_solution_thread.json")

                all_solutions[year][i] = comments_data
        
        print(f'Time to sleep 5 minutes for {int(year)+1}')
        time.sleep(300) # Sleep 5 minutes to avoid being detected as a bot
    
    return all_solutions


def scrape_comments(url: str) -> Dict:
    """Scrape all Python code comments from a given Reddit thread."""
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Failed to fetch thread {url}: {response.status_code}")
        return {}

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract the puzzle_id from the URL using regex
    pattern = r"(\d{4})_day_(\d{1,2})"
    match = re.search(pattern, url)
    puzzle_id = f"{match.group(1)}_{match.group(2)}" if match else None

    thread_data = {"puzzle_id": puzzle_id, "url": url, "comments": []}

    # Extract comments
    comment_divs = soup.find_all("div", class_=lambda c: c and "comment" in c)
    for comment in comment_divs:
        author_tag = comment.find("a", class_=lambda c: c and "author" in c)
        author = author_tag.get_text(strip=True) if author_tag else None

        comment_body_parts = comment.select(".md > p")
        comment_body = "\n".join(p.get_text(strip=True) for p in comment_body_parts)

        code_tag = comment.select_one(".md pre code")
        code = code_tag.get_text(strip=True) if code_tag else None

        score_tag = comment.find("span", class_=lambda c: c and "score" in c)
        upvotes = score_tag.get_text(strip=True) if score_tag else None

        language = guess_lexer(code).name if code else None
        if language != "Python":
            continue

        thread_data["comments"].append({
            "author": author,
            "text": comment_body,
            "code": code,
            "upvotes": upvotes,
            "language": language,
        })

    print(f"Scraped {len(thread_data['comments'])} comments from {url}")
    return thread_data



def scrape_reddit():
    "Get solutions from r/adventofcode megathreads for all puzzles."

    # First, get the urls of the megathread webpages

    # If megathread urls are already obtained, skip this step
    if Path("data/megathread_urls.json").exists() and Path("data/megathread_urls.json").stat().st_size > 0:
        print(f'Megathread urls already exist, skipping retrieving megathread urls.')
        with open('data/megathread_urls.json', 'r') as f:
            megathread_urls = json.load(f)
    else:
        megathread_urls = get_megathread_urls()
    
    solutions = get_solutions(megathread_urls)

    with open('data/reddit_solutions.json', 'w') as f:
        json.dump(solutions, f, indent=4)






