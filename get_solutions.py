import argparse
import requests
from collections import defaultdict
from bs4 import BeautifulSoup
import json
import re
import os
import time


def scrape_aoc_leaderboard():
    github_data = defaultdict(lambda: defaultdict(list)) # {2015: 1: {repo1, repo2}}
    BASE_URL = "https://adventofcode.com/{year}/leaderboard/day/{day}"

    for year in range(2015, 2016):
        for day in range(1,2):
            github_data[year][day] = {}
            print(f'Searching for solutions in Github repositories from year {year} day {day}')
            url = BASE_URL.format(year=year, day=day)
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all github links on the leaderboard page
            links = soup.find_all('a', href=True)
            for link in links:
                if 'github.com' in link['href'] and link['href'] not in github_data.get(str(year), {}).get(str(day), []):
                    # github_data[year][day].append(link['href'])
                    # Search for solution for year and day in link['href']
                    result = search_solution(link['href'], year, day)
                    if result is not None:
                        user, solution_url = result
                        solution_code = get_python_code(solution_url)
                        github_data[year][day][solution_url] = solution_code
    
    # Save to json
    with open('data.json', 'w') as f:
        json.dump(github_data, f, indent=4)


def search_solution(url, year, day, token=os.getenv('GITHUB_TOKEN')):
    match = re.match(r"https?://github\.com/([^/?#]+)", url)
    if not match:
        print(f"Warning: Invalid GitHub profile URL pattern: {url}")
        return []
    username = match.group(1)

    api_url = f"https://api.github.com/users/{username}/repos"

    # Set up headers for authentication
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"
        headers["User-Agent"] = "MarkdenO"
        headers["Accept"] = "application/vnd.github.v3+json"
        headers["X-GitHub-Api-Version"] = "2022-11-28"

    response = requests.get(api_url, headers=headers)

    if response.status_code != 200:
        raise Exception(f"GitHub API error: {response.status_code} - {response.text}")

    # All repos from user
    repos = response.json()

    # Keywords to identify Advent of Code repos
    keywords = ['advent', 'aoc', 'advent-of-code', 'adventofcode']

    aoc_repos = []
    if isinstance(repos, list):
        for repo in repos:
            if isinstance(repo, dict) and 'name' in repo:
                name = repo['name'].lower()
                description = (repo.get('description') or "").lower()

                # Check if repo is year-specific or if repo has folder for every year
                year_specific = any(str(year) in name for year in range(2015, 2025)) or (any(str(year) in name for year in range(15, 25)))
                if year_specific:
                    # Only use year-specific repo
                    if any(keyword in name or keyword in description for keyword in keywords):
                        aoc_repos.append({
                            'name': repo['name'],
                            'html_url': repo['html_url'],
                            'description': repo.get('description'),
                            'year_specific': True
                        })
                    continue # If repo is year-specific, skip the rest

            else:
                if any(keyword in name or keyword in description for keyword in keywords):
                    aoc_repos.append({
                        'name': repo['name'],
                        'html_url': repo['html_url'],
                        'description': repo.get('description'),
                        'year_specific': False
                    })
  

    # Filter out repos that are for another year
    if len(aoc_repos) > 1:
        aoc_repos = [repo for repo in aoc_repos if str(year) in repo['name'] or str(year) in repo['description']]
        # print(aoc_repo)
    
    if aoc_repos == []:
        return None

    # Find solution file in repo
    solution_code = get_solution_code(username, aoc_repos[0]['name'], year, day)
    if solution_code is None:
        return None

    return username, solution_code


def get_solution_code(user, repo_name, year, day, token=os.getenv('GITHUB_TOKEN')):
    
    search_paths = [
        "",  # Current directory
        "solutions",  # Solutions folder
        "src",  # Source folder
        "day" + str(day),  # Day-specific folder
        str(year) + "/day" + str(day),  # Year/Day folder
        str(year) + "/" + str(day),  # Year/Day folder without 'day'
    ]

    for path in search_paths:
        api_url = f"https://api.github.com/repos/{user}/{repo_name}/contents/{path}"
        contents = _make_github_api_request(api_url, token)

        if not contents:
            print(f"No content found in {api_url}")
            return None
        
        solution = search_contents(contents, user, repo_name, day)

        if solution:
            return solution
        
    print(f"No solution file found for Year {year} Day {day} in repo {user}/{repo_name}")
    return None


def search_contents(contents, user, repo_name, day, token=os.getenv('GITHUB_TOKEN')):
    padded_day = f"{day:02d}"
    day_str = str(day)
    # 
    # regex patterns for day matching in filenames/dirs
    # Allows dayXX, dayX, XX, X potentially with separators
    file_pattern = re.compile(r"(^|[\D_])(" + day_str + r"|" + padded_day + r")($|[\D_])?\.py$", re.IGNORECASE)
    dir_pattern = re.compile(r"^day(" + day_str + r"|" + padded_day + r")$|^(" + day_str + r"|" + padded_day + r")$", re.IGNORECASE)
    generic_py_pattern = re.compile(r"\.py$", re.IGNORECASE)

    matching_dirs = []
    # check files directly in the current search_path
    for item in contents:
        if item.get('type') == 'file' and item.get('name', '').endswith('.py'):
            # Check if filename matches the day pattern
            if file_pattern.search(item['name']):
                print(f"  DEBUG: Found matching file directly: {item['path']}")
                return item.get('html_url') # Found a likely candidate

    # check directories matching the day pattern
    for item in contents:
        if item.get('type') == 'dir':
            if dir_pattern.match(item.get('name', '')):
                print(f"  DEBUG: Found potential day directory: {item['path']}")
                matching_dirs.append(item['path'])

    # search inside the matching directories found
    for dir_path in matching_dirs:
        print(f"  DEBUG: Searching inside directory: {dir_path}")
        dir_api_url = f"https://api.github.com/repos/{user}/{repo_name}/contents/{dir_path}"
        dir_contents = _make_github_api_request(dir_api_url, token)
        if dir_contents and isinstance(dir_contents, list):
            for item in dir_contents:
                 # Look for any .py file inside the day-specific folder
                if item.get('type') == 'file' and generic_py_pattern.search(item.get('name', '')):
                    print(f"  DEBUG: Found .py file inside day directory: {item['path']}")
                    return item.get('html_url') # Return the first .py file found inside

    print(f"  DEBUG: No specific Python file found for day {day} in path '{user}/{repo_name}' or its subdirs.")
    return None # No suitable file found
        

def get_python_code(solution_url, token=os.getenv("GITHUB_TOKEN")):
    """
        Downloads and returns the content of a Python script from a GitHub blob URL.
        :param: solution_url (str): The GitHub blob URL of the Python file.
        :returns: The content of the Python script as a string.
    """
    if 'github.com' not in solution_url or '/blob/' not in solution_url:
        raise ValueError("Invalid GitHub blob URL")
    
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "MarkdenO"
    }
    if token:
        headers["Authorization"] = f"token {token}"
        headers["User-Agent"] = "MarkdenO"
        headers["Accept"] = "application/vnd.github.v3+json"
        headers["X-GitHub-Api-Version"] = "2022-11-28"
    else:
        print("No Github token set in environment.")

    raw_url = solution_url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')

    response = requests.get(raw_url)
    response.raise_for_status()

    return response.text



def _make_github_api_request(api_url, token):
    """Function to make authenticated GitHub API requests."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "MarkdenO"
    }
    if token:
        headers["Authorization"] = f"token {token}"
        headers["User-Agent"] = "MarkdenO"
        headers["Accept"] = "application/vnd.github.v3+json"
        headers["X-GitHub-Api-Version"] = "2022-11-28"

    try:
        response = requests.get(api_url, headers=headers, timeout=15)
        if response.status_code == 404:
            print(f"  DEBUG: API URL not found (404): {api_url}")
            return None
        elif response.status_code == 403:
            # Handle rate limit or forbidden access
            print(f"  WARNING: GitHub API forbidden (403) for {api_url}.")
            print(f"  Rate Limit Remaining: {response.headers.get('X-RateLimit-Remaining')}")
            print(f"  Message: {response.text}")
            time.sleep(3600) # Sleep for 1 hr to fix rate limit
            response = requests.get(api_url, headers=headers, timeout=15)
            if response.status_code == 403:
                print(f"  ERROR: GitHub API still forbidden (403) after sleep for {api_url}.")
                return None
            elif response.status_code == 404:
                print(f"  DEBUG: API URL not found (404): {api_url}")
                return None

        response.raise_for_status() # Raise other HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"  ERROR: GitHub API request failed for {api_url}: {e}")
        return None
    except json.JSONDecodeError:
        print(f"  ERROR: Failed to decode JSON from {api_url}")
        return None



def main():
    parser = argparse.ArgumentParser(description="Process some optional flags.")
    parser.add_argument('--leaderboard', action='store_true', help='Get AdventOfCode leaderboard results')
    parser.add_argument('--reddit', action='store_true', help='Fetch data from r/AdventOfCode subreddit')

    args = parser.parse_args()

    if args.leaderboard:
        print("AoC leaderboard flag is set.\nScraping AoC leaderboard")
        scrape_aoc_leaderboard()

    if args.reddit:
        print("Reddit flag is set.")
        # Add Reddit route


if __name__ == "__main__":
    main()