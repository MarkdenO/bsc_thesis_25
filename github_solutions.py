import requests
from collections import defaultdict
from bs4 import BeautifulSoup
import json
import re
import os
import time
from docx import Document


def get_repos_from_file(file_path):
    """
    Reads a .docx file and extracts GitHub repository URLs from it.
    :param file_path: Path to the .docx file.
    :return: List of GitHub repository URLs.
    """
    
    doc = Document(file_path)
    urls = []
    for para in doc.paragraphs:
        if 'github.com' in para.text:
            urls.append(para.text)

    data = defaultdict(lambda: defaultdict(list)) # {2015: 1: {repo1, repo2}}
    
    # Search through repositories
    for year in range(2021, 2024):
        for day in range(1, 26):
            data[year][day] = {}
            for url in urls:
                username, code = find_aoc_python_solution(url, year, day)
                if username is not None and code is not None:
                    data[year][day][username] = code
            os.makedirs(f'data/repos', exist_ok=True)
        with open(f'data/repos/extra_repos_{year}.json', 'a') as f:
            json.dump(data[year], f, indent=4)
    
    with open('data/repos/extra_repos_all_years.json', 'w') as f:
        json.dump(data, f, indent=4)



def scrape_aoc_leaderboard():
    """
    Get solutions from all years and days from the adventofcode leaderboard and stored them in a JSON file.
    """
    github_data = defaultdict(lambda: defaultdict(list)) # {2015: 1: {repo1, repo2}}
    BASE_URL = "https://adventofcode.com/{year}/leaderboard/day/{day}"

    for year in range(2017, 2024):
        for day in range(1,25):
            github_data[year][day] = {}
            print(f'Searching for solutions in Github repositories from year {year} day {day}')
            url = BASE_URL.format(year=year, day=day)
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all github links on the leaderboard page
            links = soup.find_all('a', href=True)
            for link in links:
                if 'github.com' in link['href'] and link['href'] not in github_data.get(str(year), {}).get(str(day), []):
                    print('\n')
                    # github_data[year][day].append(link['href'])
                    # Search for solution for year and day in link['href']
                    result = search_solution(link['href'], year, day)
                    # print(result)
                    if result is not None:
                        user, solution_url = result
                        solution_code = get_python_code(solution_url)
                        github_data[year][day][solution_url] = solution_code
        os.makedirs(f'code/{year}', exist_ok=True)
        with open(f'code/{year}/data_{year}.json', 'w') as f:
            json.dump(github_data[year], f, indent=4) 
    
    # Save to json
    os.makedirs('code', exist_ok=True)
    with open('code/data.json', 'w') as f:
        json.dump(github_data, f, indent=4)


def search_solution(url, year, day, token=os.getenv('GITHUB_TOKEN')):
    """
    From the url from a Github user, first search for the users' AoC repo for that year, 
    than search for the solution file from that specific year and day in the repo.

    :param: url (str): The url for the users AoC repo. For example: https://www.github.com/username123/advent-of-code-2015
    :param: year (int): The year of the solution/puzzle
    :param: day (int): The dya of the solution/puzzle
    :param: token(str): The Github token for authenticated users
    """
    # Use the new AoC solution finder function for unified logic
    return find_aoc_python_solution(url, year, day, token)


def get_solution_code(user, repo_name, year, day, token=os.getenv('GITHUB_TOKEN')):
    
    search_paths = [
        "",  # Current directory
        "solutions",  # Solutions folder
        "src",  # Source folder
        "day" + str(day),  # Day-specific folder
        str(year) + "/day" + str(day),  # Year/Day folder
        str(year) + "/" + str(day),  # Year/Day folder without 'day'
        "aoc" + str(year), # aoc2015
        "AoC" + str(year) # AoC2015
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
        print(solution_url)
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
    

def find_aoc_python_solution(repo_url, year, day, token=os.getenv('GITHUB_TOKEN')):
    """
    Given a GitHub repository URL, AoC year, and day, check if there is a Python solution for that day/year.
    Returns (username, code) if found, else (None, None).
    Handles errors with informative messages.
    """
    import re
    import requests
    import base64

    # 1. Parse the GitHub URL
    url_pattern = re.compile(
        r"https?://github\.com/(?P<username>[^/]+)/(?P<repo>[^/]+)(?:/|$)"
    )
    m = url_pattern.match(repo_url.strip())
    if not m:
        print(f"Error: Malformed GitHub URL: {repo_url}")
        return None, None
    username, repo = m.group("username"), m.group("repo")

    # 2. Prepare API headers
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "MarkdenO"
    }
    if token:
        headers["Authorization"] = f"token {token}"
        headers["User-Agent"] = "MarkdenO"
        headers["Accept"] = "application/vnd.github.v3+json"
        headers["X-GitHub-Api-Version"] = "2022-11-28"

    # 3. List repo contents recursively
    api_url = f"https://api.github.com/repos/{username}/{repo}/git/trees/HEAD?recursive=1"
    try:
        resp = requests.get(api_url, headers=headers, timeout=20)
        if resp.status_code == 404:
            print(f"Error: Repository not found: {repo_url}")
            return None, None
        if resp.status_code != 200:
            print(f"Error: GitHub API error ({resp.status_code}): {resp.text}")
            return None, None
        tree = resp.json().get("tree", [])
    except Exception as e:
        print(f"Error: Failed to fetch repo contents: {e}")
        return None, None

    # 4. Build possible filename patterns
    padded_day = f"{int(day):02d}"
    patterns = [
        re.compile(rf"(aoc)?{year}/day[_-]?{padded_day}\.py$", re.IGNORECASE),
        re.compile(rf"(aoc)?{year}/day[_-]?{int(day)}\.py$", re.IGNORECASE),
        re.compile(rf"{year}/day[_-]?{padded_day}\.py$", re.IGNORECASE),
        re.compile(rf"{year}/day[_-]?{int(day)}\.py$", re.IGNORECASE),
        re.compile(rf"day[_-]?{year}_{padded_day}\.py$", re.IGNORECASE),
        re.compile(rf"day[_-]?{year}_{int(day)}\.py$", re.IGNORECASE),
        re.compile(rf"day[_-]?{padded_day}\.py$", re.IGNORECASE),
        re.compile(rf"day[_-]?{int(day)}\.py$", re.IGNORECASE),
        re.compile(rf"{year}/\d{{1,2}}/day[_-]?{padded_day}\.py$", re.IGNORECASE),
        re.compile(rf"{year}/\d{{1,2}}/day[_-]?{int(day)}\.py$", re.IGNORECASE),
        re.compile(rf"{year}/day{padded_day}\.py$", re.IGNORECASE),
        re.compile(rf"{year}/day{int(day)}\.py$", re.IGNORECASE),
        re.compile(rf"{year}/\d{{1,2}}\.py$", re.IGNORECASE),
        re.compile(rf"{year}/day{padded_day}/.*\.py$", re.IGNORECASE),
        re.compile(rf"{year}/day{int(day)}/.*\.py$", re.IGNORECASE),
        re.compile(rf"{year}/.*day[_-]?{padded_day}\.py$", re.IGNORECASE),
        re.compile(rf"{year}/.*day[_-]?{int(day)}\.py$", re.IGNORECASE),
        re.compile(rf".*day[_-]?{padded_day}\.py$", re.IGNORECASE),
        re.compile(rf".*day[_-]?{int(day)}\.py$", re.IGNORECASE),
    ]

    # 5. Prioritize files in year-named directories
    candidates = []
    for obj in tree:
        if obj.get("type") != "blob":
            continue
        path = obj.get("path", "")
        for pat in patterns:
            if pat.search(path):
                candidates.append(path)
                break

    # Prefer files in year-named directories
    prioritized = [p for p in candidates if f"{year}/" in p or f"aoc{year}/" in p.lower()]
    search_order = prioritized + [p for p in candidates if p not in prioritized]

    if not search_order:
        print(f"No Python solution found for year {year} day {day} in {repo_url}")
        return None, None

    # 6. Fetch and decode the code
    for path in search_order:
        file_api = f"https://api.github.com/repos/{username}/{repo}/contents/{path}"
        try:
            file_resp = requests.get(file_api, headers=headers, timeout=20)
            if file_resp.status_code == 200:
                file_json = file_resp.json()
                if file_json.get("encoding") == "base64":
                    code = base64.b64decode(file_json["content"]).decode("utf-8")
                else:
                    code = file_json.get("content", "")
                return username, code
            else:
                continue
        except Exception as e:
            print(f"Error fetching file {path}: {e}")
            continue

    print(f"No accessible Python solution found for year {year} day {day} in {repo_url}")
    return None, None