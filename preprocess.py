import argparse
import json
import re
import os
from pygments.lexers import guess_lexer



def preprocess_github(data):
    pp_data = {}
    for day in range(1, 26):
        day_repos = data.get(str(day), {})
        pp_data[day] = {}
        for repo_url, code in day_repos.items():
            code = preprocess_code_snippet(code)
            pp_data[day][repo_url] = code

    return pp_data


def preprocess_code_snippet(code):
    # Remove comments
    # code = re.sub(r'#.*', '', code)


    # Remove docstrings
    code = re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', '', code, flags=re.DOTALL)

    # Remove imports
    code = re.sub(r'^\s*(import|from)\s+[^\n]+', '', code, flags=re.MULTILINE)

    # # Generalize variable names
    # code = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', lambda m: 'var' if not m.group(0) in ('def', 'return', 'if', 'else', 'for', 'in', 'while', 'print', 'range', 'len', 'input') else m.group(0), code)

    # # Generalize function names
    # code = re.sub(r'def\s+(\w+)', r'def func', code)

    # Remove extra whitespaces
    # lines = [line.strip() for line in code.splitlines() if line.strip()]
    # code = '\n'.join(lines)

    return code


def preprocess_reddit(data):
    pp_data = {}

    for day in range(1, 26):
        day_str = str(day)
        comments = data.get(day_str, {}).get('comments', [])

        for comment in comments:

            author = comment.get('author', 'unknown')
            code = comment.get('code', '')


            code = preprocess_code_snippet(code)

            if day not in pp_data:
                pp_data[day] = {}
            pp_data[day][author] = code

    return pp_data


def main():
    for year in range(2015,2024):
        # Preprocess Github data
        github_repos_path = f'code/{year}/data_{year}.json'
        with open(github_repos_path, 'r') as f:
            data = json.load(f)
        pp_data = preprocess_github(data)
        
        # Write preprocessed data to file
        os.makedirs('preprocessed_code', exist_ok=True)
        with open(f'preprocessed_code/lb_{year}.json', 'w') as f:
            json.dump(pp_data, f, indent=4)

        # Preprocess Reddit data
        reddit_solutions_path = f'data/reddit_solutions.json'
        with open(reddit_solutions_path, 'r') as f:
            data = json.load(f)
        pp_data_reddit = preprocess_reddit(data[str(year)])

        # Write preprocessed data to file
        os.makedirs('preprocessed_code', exist_ok=True)
        with open(f'preprocessed_code/reddit_{year}.json', 'w') as f:
            json.dump(pp_data_reddit, f, indent=4)






if __name__ == '__main__':
    main()