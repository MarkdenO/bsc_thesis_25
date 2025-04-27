import argparse
import json
import re


def preprocess_data(data):
    pp_data = {}
    for day in range(1, 5):
        day_repos = data.get(str(day), {})
        pp_data[day] = {}
        for repo_url, code in day_repos.items():
            # Remove comments
            code = re.sub(r'#.*', '', code)

            # Remove docstrings
            code = re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', '', code, flags=re.DOTALL)

            # Remove imports
            code = re.sub(r'^\s*(import|from)\s+[^\n]+', '', code, flags=re.MULTILINE)

            # # Generalize variable names
            # code = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', lambda m: 'var' if not m.group(0) in ('def', 'return', 'if', 'else', 'for', 'in', 'while', 'print', 'range', 'len', 'input') else m.group(0), code)

            # # Generalize function names
            # code = re.sub(r'def\s+(\w+)', r'def func', code)

            # Remove extra whitespaces
            lines = [line.strip() for line in code.splitlines() if line.strip()]
            code = '\n'.join(lines)

            pp_data[day][repo_url] = code

    return pp_data



def main():
    for year in range(2015,2016):
        github_repos_path = f'code/{year}/data_{year}.json'
        with open(github_repos_path, 'r') as f:
            data = json.load(f)
        pp_data = preprocess_data(data)
        print(pp_data)



if __name__ == '__main__':
    main()