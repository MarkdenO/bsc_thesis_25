import argparse
import json
import re
import os
from pygments.lexers import guess_lexer
import io, tokenize


def remove_comments_and_docstrings(source):
    """
    Returns 'source' minus comments and docstrings.
    """
    io_obj = io.StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    try:
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            # The following two conditionals preserve indentation.
            # This is necessary because we're not using tokenize.untokenize()
            # (because it spits out code with copious amounts of oddly-placed
            # whitespace).
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
            # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        # Note regarding NEWLINE vs NL: The tokenize module
                        # differentiates between newlines that start a new statement
                        # and newlines inside of operators such as parens, brackes,
                        # and curly braces.  Newlines inside of operators are
                        # NEWLINE and newlines that start new code are NL.
                        # Catch whole-module docstrings:
                        if start_col > 0:
                            # Unlabelled indentation means we're inside an operator
                            out += token_string
                        # Note regarding the INDENT token: The tokenize module does
                        # not label indentation inside of an operator (parens,
                        # brackets, and curly braces) as actual indentation.
                        # For example:
                        # def foo():
                        #     "The spaces before this docstring are tokenize.INDENT"
                        #     test = [
                        #         "The spaces before this string do not get a token"
                        #     ]
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        return out
    except (tokenize.TokenError, IndentationError):
        return None
    


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

    # Remove imports
    code = re.sub(r'^\s*(import|from)\s+[^\n]+', '', code, flags=re.MULTILINE)

    code = remove_comments_and_docstrings(code)

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


def split_all_data(preprocessed_dir='preprocessed_code', output_dir='datasets'):
    import glob
    from collections import defaultdict
    import random

    all_samples = []

    for file_path in glob.glob(os.path.join(preprocessed_dir, '*.json')):
        filename = os.path.basename(file_path)
        year = int(re.search(r'\d{4}', filename).group())
        source_type = 'reddit' if 'reddit' in filename else 'leaderboard' if 'lb' in filename else 'extra'

        with open(file_path, 'r') as f:
            data = json.load(f)

        for day_str, day_data in data.items():
            day = int(day_str)
            for identifier, code in day_data.items():
                if code:
                    all_samples.append({
                        'code': code,
                        'source': source_type,
                        'year': year,
                        'day': day,
                        'id': identifier
                    })

    # Shuffle and split
    random.shuffle(all_samples)
    total = len(all_samples)
    train_end = int(0.8 * total)
    val_end = train_end + int(0.1 * total)

    train_data = all_samples[:train_end]
    val_data = all_samples[train_end:val_end]
    test_data = all_samples[val_end:]

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(train_data, f, indent=2)
    with open(os.path.join(output_dir, 'val.json'), 'w') as f:
        json.dump(val_data, f, indent=2)
    with open(os.path.join(output_dir, 'test.json'), 'w') as f:
        json.dump(test_data, f, indent=2)

    print(f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test.")


def main():
    for year in range(2015,2024):
        # Preprocess AoC leaderboard data
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

        # Preprocess extra repositories
        extra_repos_path = f'data/repos/extra_repos_{year}.json'
        with open(extra_repos_path, 'r') as f:
            data = json.load(f)
        pp_data_extra = preprocess_github(data)

        # Write preprocessed data to file
        os.makedirs('preprocessed_code', exist_ok=True)
        with open(f'preprocessed_code/extra_repos_{year}.json', 'w') as f:
            json.dump(pp_data_extra, f, indent=4)

    # Split all data into train, validation, and test sets
    split_all_data(preprocessed_dir='preprocessed_code', output_dir='datasets')
    print("Preprocessing complete.")





if __name__ == '__main__':
    main()