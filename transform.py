# Transform code into different types of code representation

import os
import re
import ast
import json
import argparse
import subprocess
import tempfile



def convert_py2_to_py3(code: str) -> str:
    """Convert Python 2 code to Python 3 code using python-modernize."""
    try:
        with tempfile.NamedTemporaryFile('w+', suffix='.py', delete=False) as temp:
            temp.write(code)
            temp_filename = temp.name

        subprocess.run(['python-modernize', '-w', temp_filename], check=True)

        with open(temp_filename, 'r') as f:
            converted_code = f.read()

        return converted_code
    except subprocess.CalledProcessError as e:
        print(f"Modernize failed: {e}")
        return None
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


def code_to_ast(code: str) -> str:
    """Transform Python code into an AST representation, with Py2 fallback."""
    try:
        return ast.dump(ast.parse(code))
    except SyntaxError:
        print("SyntaxError in original code, trying to convert from Python 2 to 3:")
        code_py3 = convert_py2_to_py3(code)
        if code_py3 is None:
            return None
        try:
            return ast.dump(ast.parse(code_py3))
        except SyntaxError as e:
            print(f"Even after conversion: SyntaxError: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description="Process some optional flags.")
    parser.add_argument('--ast', action='store_true', help='Turn code into AST represenations')
    parser.add_argument('--cfg', action='store_true', help='Turn code into CFG represenations')
    parser.add_argument('--ngrams', action='store_true', help='Use token-level n-grams to represent code')
    parser.add_argument('--embed', action='store_true', help='Turn code into code embeddings')
    parser.add_argument('--tfidf', action='store_true', help='Turn code into TFIDF represenations')

    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)

    data_dir = f'preprocessed_code'
    for year in range(2015,2024):
        data_path = f'{data_dir}/{year}.json'
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # AST
        if args.ast:
            os.makedirs('results/ast', exist_ok=True)
            ast_data = {}
            for day in range(1,26):
                day_repos = data.get(str(day), {})
                ast_data[day] = {}
                for repo_url, code in day_repos.items():
                    ast_representations = code_to_ast(code)

                    ast_data[day][repo_url] = ast_representations
            with open(f'results/ast/{year}.json', 'w') as f:
                json.dump(ast_data, f, indent=4)
        


if __name__ == "__main__":
    main()

