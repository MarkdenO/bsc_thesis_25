# Transform code into different types of code representation

import os
import re
import ast
import json
import argparse


def code_to_ast(code):
    '''Transform python code into an AST representation'''
    try:
        tree = ast.parse(code)
        return ast.dump(tree)
    except SyntaxError as e:
        print(f'Syntax error while parsing code: {e}')
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
    for year in range(2015,2016):
        data_path = f'{data_dir}/{year}.json'
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # AST
        if args.ast:
            os.makedirs('results/ast', exist_ok=True)
            ast_data = {}
            for day in range(1,3):
                day_repos = data.get(str(day), {})
                ast_data[day] = {}
                for repo_url, code in day_repos.items():
                    ast_representations = code_to_ast(code)

                    ast_data[day][repo_url] = ast_representations
            with open(f'results/ast/{year}.json', 'w') as f:
                json.dump(ast_data, f, indent=4)
        


if __name__ == "__main__":
    main()

