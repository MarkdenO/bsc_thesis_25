# Transform code into different types of code representation

import os
import re
import ast
import json
import argparse
import subprocess
import tempfile
import pandas as pd
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import tokenize
from io import BytesIO
from nltk import ngrams
from transformers import RobertaTokenizer, RobertaModel
from sklearn.feature_extraction.text import TfidfVectorizer
import torch


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
        # print(f"Modernize failed: {e}")
        return None
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


def code_to_ast(code: str) -> str:
    """Transform Python code into an AST representation, with Py2 fallback."""
    try:
        # print(code)
        return ast.parse(code)
    except SyntaxError:
        # print("SyntaxError in original code, trying to convert from Python 2 to 3:")
        code_py3 = convert_py2_to_py3(code)
        if code_py3 is None:
            return None
        try:
            return ast.parse(code_py3)
        except SyntaxError as e:
            # print(f"Even after conversion: SyntaxError: {e}")
            return None


def code_to_ngrams(code, n=3):
    """Turn python code into ngram representation"""
    tokens = code.split()
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    

def code_to_embed(code: str):
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = RobertaModel.from_pretrained("microsoft/codebert-base")
    tokens = tokenizer(code, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy().tolist() 


def code_to_tfidf(code):
    def code_tokenizer(code):
        return re.findall(r'[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|[-+*/%=(){}\[\]:.,]', code)

    vectorizer = TfidfVectorizer(tokenizer=code_tokenizer, lowercase=False)
    try:
        tfidf_matrix = vectorizer.fit_transform([code])
        
        tfidf_array = tfidf_matrix.toarray().tolist()  # Convert to list of lists
        feature_names = vectorizer.get_feature_names_out().tolist()
        
        return tfidf_array, feature_names
    except ValueError:
        return [], []



def main():
    parser = argparse.ArgumentParser(description="Process some optional flags.")
    parser.add_argument('--ast', action='store_true', help='Turn code into AST represenations')
    parser.add_argument('--cfg', action='store_true', help='Turn code into CFG represenations')
    parser.add_argument('--ngrams', action='store_true', help='Use token-level n-grams to represent code')
    parser.add_argument('--embed', action='store_true', help='Turn code into code embeddings')
    parser.add_argument('--tfidf', action='store_true', help='Turn code into TFIDF represenations')

    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)

    # Load df from labelled_data.pkl
    train_data = 'datasets/train.json'
    val_data = 'datasets/val.json'
    test_data = 'datasets/test.json'
    df_train = pd.read_json(train_data)
    df_val = pd.read_json(val_data)
    df_test = pd.read_json(test_data)

    dfs = [df_train, df_val, df_test]

    if args.ngrams:
        # Create train_ngrams.json, val_ngrams.json, test_ngrams.json
        n = 3

        # Files and dfs paired for easy iteration and saving
        file_names = ['train_ngrams.json', 'val_ngrams.json', 'test_ngrams.json']
        dfs = [df_train, df_val, df_test]

        for df, file_name in zip(dfs, file_names):
            df['ngrams'] = df['Data'].apply(lambda x: code_to_ngrams(x, n))
            df.to_json(f'datasets/{file_name}', indent=4, orient='records')
            print(f"Saved {file_name} with n-grams.")

        

if __name__ == "__main__":
    main()

