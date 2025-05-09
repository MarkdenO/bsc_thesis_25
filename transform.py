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
        print(f"Modernize failed: {e}")
        return None
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


def code_to_ast(code: str) -> str:
    """Transform Python code into an AST representation, with Py2 fallback."""
    try:
        # print(code)
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


def cfg_to_json(graph, func_name):
    nodes = [{"id": str(n), "label": graph.nodes[n]["label"]} for n in graph.nodes]
    edges = [[str(u), str(v)] for u, v in graph.edges]
    return {"function_name": func_name, "nodes": nodes, "edges": edges}

class CFGBuilder(ast.NodeVisitor):
    def __init__(self):
        self.graph = nx.DiGraph()
        self.counter = 0
        self.last_node = None

    def _add_node(self, label):
        node_id = self.counter
        self.graph.add_node(node_id, label=label)
        if self.last_node is not None:
            self.graph.add_edge(self.last_node, node_id)
        self.last_node = node_id
        self.counter += 1
        return node_id

    def visit_FunctionDef(self, node):
        self._add_node(f"Function: {node.name}")
        for stmt in node.body:
            self.visit(stmt)

    def visit_If(self, node):
        cond_id = self._add_node("If condition")
        prev = self.last_node
        for stmt in node.body:
            self.visit(stmt)
        then_id = self.last_node
        self.last_node = prev
        for stmt in node.orelse:
            self.visit(stmt)
        else_id = self.last_node
        self.last_node = cond_id 

    def visit_Expr(self, node):
        self._add_node(f"Expr: {ast.dump(node)}")

    def visit_Return(self, node):
        self._add_node(f"Return: {ast.dump(node)}")

    def visit_Assign(self, node):
        self._add_node(f"Assign: {ast.dump(node)}")



def ast_to_cfg(day, year, author, source='github'):
    """Convert AST to CFG given author,  year and code"""
    if source == 'github':
        with open(f'results/ast/github/{year}.json', 'r') as f:
            data = json.load(f)
        
        tree_str = data[str(day)].get(author)
        
        if tree_str:
            # Convert the string back into an AST object using ast.literal_eval
            try:
                # Safe eval: first eval the string into a valid AST node
                tree = ast.parse(tree_str)  # Parse string into an AST object
                
                if tree:
                    cfg_builder = CFGBuilder()
                    for node in tree.body:
                        if isinstance(node, ast.FunctionDef):
                            cfg_builder.visit(node)

                    data = json_graph.node_link_data(cfg_builder.graph)
                    return json.dumps(data, indent=2)
            except Exception as e:
                print(f"Error converting string to AST: {e}")


def code_to_ngrams(code, n=3):
    """Turn python code into ngram representation"""
    try:
        tokens = [tok.string for tok in tokenize.tokenize(BytesIO(code.encode('utf-8')).readline)
                  if tok.type in (tokenize.NAME, tokenize.OP, tokenize.NUMBER, tokenize.STRING)]
        return list(ngrams(tokens, n))
    except tokenize.TokenError:
        return []
    except IndentationError:
        return []
    

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
    df = pd.read_pickle('labelled_data.pkl')
    github_df = df[df['DataSource'] == 'github']
    reddit_df = df[df['DataSource'] == 'reddit']

    for year in range(2015,2024):
        
        
        # AST
        if args.ast:
            
            # Convert every code snippet to AST and add to dataframe

            # Go over all rows in the dataframe
            for index, row in df.iterrows():
                code = row['Data']


                df.at[index, 'ast'] = code_to_ast(code) if row['Data'] and type(row['Data']) == str else None

            # Save the dataframe as json and pickle files
            os.makedirs('results', exist_ok=True)
            df.to_json(f'results/ast.json', orient='records', lines=True)
            df.to_pickle(f'results/ast.pkl')


        # Ngrams
        if args.ngrams:
            
            for index, row in df.iterrows():
                code = row['Data']
                df.at[index, 'ngrams'] = code_to_ngrams(code) if row['Data'] and type(row['Data']) == str else None

            # Save the dataframe as json and pickle files
            os.makedirs('results', exist_ok=True)
            df.to_json(f'results/ngrams.json', orient='records', lines=True)
            df.to_pickle(f'results/ngrams.pkl')


        # Embeddings
        if args.embed:
            
            # Use contrastive learning to get the embeddings
            result = contrastive_learning(df)
            pass

        if args.tfidf:
            pass

if __name__ == "__main__":
    main()

