import ast
import joblib
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import hamming_loss, f1_score, precision_score, recall_score

from sklearn.feature_extraction import DictVectorizer

from collections import Counter, defaultdict

# Core Data Types
@dataclass
class CodeSolution:
    """Represents a code solution with its metadata and labels."""
    code: str
    labels: List[str] 


@dataclass
class PathContext:
    """Represents a path context between two AST nodes."""
    start_token: str
    path: str
    end_token: str

    def __str__(self):
        return f"{self.start_token}|{self.path}|{self.end_token}"

@dataclass
class ASTNode:
    """Custom AST node representation."""
    node_type: str
    children: List['ASTNode']
    token: Optional[str]
    parent: Optional['ASTNode'] = None
    depth: int = 0

    def __hash__(self):
        """Make ASTNode hashable for use in sets."""
        return hash((self.node_type, self.token, id(self)))

    def __eq__(self, other):
        """Define equality for ASTNode."""
        if not isinstance(other, ASTNode):
            return False
        return id(self) == id(other)

# Map CSV column labels to standardized labels (from original code)
LABEL_MAPPING = {
    'Warm': 'warm_up',
    'Gram': 'grammar_parsing',
    'Str': 'string_manipulation',
    'Math': 'mathematical',
    'Sptl': 'spatial',
    'Img': 'image_processing',
    'Cell': 'cellular_automata',
    'Grid': 'grid_traversal',
    'Grph': 'graph_algorithms',
    'Path': 'pathfinding',
    'BFS': 'breadth_first_search',
    'DFS': 'depth_first_search',
    'Dyn': 'dynamic_programming',
    'Memo': 'memoization',
    'Opt': 'optimization',
    'Log': 'logarithmic',
    'Bit': 'bit_manipulation',
    'VM': 'virtual_machine',
    'Rev': 'reverse_engineering',
    'Sim': 'simulation',
    'Inp': 'input_parsing',
    'Scal': 'scaling'
}

# Technique labels based on your original CSV structure
SOLUTION_LABELS = list(LABEL_MAPPING.values())

class ASTProcessor:
    """AST processor with path extraction and node analysis."""

    @staticmethod
    def generate_python_ast(code: str) -> ast.AST:
        """Generate AST from Python code using ast module."""
        try:
            return ast.parse(code)
        except SyntaxError as e:
            try:
                clean_code = code.encode('ascii', 'ignore').decode('ascii')
                return ast.parse(clean_code)
            except:
                raise SyntaxError(f"Failed to parse code: {e}")

    @staticmethod
    def extract_token_from_node(node: ast.AST) -> Optional[str]:
        """Extract meaningful token from AST node."""
        if hasattr(node, 'id'):
            return node.id
        elif hasattr(node, 's'):
            return str(node.s)[:50]  
        elif hasattr(node, 'n'):
            return str(node.n)
        elif hasattr(node, 'value') and hasattr(node.value, '__str__'):
            return str(node.value)[:50]
        elif hasattr(node, 'arg'):
            return node.arg
        elif hasattr(node, 'name'):
            return node.name
        elif hasattr(node, 'attr'):
            return node.attr
        elif hasattr(node, 'op'):
            return type(node.op).__name__
        return None

    @staticmethod
    def ast_to_custom_node(node: ast.AST, parent: Optional['ASTNode'] = None, depth: int = 0) -> ASTNode:
        """Convert Python ast.AST to custom ASTNode format"""
        token = ASTProcessor.extract_token_from_node(node)

        custom_node = ASTNode(
            node_type=type(node).__name__,
            children=[],
            token=token,
            parent=parent,
            depth=depth
        )

        for child in ast.iter_child_nodes(node):
            child_node = ASTProcessor.ast_to_custom_node(child, custom_node, depth + 1)
            custom_node.children.append(child_node)

        return custom_node

    @staticmethod
    def extract_enhanced_paths(ast_root: ASTNode, max_length: int = 6, max_width: int = 3) -> List[PathContext]:
        """Extract path-contexts between AST nodes. All nodes are considered for path endpoints"""

        def collect_all_nodes(node: ASTNode) -> List[ASTNode]:
            """Collect all nodes from the AST"""
            all_nodes_list = [node]  # Add current node
            for child_node in node.children:
                all_nodes_list.extend(collect_all_nodes(child_node)) # Recursively add nodes from children
            return all_nodes_list

        def find_path_between_nodes(start: ASTNode, end: ASTNode) -> Optional[str]:
            """Find simplified path between two nodes."""
            if start == end:
                return None

            # Find paths to root
            start_path_to_root = []
            current = start
            while current and len(start_path_to_root) < max_length * 2:
                start_path_to_root.append(current)
                current = current.parent

            end_path_to_root = []
            current = end
            while current and len(end_path_to_root) < max_length * 2:
                end_path_to_root.append(current)
                current = current.parent

            # Find LCA
            start_ids = {id(n): i for i, n in enumerate(start_path_to_root)}
            lca_idx_in_start = -1
            lca_idx_in_end = -1

            for i, node_in_end_path in enumerate(end_path_to_root):
                if id(node_in_end_path) in start_ids:
                    lca_idx_in_start = start_ids[id(node_in_end_path)]
                    lca_idx_in_end = i
                    break
            
            if lca_idx_in_start == -1: # No LCA found
                return None

            # Build simplified path
            path_parts = []

            # Up from start to LCA
            # Path from start to node just before LCA
            for i in range(lca_idx_in_start):
                path_parts.append(f"{start_path_to_root[i].node_type}↑")

            # Down from LCA to end
            # Path from node just after LCA to end
            for i in range(lca_idx_in_end - 1, -1, -1):
                path_parts.append(f"↓{end_path_to_root[i].node_type}")

            if len(path_parts) > max_length:
                return None

            return "→".join(path_parts) if path_parts else None


        # Extract path contexts
        all_ast_nodes = collect_all_nodes(ast_root)
        path_contexts = []

        # Create path contexts between nearby nodes
        for i, start_node in enumerate(all_ast_nodes):
            end_idx = min(i + max_width + 1, len(all_ast_nodes))
            for end_node in all_ast_nodes[i+1:end_idx]: # Make sure end_node is after start_node
                path = find_path_between_nodes(start_node, end_node)
                if path:
                    start_token_str = start_node.token or start_node.node_type
                    end_token_str = end_node.token or end_node.node_type

                    if start_token_str and end_token_str:
                        path_context = PathContext(
                            start_token=start_token_str,
                            path=path,
                            end_token=end_token_str
                        )
                        path_contexts.append(path_context)
        return path_contexts

    @staticmethod
    def extract_structural_features(ast_root: ASTNode) -> Dict[str, int]:
        """Extract structural features from AST"""
        features = defaultdict(int)

        def traverse(node: ASTNode):
            # Count node types
            features[f"node_type_{node.node_type}"] += 1

            # Count depth levels
            features[f"depth_level_{min(node.depth, 10)}"] += 1

            # Count children
            features[f"children_count_{min(len(node.children), 5)}"] += 1

            for child in node.children:
                traverse(child)

        traverse(ast_root)
        return dict(features)

class AoCDataLoader:
    """Data loader for json datasets"""

    def __init__(self, datasets_dir: str = "../datasets"):
        self.datasets_dir = Path(datasets_dir)

    def load_solutions_from_json(self, split: str = "train") -> List[CodeSolution]:
        """Load solutions from json file."""
        json_file = self.datasets_dir / f"{split}.json"

        if not json_file.exists():
            print(f"Error: {json_file} not found!")
            return []

        solutions = []

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f"Loaded {len(data)} entries from {json_file}")

            for idx, entry in enumerate(data):
                try:
                    # Extract code
                    code = entry.get('Data', '')
                    if not code or not code.strip():
                        continue

                    # Extract labels
                    labels = entry.get('Labels', [])
                    if isinstance(labels, str):
                        # Parse string representation of list
                        try:
                            labels = json.loads(labels)
                        except:
                            labels = [label.strip() for label in labels.split(',') if label.strip()]

                    # Standardize labels with mapping
                    standardized_labels = []
                    for label in labels:
                        label_str = str(label).strip()
                        if label_str in LABEL_MAPPING:
                            standardized_labels.append(LABEL_MAPPING[label_str])
                        elif label_str in SOLUTION_LABELS:
                            standardized_labels.append(label_str)
                        else:
                            std_label = label_str.lower().replace(' ', '_').replace('-', '_')
                            if std_label in SOLUTION_LABELS:
                                standardized_labels.append(std_label)
                            elif label_str:
                                standardized_labels.append(std_label)


                    solution = CodeSolution(
                        code=code,
                        labels=standardized_labels,
                    )
                    solutions.append(solution)

                except Exception as e:
                    print(f"Error processing entry {idx}: {e}")
                    continue

            print(f"Successfully loaded {len(solutions)} valid solutions from {split} set")
            return solutions

        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            return []

    def load_all_splits(self) -> Dict[str, List[CodeSolution]]:
        """Load all available dataset splits"""
        splits = {}
        for split_name in ['train', 'val', 'test']:
            solutions = self.load_solutions_from_json(split_name)
            if solutions:
                splits[split_name] = solutions
        return splits

    def analyze_dataset(self, solutions: List[CodeSolution]) -> Dict[str, Any]:
        """Analyze dataset statistics and identify all unique labels"""
        if not solutions:
            return {}

        # Label statistics
        all_labels = []
        unique_labels_found = set()
        for sol in solutions:
            all_labels.extend(sol.labels)
            unique_labels_found.update(sol.labels)

        print(f"\nUnique labels found in dataset: {sorted(unique_labels_found)}")
        print(f"Expected labels from mapping: {sorted(SOLUTION_LABELS)}")

        # Check for unknown labels
        unknown_labels = unique_labels_found - set(SOLUTION_LABELS)
        if unknown_labels:
            print(f"Unknown labels found: {sorted(unknown_labels)}")

        label_counts = Counter(all_labels)

        # Code statistics
        code_lengths = [len(sol.code) for sol in solutions]

        return {
            'total_solutions': len(solutions),
            'total_labels': len(all_labels),
            'unique_labels': len(label_counts),
            'unique_labels_found': sorted(unique_labels_found),
            'unknown_labels': sorted(unknown_labels),
            'label_distribution': dict(label_counts.most_common()),
            'code_length_stats': {
                'mean': np.mean(code_lengths),
                'median': np.median(code_lengths),
                'std': np.std(code_lengths),
                'min': min(code_lengths),
                'max': max(code_lengths)
            },
            'labels_per_solution': {
                'mean': np.mean([len(sol.labels) for sol in solutions]),
                'median': np.median([len(sol.labels) for sol in solutions])
            }
        }

class FeatureExtractor:
    """Enhanced feature extractor combining multiple AST-based features."""

    def __init__(self, max_features: int = 15000, min_df: int = 2, use_structural: bool = True):
        self.max_features = max_features
        self.min_df = min_df
        self.use_structural = use_structural

        self.pathcontext_vectorizer = TfidfVectorizer(
            max_features=max_features//2, # Adjusted relative to total max_features
            min_df=min_df,
            token_pattern=r'[^|→↑↓]+',
            lowercase=False
        )

        self.node_type_vectorizer = TfidfVectorizer(
            max_features=max_features//2, # Adjusted relative to total max_features
            min_df=min_df,
            token_pattern=r'\b\w+\b',
            lowercase=False
        )

        self.feature_selector = None
        self.fitted = False

    def _extract_features_from_solution(self, solution: CodeSolution) -> Tuple[str, str, Dict[str, int]]:
        """Extract all feature types from a single solution."""
        try:
            # Parse AST
            python_ast = ASTProcessor.generate_python_ast(solution.code)
            ast_root = ASTProcessor.ast_to_custom_node(python_ast)

            # Extract path contexts
            path_contexts = ASTProcessor.extract_enhanced_paths(ast_root)

            # Extract different feature strings
            full_contexts = []
            node_types = []

            for pc in path_contexts:
                if pc.start_token and pc.path and pc.end_token:
                    full_contexts.append(str(pc))

            # Collect node types
            def collect_node_types_recursive(node, types_list):
                types_list.append(node.node_type)
                for child in node.children:
                    collect_node_types_recursive(child, types_list)

            collect_node_types_recursive(ast_root, node_types)

            # Structural features
            structural_features = ASTProcessor.extract_structural_features(ast_root) if self.use_structural else {}

            return (
                " ".join(full_contexts) if full_contexts else "empty_contexts",
                " ".join(node_types) if node_types else "empty_nodetypes",
                structural_features
            )

        except Exception as e:
            print(f"Error processing solution: {e}") # Removed solution object from log for brevity
            return "empty_contexts", "empty_nodetypes", {}

    def fit_transform(self, solutions: List[CodeSolution]) -> sp.csr_matrix:
        """Fit and transform solutions to feature matrix"""
        print(f"Extracting features from {len(solutions)} solutions...")

        pathcontext_features = []
        node_type_features = []
        structural_features_list = []

        for solution in solutions:
            contexts, nodes, structural = self._extract_features_from_solution(solution)
            pathcontext_features.append(contexts)
            node_type_features.append(nodes)
            structural_features_list.append(structural)

        # Transform text features into matrices
        pathcontext_matrix = self.pathcontext_vectorizer.fit_transform(pathcontext_features)
        node_type_matrix = self.node_type_vectorizer.fit_transform(node_type_features)

        matrices = [pathcontext_matrix, node_type_matrix]

        # Add structural features
        if self.use_structural and structural_features_list:
            # Filter out empty dicts to prevent DictVectorizer issues if all are empty
            valid_structural_features = [s for s in structural_features_list if s]
            if valid_structural_features:
                dict_vectorizer = DictVectorizer(sparse=True)
                # Fit only on non-empty dicts, then transform all (empty ones become all-zero rows)
                if any(structural_features_list): # Ensure there's something to fit
                    structural_matrix = dict_vectorizer.fit_transform(structural_features_list)
                    matrices.append(structural_matrix)
                    self.dict_vectorizer = dict_vectorizer
                else: # All structural features were empty
                    # Create a zero matrix of appropriate shape if needed, or DictVectorizer might handle this.
                    # For simplicity, we just note it and DictVectorizer might not be set.
                    print("Warning: All structural features are empty. Structural matrix will not be added.")
            else:
                print("Warning: No valid structural features found.")


        # Combine all features
        if not matrices: # Should not happen with pathcontext and node_type always present
             print("Warning: No feature matrices to combine.")
             # Return an empty sparse matrix with correct number of rows
             return sp.csr_matrix((len(solutions), 0))


        combined_features = sp.hstack(matrices, format='csr')

        # Ensure max_features for TF-IDF is respected *after* hstack if desired,
        # or rely on individual vectorizer max_features.
        # Current logic: individual vectorizers control their features, then VarianceThreshold.

        if combined_features.shape[1] > self.max_features : # Check if total combined features exceed overall max_features
            print(f"Applying VarianceThreshold for feature selection: {combined_features.shape[1]} features initially.")
            # A very small threshold to remove zero-variance (or near zero-variance) features
            self.feature_selector = VarianceThreshold(threshold=(0.0001 * (1 - 0.0001))) 
            try:
                selected_features = self.feature_selector.fit_transform(combined_features)
                print(f"Features after VarianceThreshold: {selected_features.shape[1]}")
                if selected_features.shape[1] == 0 and combined_features.shape[1] > 0:
                    print("Warning: VarianceThreshold removed all features. Using original combined features.")
                    selected_features = combined_features
                    self.feature_selector = None # Disable selector if it removes everything
            except Exception as e:
                print(f"Error during VarianceThreshold: {e}. Using original combined features.")
                selected_features = combined_features
                self.feature_selector = None
        else:
            selected_features = combined_features
            self.feature_selector = None # Explicitly set to None if not used

        self.fitted = True
        print(f"Final feature matrix shape: {selected_features.shape}")
        return selected_features

    def transform(self, solutions: List[CodeSolution]) -> sp.csr_matrix:
        """Transform solutions using fitted extractors"""
        if not self.fitted:
            raise ValueError("Feature extractor must be fitted first")

        pathcontext_features = []
        node_type_features = []
        structural_features_list = []

        for solution in solutions:
            contexts, nodes, structural = self._extract_features_from_solution(solution)
            pathcontext_features.append(contexts)
            node_type_features.append(nodes)
            structural_features_list.append(structural)

        # Transform features
        pathcontext_matrix = self.pathcontext_vectorizer.transform(pathcontext_features)
        node_type_matrix = self.node_type_vectorizer.transform(node_type_features)

        matrices = [pathcontext_matrix, node_type_matrix]

        if self.use_structural and hasattr(self, 'dict_vectorizer') and self.dict_vectorizer:
            structural_matrix = self.dict_vectorizer.transform(structural_features_list)
            matrices.append(structural_matrix)
        
        if not matrices: # Should not happen
             print("Warning: No feature matrices to combine in transform.")
             return sp.csr_matrix((len(solutions), 0))

        combined_features = sp.hstack(matrices, format='csr')

        if self.feature_selector:
            try:
                selected_features = self.feature_selector.transform(combined_features)
                if selected_features.shape[1] == 0 and combined_features.shape[1] > 0:
                     print("Warning: VarianceThreshold transform removed all features. Using original combined features for this batch.")
                     return combined_features
                return selected_features
            except Exception as e:
                print(f"Error during VarianceThreshold transform: {e}. Using original combined features for this batch.")
                return combined_features
        else:
            return combined_features

class AoCClassifier:
    """Multi-label classifier for label classification"""

    def __init__(self, n_estimators: int = 500, max_depth: int = 25,
                 min_samples_split: int = 4, class_weight: str = 'balanced_subsample',
                 threshold: float = 0.3):
        # These hyperparameters are defaults if GridSearchCV is not used,
        # or can be part of the search grid.
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight # This can be passed to RFC within GridSearchCV
        self.threshold = threshold # Used for prediction, not training RF

        self.feature_extractor = FeatureExtractor()
        self.label_binarizer = MultiLabelBinarizer(classes=SOLUTION_LABELS)
        self.classifiers = {}
        self.fitted = False

    def fit(self, solutions: List[CodeSolution]):
        """Train the classifier using GridSearchCV for hyperparameter tuning."""
        print(f"Training enhanced classifier on {len(solutions)} solutions with GridSearchCV...")

        X = self.feature_extractor.fit_transform(solutions)
        y_labels = [sol.labels for sol in solutions]
        y = self.label_binarizer.fit_transform(y_labels)

        print(f"Feature matrix shape: {X.shape}")
        print(f"Label matrix shape: {y.shape}")
        if X.shape[0] == 0:
            print("Error: No samples to train on after feature extraction.")
            return
        if X.shape[1] == 0:
            print("Error: No features to train on after feature extraction.")
            return

        # Define the parameter grid
        param_grid = {
            'n_estimators': [50, 100, 150],  # Number of trees
            'max_depth': [10, 20, None],  # Max depth of trees
            'min_samples_split': [2, 5, 10],  # Min samples to split a node
            'min_samples_leaf': [1, 2, 4],   # Min samples in a leaf node
            'class_weight': [self.class_weight, 'balanced']
        }
        
        # Base RandomForestClassifier
        base_rf_clf = RandomForestClassifier(random_state=42, bootstrap=True, max_features='sqrt', n_jobs=1)

        for i, technique in enumerate(self.label_binarizer.classes_):
            positive_samples = np.sum(y[:, i])
            if positive_samples > 1:
                print(f"Training classifier for {technique} ({positive_samples} positive samples) using GridSearchCV...")

                grid_search = GridSearchCV(
                    estimator=base_rf_clf,
                    param_grid=param_grid,
                    scoring='f1', 
                    cv=3,        
                    n_jobs=-1,
                    verbose=1 
                )
                
                try:
                    grid_search.fit(X, y[:, i])
                    self.classifiers[technique] = grid_search.best_estimator_
                    print(f"  Best F1 score for {technique}: {grid_search.best_score_:.4f}")
                    print(f"  Best params for {technique}: {grid_search.best_params_}")
                except Exception as e:
                    print(f"Error during GridSearchCV for {technique}: {e}")
                    print(f"  Falling back to default parameters for {technique}.")
                    # Fallback to default classifier if GridSearchCV fails
                    fallback_clf = RandomForestClassifier(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                        class_weight=self.class_weight, 
                        random_state=42,
                        n_jobs=-1, #
                        bootstrap=True,
                        max_features='sqrt'
                    )
                    fallback_clf.fit(X, y[:,i])
                    self.classifiers[technique] = fallback_clf

            else:
                print(f"Skipping {technique} (insufficient positive samples: {positive_samples}) for GridSearchCV.")
        
        self.fitted = True
        print(f"Training completed! Trained {len(self.classifiers)} classifiers.")


    def predict_proba(self, solutions: List[CodeSolution]) -> np.ndarray:
        """Predict probabilities for each technique."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")

        X = self.feature_extractor.transform(solutions)
        if X.shape[0] == 0:
            return np.zeros((0, len(self.label_binarizer.classes_)))
        if X.shape[1] == 0:
            print("Warning: Transformed feature matrix has 0 features. Predicting all zeros.")
            return np.zeros((len(solutions), len(self.label_binarizer.classes_)))


        probabilities = np.zeros((len(solutions), len(self.label_binarizer.classes_)))

        for i, technique in enumerate(self.label_binarizer.classes_):
            if technique in self.classifiers:
                clf = self.classifiers[technique]
                probs = clf.predict_proba(X)
                if probs.shape[1] == 2:
                    probabilities[:, i] = probs[:, 1]
                elif probs.shape[1] == 1: 
                    if clf.classes_[0] == 1: 
                        probabilities[:, i] = probs[:, 0]
                    else: 
                        probabilities[:, i] = 0.0
                else: 
                    probabilities[:, i] = 0.0 
            else:
                probabilities[:, i] = 0.0
        return probabilities

    def predict(self, solutions: List[CodeSolution], threshold: Optional[float] = None) -> List[List[str]]:
        """Predict technique labels for solutions"""
        if threshold is None:
            threshold = self.threshold

        if not solutions:
            return []
            
        probabilities = self.predict_proba(solutions)
        if probabilities.shape[0] == 0: 
             return [[] for _ in solutions]

        predictions = []

        for prob_row in probabilities:
            predicted_labels = []
            for i, prob in enumerate(prob_row):
                if prob >= threshold:
                    predicted_labels.append(self.label_binarizer.classes_[i])

            if not predicted_labels and np.any(prob_row > 0.0) and np.max(prob_row) > 0.1 : 
                best_idx = np.argmax(prob_row)
                if prob_row[best_idx] > 0.1: 
                    predicted_labels.append(self.label_binarizer.classes_[best_idx])

            predictions.append(predicted_labels)
        return predictions

class ClassificationEvaluator:
    """Enhanced evaluator with detailed metrics"""

    def __init__(self, labels: List[str] = None):
        self.labels = labels if labels is not None else []


    def evaluate(self, y_true: List[List[str]], y_pred: List[List[str]], 
         all_possible_labels: Optional[List[str]] = None,
         data_sources: Optional[List[str]] = None) -> Dict[str, Any]: # Used to work, but isn't used anymore
        
        def safe_binarize(y_true, y_pred, current_labels):
            mlb = MultiLabelBinarizer(classes=current_labels)
            try:
                y_true_bin = mlb.fit_transform(y_true)
            except ValueError as e:
                print(f"Error binarizing y_true: {e}.")
                y_true_bin = np.zeros((len(y_true), len(current_labels)), dtype=int)

            try:
                y_pred_bin = mlb.transform(y_pred)
            except ValueError:
                print("Warning: y_pred contained new labels. Re-fitting with combined label set.")
                combined_labels = set(mlb.classes_)
                for lst in y_pred:
                    combined_labels.update(lst)
                mlb_new = MultiLabelBinarizer(classes=sorted(list(combined_labels)))
                y_true_bin = mlb_new.fit_transform(y_true) 
                y_pred_bin = mlb_new.transform(y_pred)
                print(f"  Original labels: {current_labels}, New labels for binarization: {mlb_new.classes_}")
                current_labels = mlb_new.classes_ # Update to the set used
            return mlb, y_true_bin, y_pred_bin, current_labels


        def compute_metrics(y_true_samples, y_pred_samples, labels_for_metric_calc, original_labels_for_reporting, prefix=""):
            """Compute all metrics for a given set of samples."""
            if not y_true_samples:
                return {}
                
            # Binarize using the potentially expanded label set from safe_binarize
            _, y_true_bin, y_pred_bin, actual_binarized_labels = safe_binarize(y_true_samples, y_pred_samples, labels_for_metric_calc)
            
            # Compute per label F1 scores based on actual_binarized_labels
            per_label_f1_all = f1_score(y_true_bin, y_pred_bin, average=None, zero_division=0)
            
            # Map F1 scores back to original_labels_for_reporting, if they differ
            per_technique_f1_reported = {}
            for label in original_labels_for_reporting:
                try:
                    idx_in_actual = actual_binarized_labels.index(label)
                    per_technique_f1_reported[label] = float(per_label_f1_all[idx_in_actual])
                except ValueError: 
                    per_technique_f1_reported[label] = 0.0


            metrics = {
                'hamming_loss': hamming_loss(y_true_bin, y_pred_bin),
                'f1_micro': f1_score(y_true_bin, y_pred_bin, average='micro', zero_division=0),
                'f1_macro': f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0), 
                'f1_weighted': f1_score(y_true_bin, y_pred_bin, average='weighted', zero_division=0), 
                'precision_micro': precision_score(y_true_bin, y_pred_bin, average='micro', zero_division=0),
                'precision_macro': precision_score(y_true_bin, y_pred_bin, average='macro', zero_division=0),
                'recall_micro': recall_score(y_true_bin, y_pred_bin, average='micro', zero_division=0),
                'recall_macro': recall_score(y_true_bin, y_pred_bin, average='macro', zero_division=0),
                'exact_match_ratio': np.mean([set(t) == set(p) for t, p in zip(y_true_samples, y_pred_samples)]),
                'one_correct_accuracy': np.mean([1 if set(t) & set(p) else 0 for t, p in zip(y_true_samples, y_pred_samples)]),
                'per_technique_f1': per_technique_f1_reported,
                'label_frequency': { 
                    'true_counts': Counter(l for sublist in y_true_samples for l in sublist if l in original_labels_for_reporting),
                    'pred_counts': Counter(l for sublist in y_pred_samples for l in sublist if l in original_labels_for_reporting)
                },
                'sample_count': len(y_true_samples),
                'evaluated_labels_for_metrics': actual_binarized_labels,
                'reported_labels_for_per_technique': original_labels_for_reporting 
                }
            
            return metrics

        if all_possible_labels:
            current_labels_for_reporting = all_possible_labels
        elif self.labels:
            current_labels_for_reporting = self.labels
        else:
            label_set = set()
            for l_list in y_true + y_pred: 
                for l_item in l_list:
                    label_set.add(l_item)
            current_labels_for_reporting = sorted(list(label_set))
            if not current_labels_for_reporting:
                return {'error': 'No labels to evaluate (all_possible_labels, self.labels, and data y_true/y_pred are empty or contain no labels).'}


        # Overall metrics
        overall_metrics = compute_metrics(y_true, y_pred, current_labels_for_reporting, current_labels_for_reporting)
        
        results = {
            'overall': overall_metrics
        }

        # Per data source evaluation
        if data_sources:
            # Group samples by data source
            grouped_true = defaultdict(list)
            grouped_pred = defaultdict(list)

            for source, y_t, y_p in zip(data_sources, y_true, y_pred):
                grouped_true[source].append(y_t)
                grouped_pred[source].append(y_p)

            # Compute metrics for each data source
            per_datasource_metrics = {}
            for source in grouped_true:
                y_t = grouped_true[source]
                y_p = grouped_pred[source]
                source_metrics = compute_metrics(y_t, y_p, current_labels_for_reporting, current_labels_for_reporting)
                per_datasource_metrics[source] = source_metrics

            results['per_datasource'] = per_datasource_metrics
            
            # Add summary statistics across datasources
            if len(per_datasource_metrics) > 1:
                datasource_summary = {}
                metric_names = ['f1_micro', 'f1_macro', 'f1_weighted', 'precision_micro', 
                            'precision_macro', 'recall_micro', 'recall_macro', 
                            'hamming_loss', 'exact_match_ratio', 'one_correct_accuracy']
                
                for metric in metric_names:
                    values = [per_datasource_metrics[ds][metric] for ds in per_datasource_metrics if metric in per_datasource_metrics[ds]]
                    if values: # Ensure list is not empty
                        datasource_summary[f'{metric}_mean'] = np.mean(values)
                        datasource_summary[f'{metric}_std'] = np.std(values)
                        datasource_summary[f'{metric}_min'] = np.min(values)
                        datasource_summary[f'{metric}_max'] = np.max(values)
                
                results['datasource_summary'] = datasource_summary

        return results

    def print_evaluation_report(self, results: Dict[str, Any]):
        """Print formatted evaluation report, including per-datasource metrics if available."""
        # Handle error
        if 'error' in results:
            print(f"Evaluation Error: {results['error']}")
            return

        # Extract overall metrics
        if 'overall' in results:
            overall = results['overall']
        else:
            overall = results # Legacy: if results is just the overall dict

        print("\n" + "=" * 70)
        print("      ADVENT OF CODE TECHNIQUE CLASSIFICATION RESULTS")
        print("=" * 70)

        # Overall metrics section
        print(f"Overall Metrics (Total samples: {overall.get('sample_count', 'N/A')}):")
        print(f"  Hamming Loss:      {overall.get('hamming_loss', 0.0):.4f}")
        print(f"  F1 Micro:          {overall.get('f1_micro', 0.0):.4f}")
        print(f"  F1 Macro:          {overall.get('f1_macro', 0.0):.4f} (avg over {len(overall.get('evaluated_labels_for_metrics',[]))} labels)")
        print(f"  F1 Weighted:       {overall.get('f1_weighted', 0.0):.4f} (avg over {len(overall.get('evaluated_labels_for_metrics',[]))} labels)")
        print(f"  Precision Micro:   {overall.get('precision_micro', 0.0):.4f}")
        print(f"  Precision Macro:   {overall.get('precision_macro', 0.0):.4f}")
        print(f"  Recall Micro:      {overall.get('recall_micro', 0.0):.4f}")
        print(f"  Recall Macro:      {overall.get('recall_macro', 0.0):.4f}")
        print(f"  Exact Match Ratio: {overall.get('exact_match_ratio', 0.0):.4f}")
        print(f"  One-Correct Acc.:  {overall.get('one_correct_accuracy', 0.0):.4f}")

        # Per-technique F1 scores
        reported_labels = overall.get('reported_labels_for_per_technique', [])
        per_technique_f1_data = overall.get('per_technique_f1', {})
        
        print(f"\nPer-Technique F1 Scores (reported for {len(reported_labels)} labels):")
        print("-" * 60)

        # Sort techniques for display
        # Ensure label frequency data exists and is correctly keyed
        true_counts_overall = overall.get('label_frequency', {}).get('true_counts', Counter())
        pred_counts_overall = overall.get('label_frequency', {}).get('pred_counts', Counter())

        sorted_techniques = sorted(
            per_technique_f1_data.items(),
            key=lambda x: (x[1], true_counts_overall.get(x[0], 0)), # Sort by F1, then true count
            reverse=True
        )

        for technique, f1 in sorted_techniques:
            true_count = true_counts_overall.get(technique, 0)
            pred_count = pred_counts_overall.get(technique, 0)
            print(f"  {technique:<30}: {f1:.4f} (true: {true_count:>3}, pred: {pred_count:>3})")


        # Per-datasource metrics
        if 'per_datasource' in results:
            print("\n" + "=" * 70)
            print("                    PER-DATASOURCE METRICS")
            print("=" * 70)
            
            per_ds = results['per_datasource']
            
            # Sort datasources by F1 micro score
            sorted_datasources = sorted(
                per_ds.items(), 
                key=lambda x: x[1].get('f1_micro', 0.0), 
                reverse=True
            )
            
            for source, metrics in sorted_datasources:
                print(f"\nDatasource: {source} ({metrics.get('sample_count','N/A')} samples)")
                print("-" * 50)
                print(f"  Hamming Loss:      {metrics.get('hamming_loss', 0.0):.4f}")
                print(f"  F1 Micro:          {metrics.get('f1_micro', 0.0):.4f}")
                # Note which labels were used for macro averages if it differs
                macro_labels_count = len(metrics.get('evaluated_labels_for_metrics', reported_labels))
                print(f"  F1 Macro:          {metrics.get('f1_macro', 0.0):.4f} (avg over {macro_labels_count} labels)")
                print(f"  F1 Weighted:       {metrics.get('f1_weighted', 0.0):.4f} (avg over {macro_labels_count} labels)")
                print(f"  Precision Micro:   {metrics.get('precision_micro', 0.0):.4f}")
                print(f"  Precision Macro:   {metrics.get('precision_macro', 0.0):.4f}")
                print(f"  Recall Micro:      {metrics.get('recall_micro', 0.0):.4f}")
                print(f"  Recall Macro:      {metrics.get('recall_macro', 0.0):.4f}")
                print(f"  Exact Match Ratio: {metrics.get('exact_match_ratio', 0.0):.4f}")
                print(f"  One-Correct Acc.:  {metrics.get('one_correct_accuracy', 0.0):.4f}")

            # Summary statistics across datasources
            if 'datasource_summary' in results and len(per_ds) > 1:
                print("\n" + "=" * 70)
                print("              DATASOURCE SUMMARY STATISTICS")
                print("=" * 70)
                summary = results['datasource_summary']
                
                print("Performance Variation Across Datasources:")
                print("-" * 50)
                
                key_metrics_display = [
                    ('F1 Micro', 'f1_micro'),
                    ('F1 Macro', 'f1_macro'),
                    ('F1 Weighted', 'f1_weighted'),
                    ('Precision Micro', 'precision_micro'),
                    ('Recall Micro', 'recall_micro'),
                    ('Exact Match Ratio', 'exact_match_ratio')
                ]
                
                for display_name, metric_key in key_metrics_display:
                    mean_val = summary.get(f'{metric_key}_mean', 0.0)
                    std_val = summary.get(f'{metric_key}_std', 0.0)
                    min_val = summary.get(f'{metric_key}_min', 0.0)
                    max_val = summary.get(f'{metric_key}_max', 0.0)
                    print(f"  {display_name:<18}: {mean_val:.4f} ± {std_val:.4f} (range: {min_val:.4f} - {max_val:.4f})")

        # Performance insights (Best/Worst datasource)
        if 'per_datasource' in results and len(results['per_datasource']) > 1:
            print("\n" + "=" * 70)
            print("                   PERFORMANCE INSIGHTS")
            print("=" * 70)
            
            # Find best and worst performing data sources based on F1 Micro
            valid_ds_metrics = {ds_name: ds_metric for ds_name, ds_metric in results['per_datasource'].items() if 'f1_micro' in ds_metric}
            if valid_ds_metrics:
                best_ds = max(valid_ds_metrics.items(), key=lambda x: x[1]['f1_micro'])
                worst_ds = min(valid_ds_metrics.items(), key=lambda x: x[1]['f1_micro'])
            
                print(f"Best performing datasource:  {best_ds[0]} (F1 Micro: {best_ds[1]['f1_micro']:.4f})")
                print(f"Worst performing datasource: {worst_ds[0]} (F1 Micro: {worst_ds[1]['f1_micro']:.4f})")
            else:
                print("Not enough data to determine best/worst performing datasources.")
            

        print("\n" + "=" * 70)


def train_enhanced_classifier(datasets_dir: str = "../datasets") -> Tuple[Optional[AoCClassifier], Dict[str, Any]]:
    """Train the enhanced classifier using JSON datasets."""
    global SOLUTION_LABELS

    # Load data
    loader = AoCDataLoader(datasets_dir)
    splits = loader.load_all_splits()

    if 'train' not in splits or not splits['train']:
        print("Error: No training data found or training data is empty!")
        return None, {}

    train_solutions = splits['train']

    # Analyze the dataset
    print("Analyzing training dataset...")
    analysis = loader.analyze_dataset(train_solutions)
    print(f"Dataset analysis: {analysis.get('total_solutions', 0)} solutions, "
          f"{analysis.get('unique_labels', 0)} unique labels found.")

    if 'unique_labels_found' in analysis and analysis['unique_labels_found']:
        all_found_labels_in_train = set(analysis['unique_labels_found'])
        # Combine with existing SOLUTION_LABELS and sort
        updated_labels = sorted(list(set(SOLUTION_LABELS) | all_found_labels_in_train))
        if updated_labels != sorted(SOLUTION_LABELS):
            print("Updating SOLUTION_LABELS with labels found in training data...")
            SOLUTION_LABELS = updated_labels
            print(f"Updated SOLUTION_LABELS: {SOLUTION_LABELS}")
    
    # Pass the definitive list of labels to the classifier and evaluator
    current_model_labels = list(SOLUTION_LABELS)

    # Use validation set if available, otherwise split training data
    if 'val' in splits and splits['val']:
        val_solutions = splits['val']
    else:
        if len(train_solutions) < 2: # Need at least 2 samples for train/test split
             print("Warning: Not enough training samples to create a validation split. Using training data for validation.")
             val_solutions = list(train_solutions) # Use a copy
        else:
            train_solutions, val_solutions = train_test_split(train_solutions, test_size=0.2, random_state=42)

    print(f"Training set: {len(train_solutions)} solutions")
    print(f"Validation set: {len(val_solutions)} solutions")

    # Train model
    # Hyperparameters passed to AoCClassifier constructor act as defaults if GridSearchCV
    # does not override them or if it fails.
    model = AoCClassifier(
        n_estimators=100, # Default, GridSearchCV will search
        max_depth=20,      # Default, GridSearchCV will search
        min_samples_split=5, # Default, GridSearchCV will search
        class_weight='balanced_subsample', # Can be part of grid search
        threshold=0.3      # For prediction
    )

    model.label_binarizer = MultiLabelBinarizer(classes=current_model_labels)


    model.fit(train_solutions) # This now uses GridSearchCV internally
    
    evaluation_results = {}

    # Evaluate on validation set
    if val_solutions:
        print("\nEvaluating on validation set...")
        val_predictions = model.predict(val_solutions)
        val_true = [sol.labels for sol in val_solutions]

        evaluator = ClassificationEvaluator(labels=current_model_labels)
        results = evaluator.evaluate(val_true, val_predictions, all_possible_labels=current_model_labels)
        evaluator.print_evaluation_report(results)
        evaluation_results['validation_results'] = results
    else:
        print("Skipping validation set evaluation as it's empty.")


    # Test on test set if available
    if 'test' in splits and splits['test']:
        print("\nEvaluating on test set...")
        test_solutions = splits['test']
        test_predictions = model.predict(test_solutions)
        test_true = [sol.labels for sol in test_solutions]


        evaluator_test = ClassificationEvaluator(labels=current_model_labels)
        # Pass data_sources=None if not applicable for the test set, or provide actual sources
        test_results = evaluator_test.evaluate(test_true, test_predictions, all_possible_labels=current_model_labels, data_sources=None)
        print("\nTEST SET RESULTS:")
        evaluator_test.print_evaluation_report(test_results)
        evaluation_results['test_results'] = test_results
    else:
        print("No test set found or test set is empty. Skipping test set evaluation.")


    return model, evaluation_results


if __name__ == "__main__":
    # Train model with JSON datasets
    print("Training Enhanced AST-based AoC Technique Classifier with GridSearchCV...")
    
    # Ensure the datasets directory exists or provide a correct path
    datasets_path = Path("../datasets")
    if not datasets_path.exists():
        print(f"ERROR: Datasets directory '{datasets_path.resolve()}' not found.")
        print("Please create it and place train.json, val.json (optional), test.json (optional) inside.")
        datasets_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {datasets_path.resolve()}")
        print("You'll need to populate it with actual data for the model to train.")


    print(f"Loading data from {datasets_path.resolve()}")
    
    model, results = train_enhanced_classifier(str(datasets_path))

    if model is not None and model.fitted:
        print("\n" + "="*70)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)

        # Save model
        model_filename = f"models/ast_classifier_gridsearch.joblib"
        Path("models").mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_filename)
        print(f"Model saved to {model_filename}")        



    elif model is not None and not model.fitted:
        print("Model was initialized but training did not complete successfully (e.g. no data or features).")
    else:
        print("Training failed. Please check your dataset files and paths.")