import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib
import os
from contrastive_learning import *

LABEL_COLUMNS = [
    'Warm', 'Gram', 'Str', 'Math', 'Sptl', 'Img', 'Cell', 'Grid', 'Grph', 'Path', 'BFS', 'DFS',
    'Dyn', 'Memo', 'Opt', 'Log', 'Bit', 'VM', 'Rev', 'Sim', 'Inp', 'Scal'
]

def preprocess(df, preprocessing_type, label_columns):
    # df_filtered = dataframe[preprocessing_type]
    # print(f"[{preprocessing_type}] Rows remaining after filtering: {len(df_filtered)}")

    X = df[preprocessing_type].astype(str).fillna('')
    Y = df[label_columns].notna().astype(int)
    vectorizer = CountVectorizer()
    X_vect = vectorizer.fit_transform(X)
    return X_vect, Y, vectorizer


def train_mlp_on_ast(dataframe, label_columns, preprocessing_type):
    X, Y, vectorizer = preprocess(dataframe, preprocessing_type, label_columns)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = MultiOutputClassifier(MLPClassifier(hidden_layer_sizes=(512,), max_iter=1000, random_state=42))
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    print("MLP Classification Report:")
    print(classification_report(Y_test, Y_pred, target_names=label_columns, zero_division=0))

    # Calculate accuracy and accuracy as Jaccard subset
    accuracy_test = accuracy_score(Y_test, Y_pred)
    print("Exact match accuracy (on TEST set):", accuracy_test)

    correct_predictions_test = np.all(Y_test == Y_pred, axis=1) # Exact match
    at_least_one_correct_test = np.sum(np.sum((Y_test == 1) & (Y_pred == 1), axis=1) > 0) / len(Y_test)
    print("At least one correct prediction:", at_least_one_correct_test)


    return model

def train_svm(dataframe, label_columns, preprocessing_type):
    X, Y, _ = preprocess(dataframe, preprocessing_type, label_columns)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = MultiOutputClassifier(LinearSVC(max_iter=5000))
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    print("SVM Classification Report:")
    print(classification_report(Y_test, Y_pred, target_names=label_columns, zero_division=0))

    # Calculate accuracy and accuracy as Jaccard subset
    accuracy_test = accuracy_score(Y_test, Y_pred)
    print("Exact match accuracy (on TEST set):", accuracy_test)

    correct_predictions_test = np.all(Y_test == Y_pred, axis=1) # Exact match
    at_least_one_correct_test = np.sum(np.sum((Y_test == 1) & (Y_pred == 1), axis=1) > 0) / len(Y_test)
    print("At least one correct prediction:", at_least_one_correct_test)

    return model

def train_random_forest(dataframe, label_columns, preprocessing_type):
    X, Y, _ = preprocess(dataframe, preprocessing_type, label_columns)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    print("Random Forest Classification Report:")
    print(classification_report(Y_test, Y_pred, target_names=label_columns, zero_division=0))

    # Calculate accuracy and accuracy as Jaccard subset
    accuracy_test = accuracy_score(Y_test, Y_pred)
    print("Exact match accuracy (on TEST set):", accuracy_test)

    correct_predictions_test = np.all(Y_test == Y_pred, axis=1) # Exact match
    at_least_one_correct_test = np.sum(np.sum((Y_test == 1) & (Y_pred == 1), axis=1) > 0) / len(Y_test)
    print("At least one correct prediction:", at_least_one_correct_test)

    return model

def train_knn(dataframe, label_columns, preprocessing_type):
    X, Y, _ = preprocess(dataframe, preprocessing_type, label_columns)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5))
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    print("K-Nearest Neighbors Classification Report:")
    print(classification_report(Y_test, Y_pred, target_names=label_columns, zero_division=0))

    # Calculate accuracy and accuracy as Jaccard subset
    accuracy_test = accuracy_score(Y_test, Y_pred)
    print("Exact match accuracy (on TEST set):", accuracy_test)

    correct_predictions_test = np.all(Y_test == Y_pred, axis=1) # Exact match
    at_least_one_correct_test = np.sum(np.sum((Y_test == 1) & (Y_pred == 1), axis=1) > 0) / len(Y_test)
    print("At least one correct prediction:", at_least_one_correct_test)

    return model


def show_example_predictions(X_test, Y_test, Y_pred, label_columns, n=5):
    print("\n--- Example Predictions ---")
    n = min(n, len(X_test))
    if n == 0:
        print("No test samples to show.")
        return

    for i in range(n):
        true_labels_indices = np.where(Y_test[i] == 1)[0]
        predicted_labels_indices = np.where(Y_pred[i] == 1)[0]

        true_labels = [label_columns[idx] for idx in true_labels_indices]
        predicted_labels = [label_columns[idx] for idx in predicted_labels_indices]

        print(f"Sample {i + 1}")
        print(f"  True:      {true_labels if true_labels else ['(None)']}")
        print(f"  Predicted: {predicted_labels if predicted_labels else ['(None)']}")
        print("------")


def main():    
    # AST
    print("\nAST")
    print("Training MLP on AST")
    df_ast = pd.read_pickle('results/ast.pkl')
    ast_model = train_mlp_on_ast(df_ast, LABEL_COLUMNS, 'ast')
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(ast_model, 'models/ast_mlp_model.joblib')

    print("\nTraining SVM on AST")
    svm_model = train_svm(df_ast, LABEL_COLUMNS, 'ast')
    joblib.dump(svm_model, 'models/ast_svm_model.joblib')

    # Ngrams
    print("\nNgrams")
    print("Training Random Forest on Ngrams")
    df_ngrams = pd.read_pickle('results/ngrams.pkl')
    rf_model = train_random_forest(df_ngrams, LABEL_COLUMNS, 'ngrams')
    joblib.dump(rf_model, 'models/ngrams_rf_model.joblib')

    print("\nTraining KNN on Ngrams")
    knn_model = train_knn(df_ngrams, LABEL_COLUMNS, 'ngrams')
    joblib.dump(knn_model, 'models/ngrams_knn_model.joblib')
    print("\nTraining Logistic Regression on Ngrams")


if __name__ == "__main__":
    main()

