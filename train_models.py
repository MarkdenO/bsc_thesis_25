from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score, hamming_loss
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import re


LABEL_COLUMNS = [
    'Warm', 'Gram', 'Str', 'Math', 'Sptl', 'Img', 'Cell', 'Grid', 'Grph', 'Path', 'BFS', 'DFS',
    'Dyn', 'Memo', 'Opt', 'Log', 'Bit', 'VM', 'Rev', 'Sim', 'Inp', 'Scal'
]


def code_tokenizer(code):
    """
    Tokenizes code, while keeping operators and numbers as separate tokens.
    """
    tokens = re.findall(r'\w+|[+\-*/=<>!&|]+|\d+|[(){}\[\];,.]', code)
    return tokens


def load_datasets(data_type=''):
    """
    Loads train, validation, and test datasets from JSON files in the 'datasets' directory.
    """
    train_path = os.path.join('datasets', f'train{data_type}.json')
    val_path = os.path.join('datasets', f'val{data_type}.json')
    test_path = os.path.join('datasets', f'test{data_type}.json')

    if not all(os.path.exists(path) for path in [train_path, val_path, test_path]):
        raise FileNotFoundError("train.json, val.json, or test.json not found in /datasets directory.")

    train_data = pd.read_json(train_path)
    val_data = pd.read_json(val_path)
    test_data = pd.read_json(test_path)

    return train_data, val_data, test_data

def preprocess(df):
    """
    Preprocesses the DataFrame by tokenizing the 'Data' column and creating a binary label matrix.
    """
    X = df['Data'].astype(str).fillna('')
    Y = pd.DataFrame(0, index=np.arange(len(df)), columns=LABEL_COLUMNS)

    for i, label_list in enumerate(df['Labels']):
        for label in label_list:
            if label in LABEL_COLUMNS:
                Y.at[i, label] = 1

    vectorizer = CountVectorizer(tokenizer=code_tokenizer, lowercase=False, token_pattern=None)
    X_vect = vectorizer.fit_transform(X)
    return X_vect, Y, vectorizer


def preprocess_with_existing_vectorizer(df, vectorizer):
    """
    Preprocesses the DataFrame using an existing CountVectorizer.
    """
    X = df['Data'].astype(str).fillna('')
    Y = pd.DataFrame(0, index=np.arange(len(df)), columns=LABEL_COLUMNS)

    for i, label_list in enumerate(df['Labels']):
        for label in label_list:
            if label in LABEL_COLUMNS:
                Y.at[i, label] = 1

    X_vect = vectorizer.transform(X)
    return X_vect, Y


def train_model_with_cv(train_data, test_data, label_columns, text_column, model_cls, param_grid, model_name="model"):
    """
    Trains a model using GridSearchCV with the provided training and test datasets.
    """
    # Preprocess train and test
    X_train, Y_train, vectorizer = preprocess(train_data)
    X_test, Y_test = preprocess_with_existing_vectorizer(test_data, vectorizer)

    # Setup model and GridSearchCV
    base_model = model_cls()
    clf = OneVsRestClassifier(base_model)

    grid_search = GridSearchCV(
        clf,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=3,
        verbose=2,
        n_jobs=-1
    )

    # Train model
    grid_search.fit(X_train, Y_train)

    # Best model
    best_model = grid_search.best_estimator_
    print("Best parameters found:", grid_search.best_params_)

    # Evaluate on test set
    Y_pred = best_model.predict(X_test)
    print(f"\n{model_name} Classification Report (on TEST set):")
    print(classification_report(Y_test, Y_pred, target_names=label_columns, zero_division=0))
    print("Exact match accuracy:", accuracy_score(Y_test, Y_pred))
    at_least_one_correct_test = np.sum(np.sum((Y_test == 1) & (Y_pred == 1), axis=1) > 0) / len(Y_test)
    print("At least one correct prediction:", at_least_one_correct_test)

    # Evaluate on test set per DataSource
    # data_sources = test_data['DataSource'].unique()
    # for source in data_sources:
    #     X_test_source = test_data[test_data['DataSource'] == source]
    #     X_test_vect, Y_test_source = preprocess_with_existing_vectorizer(X_test_source, vectorizer)
    #     Y_pred_source = best_model.predict(X_test_vect)
    #     print(f"\n{model_name} Classification Report for DataSource '{source}':")
    #     print(classification_report(Y_test_source, Y_pred_source, target_names=label_columns, zero_division=0))
    #     print("Exact match accuracy for DataSource '{}':".format(source), accuracy_score(Y_test_source, Y_pred_source))
    #     print("At least one correct prediction for DataSource '{}':".format(source),
    #           np.sum(np.sum((Y_test_source == 1) & (Y_pred_source == 1), axis=1) > 0) / len(Y_test_source))
    #     print("Hamming loss for DataSource '{}':".format(source), hamming_loss(Y_test_source, Y_pred_source))

    # Evaluate on test set for multi-label and single-label cases
    test_multi_label = test_data[test_data['Labels'].apply(lambda x: len(x) > 1)]
    test_single_label = test_data[test_data['Labels'].apply(lambda x: len(x) == 1)]

    if not test_multi_label.empty:
        X_test_multi, Y_test_multi = preprocess_with_existing_vectorizer(test_multi_label, vectorizer)
        Y_pred_multi = best_model.predict(X_test_multi)
        print(f"\n{model_name} Classification Report for Multi-label Test Set:")
        print(classification_report(Y_test_multi, Y_pred_multi, target_names=label_columns, zero_division=0))
        print("Exact match accuracy for Multi-label Test Set:", accuracy_score(Y_test_multi, Y_pred_multi))
        print("At least one correct prediction for Multi-label Test Set:",
              np.sum(np.sum((Y_test_multi == 1) & (Y_pred_multi == 1), axis=1) > 0) / len(Y_test_multi))
        print("Hamming loss for Multi-label Test Set:", hamming_loss(Y_test_multi, Y_pred_multi))

    if not test_single_label.empty:
        X_test_single, Y_test_single = preprocess_with_existing_vectorizer(test_single_label, vectorizer)
        Y_pred_single = best_model.predict(X_test_single)
        print(f"\n{model_name} Classification Report for Single-label Test Set:")
        print(classification_report(Y_test_single, Y_pred_single, target_names=label_columns, zero_division=0))
        print("Exact match accuracy for Single-label Test Set:", accuracy_score(Y_test_single, Y_pred_single))
        print("At least one correct prediction for Single-label Test Set:",
              np.sum(np.sum((Y_test_single == 1) & (Y_pred_single == 1), axis=1) > 0) / len(Y_test_single))
        print("Hamming loss for Single-label Test Set:", hamming_loss(Y_test_single, Y_pred_single))

    # Hamming loss
    print("Hamming loss:", hamming_loss(Y_test, Y_pred))

    # Save model and vectorizer
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, f'models/{model_name}.joblib')
    joblib.dump(vectorizer, f'models/{model_name}_vectorizer.joblib')

    return best_model

def main():
    train_data, val_data, test_data = load_datasets()
    train_ngrams , val_ngrams, test_ngrams = load_datasets(data_type='_ngrams')

    test_multi_label = test_data[test_data['Labels'].apply(lambda x: len(x) > 1)]
    test_single_label = test_data[test_data['Labels'].apply(lambda x: len(x) == 1)]


    # Parameter grid for Random Forest
    param_grid_rf = {
        'estimator__n_estimators': [100, 200],
        'estimator__max_depth': [None, 10, 20],
        'estimator__min_samples_split': [2, 5],
        'estimator__min_samples_leaf': [1, 2]
    }

    param_grid_lr = {
        'estimator__C': [0.1, 1, 10],
        'estimator__max_iter': [100, 200]
    }

    params_ngrams = {
        'estimator__n_estimators': [100],
        'estimator__max_depth': [None],
        'estimator__min_samples_split': [2],
        'estimator__min_samples_leaf': [1]
    }

    params_bow = {
        'estimator__C': [10],
        'estimator__max_iter': [200]
    }

    # # Train Random Forest with GridSearchCV
    # rf_model = train_model_with_cv(train_data, test_data, LABEL_COLUMNS, 'Data', RandomForestClassifier, param_grid_rf, model_name="rf_model")
    # # Save the model and vectorizer
    # os.makedirs('models', exist_ok=True)
    # joblib.dump(rf_model, 'models/rf_model.joblib')
    # joblib.dump(rf_model, 'models/rf_vectorizer.joblib')

    # Train Bag of Words Classifier
    bow_model = train_model_with_cv(train_data, test_data, LABEL_COLUMNS, 'Data', LogisticRegression, params_bow, model_name="bow_model")
    
    # Save the model and vectorizer
    os.makedirs('models', exist_ok=True)
    joblib.dump(bow_model, 'models/bow_model.joblib')
    joblib.dump(bow_model, 'models/bow_vectorizer.joblib')

    # Train Ngram model
    ngram_model = train_model_with_cv(train_ngrams, test_ngrams, LABEL_COLUMNS, 'Data', RandomForestClassifier, params_ngrams, model_name="ngram_model")
    # Save the model and vectorizer
    os.makedirs('models', exist_ok=True)
    joblib.dump(ngram_model, 'models/ngram_rf_model.joblib')
    joblib.dump(ngram_model, 'models/ngram_rf_vectorizer.joblib')


if __name__ == "__main__":
    main()