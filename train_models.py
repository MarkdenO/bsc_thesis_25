
# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from sklearn.svm import LinearSVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.linear_model import LogisticRegression
# import numpy as np
# import joblib
# import os
# from contrastive_learning import *
# from sklearn.multiclass import OneVsRestClassifier
# from skmultilearn.adapt import MLkNN
# from scipy.sparse import csr_matrix
# import re


# def load_datasets():
#     """
#     Load pre-split datasets from the /datasets directory.
#     Returns:
#         train_data (pd.DataFrame): Training dataset.
#         val_data (pd.DataFrame): Validation dataset.
#         test_data (pd.DataFrame): Test dataset.
#     """
#     train_path = os.path.join('datasets', 'train.json')
#     val_path = os.path.join('datasets', 'val.json')
#     test_path = os.path.join('datasets', 'test.json')

#     if not all(os.path.exists(path) for path in [train_path, val_path, test_path]):
#         raise FileNotFoundError("train.json, val.json, or test.json not found in /datasets directory.")

#     train_data = pd.read_json(train_path)
#     val_data = pd.read_json(val_path)
#     test_data = pd.read_json(test_path)

#     return train_data, val_data, test_data

# def custom_tokenizer(text):
#     # Split on word boundaries, keep symbols as tokens
#     return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)



# LABEL_COLUMNS = [
#     'Warm', 'Gram', 'Str', 'Math', 'Sptl', 'Img', 'Cell', 'Grid', 'Grph', 'Path', 'BFS', 'DFS',
#     'Dyn', 'Memo', 'Opt', 'Log', 'Bit', 'VM', 'Rev', 'Sim', 'Inp', 'Scal'
# ]

# def preprocess(df, preprocessing_type, label_columns):
#     # df_filtered = dataframe[preprocessing_type]
#     # print(f"[{preprocessing_type}] Rows remaining after filtering: {len(df_filtered)}")

#     X = df[preprocessing_type].astype(str).fillna('')
#     Y = df[label_columns].notna().astype(int)
#     vectorizer = CountVectorizer()
#     X_vect = vectorizer.fit_transform(X)
#     return X_vect, Y, vectorizer



# def train_mlp_on_ast(train_data, val_data, test_data, label_columns, preprocessing_type):
#     X_train, Y_train, vectorizer = preprocess(train_data, preprocessing_type, label_columns)
#     X_val, Y_val, _ = preprocess(val_data, preprocessing_type, label_columns)
#     X_test, Y_test, _ = preprocess(test_data, preprocessing_type, label_columns)
#     model = MultiOutputClassifier(MLPClassifier(hidden_layer_sizes=(512,), max_iter=1000, random_state=42, verbose=True))
#     model.fit(X_train, Y_train)
#     Y_pred = model.predict(X_test)
#     print("MLP Classification Report:")
#     print(classification_report(Y_test, Y_pred, target_names=label_columns, zero_division=0))

#     # Calculate accuracy and accuracy as Jaccard subset
#     accuracy_test = accuracy_score(Y_test, Y_pred)
#     print("Exact match accuracy (on TEST set):", accuracy_test)

#     correct_predictions_test = np.all(Y_test == Y_pred, axis=1) # Exact match
#     at_least_one_correct_test = np.sum(np.sum((Y_test == 1) & (Y_pred == 1), axis=1) > 0) / len(Y_test)
#     print("At least one correct prediction:", at_least_one_correct_test)

#     # Save vectorizer
#     joblib.dump(vectorizer, f'models/{preprocessing_type}_mlp_vectorizer.joblib')


#     return model


# def train_svm(train_data, val_data, test_data, label_columns, preprocessing_type):
#     X_train, Y_train, vectorizer = preprocess(train_data, preprocessing_type, label_columns)
#     X_val, Y_val, _ = preprocess(val_data, preprocessing_type, label_columns)
#     X_test, Y_test, _ = preprocess(test_data, preprocessing_type, label_columns)

#     model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
#     model.fit(X_train, Y_train)
#     Y_prob = model.predict_proba(X_test)

#     # Dit tunen zorgt ervoor dat er iets wordt voorspeld, maar de voorspelling is alsnog matig.
#     Y_pred = (Y_prob > 0.1).astype(int)

#     print("SVM Classification Report:")
#     print(classification_report(Y_test, Y_pred, target_names=label_columns, zero_division=0))

#     # show_example_predictions(X_test, Y_test, Y_pred, label_columns)

#     accuracy_test = accuracy_score(Y_test, Y_pred)
#     print("Exact match accuracy (on TEST set):", accuracy_test)

#     correct_predictions_test = np.all(Y_test == Y_pred, axis=1)
#     at_least_one_correct_test = np.sum(np.sum((Y_test == 1) & (Y_pred == 1), axis=1) > 0) / len(Y_test)
#     print("At least one correct prediction:", at_least_one_correct_test)

#     joblib.dump(vectorizer, f'models/{preprocessing_type}_svm_vectorizer.joblib')

#     return model

# def train_random_forest(train_data, val_data, test_data, label_columns, preprocessing_type):
#     X_train, Y_train, vectorizer = preprocess(train_data, preprocessing_type, label_columns)
#     X_val, Y_val, _ = preprocess(val_data, preprocessing_type, label_columns)
#     X_test, Y_test, _ = preprocess(test_data, preprocessing_type, label_columns)
#     # model = MultiOutputClassifier(RandomForestClassifier(n_estimators=200, random_state=42))

#     model = OneVsRestClassifier(RandomForestClassifier(n_estimators=200, random_state=42))
#     model.fit(X_train, Y_train)
#     Y_pred = model.predict(X_test)
#     print("Random Forest Classification Report:")
#     print(classification_report(Y_test, Y_pred, target_names=label_columns, zero_division=0))

#     show_example_predictions(X_test, Y_test, Y_pred, label_columns)

#     # Calculate accuracy and accuracy as Jaccard subset
#     accuracy_test = accuracy_score(Y_test, Y_pred)
#     print("Exact match accuracy (on TEST set):", accuracy_test)

#     correct_predictions_test = np.all(Y_test == Y_pred, axis=1) # Exact match
#     at_least_one_correct_test = np.sum(np.sum((Y_test == 1) & (Y_pred == 1), axis=1) > 0) / len(Y_test)
#     print("At least one correct prediction:", at_least_one_correct_test)

#     # Save vectorizer
#     joblib.dump(vectorizer, f'models/{preprocessing_type}_rf_vectorizer.joblib')


#     return model


# # def train_bow_classifier(dataframe, label_columns, preprocessing_type):
# #     X = dataframe[preprocessing_type].astype(str).fillna('')
# #     Y = dataframe[label_columns].notna().astype(int)

# #     vectorizer = CountVectorizer(tokenizer=custom_tokenizer, lowercase=False, token_pattern=None)
# #     X_vect = vectorizer.fit_transform(X)

# #     X_train, X_test, Y_train, Y_test = train_test_split(X_vect, Y, test_size=0.2, random_state=42)

# #     model = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42))
# #     model.fit(X_train, Y_train)

# #     Y_pred = model.predict(X_test)
# #     print("Bag-of-Tokens Logistic Regression Classification Report:")
# #     print(classification_report(Y_test, Y_pred, target_names=label_columns, zero_division=0))
# #     accuracy_test = accuracy_score(Y_test, Y_pred)
# #     print("Exact match accuracy (on TEST set):", accuracy_test)
    
# #     show_example_predictions(X_test, Y_test, Y_pred, label_columns)

# #     os.makedirs('models', exist_ok=True)
# #     joblib.dump(model, f'models/{preprocessing_type}_bow_lr_model.joblib')
# #     joblib.dump(vectorizer, f'models/{preprocessing_type}_bow_vectorizer.joblib')


# #     return model


# def train_bow_classifier(train_data, val_data, test_data, label_columns, preprocessing_type):
#     X_train, Y_train, vectorizer = preprocess(train_data, preprocessing_type, label_columns)
#     X_val, Y_val, _ = preprocess(val_data, preprocessing_type, label_columns)
#     X_test, Y_test, _ = preprocess(test_data, preprocessing_type, label_columns)

#     model = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=42))
#     model.fit(X_train, Y_train)

#     Y_pred = model.predict(X_test)
#     print("Bag-of-Tokens Logistic Regression Classification Report:")
#     print(classification_report(Y_test, Y_pred, target_names=label_columns, zero_division=0))

#     accuracy_test = accuracy_score(Y_test, Y_pred)
#     print("Exact match accuracy (on TEST set):", accuracy_test)

#     correct_predictions_test = np.all(Y_test == Y_pred, axis=1) # Exact match
#     at_least_one_correct_test = np.sum(np.sum((Y_test == 1) & (Y_pred == 1), axis=1) > 0) / len(Y_test)
#     print("At least one correct prediction:", at_least_one_correct_test)

#     joblib.dump(vectorizer, f'models/{preprocessing_type}_bow_vectorizer.joblib')

#     return model



# def train_knn(dataframe, label_columns, preprocessing_type):
#     X, Y, vectorizer = preprocess(dataframe, preprocessing_type, label_columns)
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#     model = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5))
#     model.fit(X_train, Y_train)
#     Y_pred = model.predict(X_test)
#     print("K-Nearest Neighbors Classification Report:")
#     print(classification_report(Y_test, Y_pred, target_names=label_columns, zero_division=0))

#     # show_example_predictions(X_test, Y_test, Y_pred, label_columns)

#     # Calculate accuracy and accuracy as Jaccard subset
#     accuracy_test = accuracy_score(Y_test, Y_pred)
#     print("Exact match accuracy (on TEST set):", accuracy_test)

#     correct_predictions_test = np.all(Y_test == Y_pred, axis=1) # Exact match
#     at_least_one_correct_test = np.sum(np.sum((Y_test == 1) & (Y_pred == 1), axis=1) > 0) / len(Y_test)
#     print("At least one correct prediction:", at_least_one_correct_test)

#     joblib.dump(vectorizer, f'models/{preprocessing_type}_knn_vectorizer.joblib')


#     return model


# def show_example_predictions(X_test, Y_test, Y_pred, label_columns, n=10, original_texts=None):
#     print("\n--- Example Predictions ---")

#     n_samples = X_test.shape[0]
#     n = min(n, n_samples)

#     if n == 0:
#         print("No test samples to show.")
#         return

#     for i in range(n):
#         true_labels_indices = np.where(Y_test.iloc[i] == 1)[0]
#         predicted_labels_indices = np.where(Y_pred[i] == 1)[0]

#         true_labels = [label_columns[idx] for idx in true_labels_indices]
#         predicted_labels = [label_columns[idx] for idx in predicted_labels_indices]

#         print(f"Sample {i + 1}")
#         if original_texts is not None:
#             print(f"  Text: {original_texts.iloc[i]}")
#         print(f"  True:      {true_labels if true_labels else ['(None)']}")
#         print(f"  Predicted: {predicted_labels if predicted_labels else ['(None)']}")
#         print("------")


# def show_astnn_predictions(Y_true, Y_pred, mlb, n=10):
#     """
#     Display example predictions for ASTNN using mlb.inverse_transform.
#     """
#     n = min(n, len(Y_true))
#     true_label_names = mlb.inverse_transform(Y_true)
#     pred_label_names = mlb.inverse_transform(Y_pred)
#     print("\n--- Example Predictions (ASTNN) ---")
#     for i in range(n):
#         print(f"Sample {i + 1}")
#         print(f"  True:      {true_label_names[i] if true_label_names[i] else ['(None)']}")
#         print(f"  Predicted: {pred_label_names[i] if pred_label_names[i] else ['(None)']}")
#         print("------")

# def main():
#     # Load pre-split datasets
#     train_data, val_data, test_data = load_datasets()

#     print("\nNgrams")
#     print("Training Random Forest on Ngrams")
#     rf_model = train_random_forest(train_data, val_data, test_data, LABEL_COLUMNS, 'Data')
#     joblib.dump(rf_model, 'models/ngrams_rf_model.joblib')


#     # # AST
#     # print("\nAST")
#     # # print("Training MLP on AST")
#     # df_ast = pd.read_pickle('results/ast.pkl')
#     # ast_model = train_mlp_on_ast(df_ast, LABEL_COLUMNS, 'ast')
    
#     # # Save model
#     # os.makedirs('models', exist_ok=True)
#     # joblib.dump(ast_model, 'models/ast_mlp_model.joblib')


#     # print("\nTraining SVM on AST")
#     # svm_model = train_svm(df_ast, LABEL_COLUMNS, 'ast')
#     # joblib.dump(svm_model, 'models/ast_svm_model.joblib')

#     # Bag of Words
#     print("\nBag of Words")
#     print("Training Bag of Words Classifier")
#     bow_model = train_bow_classifier(train_data, val_data, test_data, LABEL_COLUMNS, 'Data')
#     joblib.dump(bow_model, 'models/bow_lr_model.joblib')
    


# if __name__ == "__main__":
#     main()

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
    tokens = re.findall(r'\w+|[+\-*/=<>!&|]+|\d+|[(){}\[\];,.]', code)
    return tokens


def load_datasets(data_type=''):
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
    X = df['Data'].astype(str).fillna('')
    Y = pd.DataFrame(0, index=np.arange(len(df)), columns=LABEL_COLUMNS)

    for i, label_list in enumerate(df['Labels']):
        for label in label_list:
            if label in LABEL_COLUMNS:
                Y.at[i, label] = 1

    X_vect = vectorizer.transform(X)
    return X_vect, Y



def train_model_with_cv(train_data, test_data, label_columns, text_column, model_cls, param_grid, model_name="model"):
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
    data_sources = test_data['DataSource'].unique()
    for source in data_sources:
        X_test_source = test_data[test_data['DataSource'] == source]
        X_test_vect, Y_test_source = preprocess_with_existing_vectorizer(X_test_source, vectorizer)
        Y_pred_source = best_model.predict(X_test_vect)
        print(f"\n{model_name} Classification Report for DataSource '{source}':")
        print(classification_report(Y_test_source, Y_pred_source, target_names=label_columns, zero_division=0))
        print("Exact match accuracy for DataSource '{}':".format(source), accuracy_score(Y_test_source, Y_pred_source))
        print("At least one correct prediction for DataSource '{}':".format(source),
              np.sum(np.sum((Y_test_source == 1) & (Y_pred_source == 1), axis=1) > 0) / len(Y_test_source))
        print("Hamming loss for DataSource '{}':".format(source), hamming_loss(Y_test_source, Y_pred_source))

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


    # Define parameter grid for Random Forest
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