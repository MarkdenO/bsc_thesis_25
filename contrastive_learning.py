import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score, hamming_loss
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
import time
from sklearn.manifold import TSNE
import pandas as pd
import plotly.express as px
import umap.umap_ as umap
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


def load_dataset(file_path):
    """
    Load dataset from JSON file
    """
    print(f"Loading dataset from {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    code_snippets = []
    labels = []
    
    for item in data:
        # Check for both possible key formats
        code_key = 'Data' if 'Data' in item else 'code'
        labels_key = 'Labels' if 'Labels' in item else 'labels'
        
        if code_key in item and labels_key in item:
            # Filter out null/empty data
            if item[code_key] is not None and str(item[code_key]).strip():
                code_snippets.append(str(item[code_key]))
                
                # Handle both string and list labels
                item_labels = item[labels_key]
                if isinstance(item_labels, str):
                    labels.append([item_labels])
                elif isinstance(item_labels, list):
                    # Filter out empty or null labels
                    clean_labels = [label for label in item_labels if label is not None and str(label).strip()]
                    labels.append(clean_labels if clean_labels else ['Unknown'])
                else:
                    labels.append(['Unknown'])
    
    print(f"Loaded {len(code_snippets)} samples from {file_path}")
    return code_snippets, labels


def create_tsne(X_train, X_train_embedded, y_train, mlb):
    """
    Create t-SNE visualization for TRAIN embeddings
    """
    print("Running t-SNE on TRAIN embeddings...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(X_train_embedded)-1), n_iter=1000, random_state=42, init='pca', learning_rate='auto')
    if len(X_train_embedded) <= tsne.perplexity:
         print(f"Skipping t-SNE: Number of samples ({len(X_train_embedded)}) is too small for perplexity ({tsne.perplexity}).")
    else:
        X_tsne_train = tsne.fit_transform(X_train_embedded)
        print(f"t-SNE completed. Shape: {X_tsne_train.shape}")
        print("Preparing interactive t-SNE plot for TRAIN data...")

        # Prepare tooltip labels from y_train
        labels_text_train = [
            ', '.join(lbls) if lbls else 'No Label'
            for lbls in mlb.inverse_transform(y_train) 
        ]
        truncated_code_snippets_train = [code[:80] + '...' if len(code) > 80 else code for code in X_train]

        tsne_df_train = pd.DataFrame({
            'TSNE-1': X_tsne_train[:, 0],
            'TSNE-2': X_tsne_train[:, 1],
            'Label': labels_text_train,
            'Code Snippet': truncated_code_snippets_train
        })

        # Save tsne results and metadata to csv
        tsne_df_train.to_csv("illustrations/tsne_train_embeddings4.csv", index=False)
        
        os.makedirs("illustrations", exist_ok=True)
        fig = px.scatter(
            tsne_df_train, x='TSNE-1', y='TSNE-2', color='Label',
            hover_data={'Label': True, 'Code Snippet': True},
            title='Interactive t-SNE of TRAIN Code Embeddings'
        )
        fig.write_html("illustrations/interactive_tsne_plot_train_embeddings4.html")
        print("Interactive t-SNE plot saved to 'interactive_tsne_plot_train_embeddings4.html'")


def create_umap(X_train, X_train_embedded, y_train, mlb):
    """
    Create UMAP visualization for TRAIN embeddings
    """
    print("Running UMAP on TRAIN embeddings...")

    if len(X_train_embedded) < 2:
        print("Skipping UMAP: Not enough samples.")
        return

    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap_train = reducer.fit_transform(X_train_embedded)
    print(f"UMAP completed. Shape: {X_train_embedded.shape}")
    print("Preparing interactive UMAP plot for TRAIN data...")

    # Convert y_train multi-hot to label strings
    labels_text_train = [
        ', '.join(lbls) if lbls else 'No Label'
        for lbls in mlb.inverse_transform(y_train) 
    ]

    # Truncate code snippet representations
    truncated_code_snippets_train = [
        str(code)[:80] + '...' if len(str(code)) > 80 else str(code)
        for code in X_train
    ]

    # Build DataFrame for Plotly
    umap_df_train = pd.DataFrame({
        'UMAP-1': X_umap_train[:, 0],
        'UMAP-2': X_umap_train[:, 1],
        'Label': labels_text_train,
        'Code Snippet': truncated_code_snippets_train
    })

    os.makedirs("illustrations", exist_ok=True)
    fig = px.scatter(
        umap_df_train, x='UMAP-1', y='UMAP-2', color='Label',
        hover_data={'Label': True, 'Code Snippet': True},
        title='Interactive UMAP of TRAIN Code Embeddings'
    )
    fig.write_html("illustrations/interactive_umap_plot_train_embeddings.html")
    print("Interactive UMAP plot saved to 'interactive_umap_plot_train_embeddings.html'")


class SupConLossMultiLabel(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, multi_hot_labels):
        """
        Compute the contrastive loss for multi-label classification.
        """
        device = features.device
        multi_hot_labels = multi_hot_labels.float()

        label_similarity_matrix = torch.matmul(multi_hot_labels, multi_hot_labels.T)
        mask = (label_similarity_matrix > 0).float().to(device)

        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0]).to(device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mask_sum = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask_sum + 1e-12)

        loss = -mean_log_prob_pos.mean()
        return loss

class ContrastiveCodeModel(nn.Module):
    """
    Contrastive model using CodeBERT with a projection head.
    """
    def __init__(self, codebert_model_name_or_path, projection_dim):
        super().__init__()
        self.codebert = RobertaModel.from_pretrained(codebert_model_name_or_path)
        self.projector = nn.Sequential(
            nn.Linear(self.codebert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        """
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state

        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
        sum_mask = mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_pooled_embeddings = sum_embeddings / sum_mask

        projected_features = self.projector(mean_pooled_embeddings)
        return F.normalize(projected_features, p=2, dim=1)

class ContrastiveCodeDataset(Dataset): 
    def __init__(self, code_snippets, labels, tokenizer, max_length):
        self.code_snippets = code_snippets
        self.labels = torch.tensor(labels, dtype=torch.float32) if not isinstance(labels, torch.Tensor) else labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.code_snippets)

    def __getitem__(self, idx):
        code = str(self.code_snippets[idx])
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        return input_ids, attention_mask, self.labels[idx]

def get_embeddings(model, dataloader, device):
    """
    Generate embeddings for the dataset using the contrastive model.
    """
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for input_ids, attention_mask, _ in dataloader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            batch_embeddings = model(input_ids, attention_mask)
            all_embeddings.append(batch_embeddings.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)


def evaluate_contrastive_model(model, dataloader, loss_fn, device):
    """
    Evaluate contrastive model on validation/test set
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for input_ids, attention_mask, batch_y_labels in dataloader:
            input_ids, attention_mask, batch_y_labels = \
                input_ids.to(device), attention_mask.to(device), batch_y_labels.to(device)
            
            features = model(input_ids, attention_mask)
            loss = loss_fn(features, batch_y_labels)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0


def evaluate_downstream_classifier(clf, X_embedded, y_true, mlb):
    """
    Evaluate downstream classifier and return metrics
    """
    y_pred = clf.predict(X_embedded)
    
    # Calculate exact match accuracy
    exact_match_accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate at least one correct label accuracy
    at_least_one_correct = np.sum(np.sum((y_true == 1) & (y_pred == 1), axis=1) > 0) / len(y_true)

    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        'exact_match_accuracy': exact_match_accuracy,
        'at_least_one_correct': at_least_one_correct,
        'y_pred': y_pred,
        'f1_score': f1
    }


def main():
    CODEBERT_MODEL_NAME = 'microsoft/codebert-base'
    
    # Dataset paths
    TRAIN_PATH = './datasets/train.json'
    VAL_PATH = './datasets/val.json'
    TEST_PATH = './datasets/test.json'

    TEST_LB = './datasets/test_leaderboard.json'
    TEST_GITHUB = './datasets/test_github.json'
    TEST_REDDIT = './datasets/test_reddit.json'

    MULTI_LABEL_TEST_PATH = './datasets/test_multi.json'
    SINGLE_LABEL_TEST_PATH = './datasets/test_single.json'
    
    MAX_TOKEN_LENGTH = 512

    # Contrastive Learning Hyperparameters
    PROJECTION_DIM = 128
    CONTRASTIVE_BATCH_SIZE = 32
    CONTRASTIVE_EPOCHS = 100
    CONTRASTIVE_LR = 2e-5
    TEMPERATURE = 0.07
    
    # Early stopping parameters
    PATIENCE = 10
    MIN_DELTA = 0.001

    # Downstream Classifier Hyperparameters
    LOGISTIC_MAX_ITER = 1000
    EMBEDDING_BATCH_SIZE = 64

    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")  # Force CPU for compatibility
    print(f"Using device: {device}")

    # Load datasets
    print("Loading datasets...")
    X_train, y_train_labels = load_dataset(TRAIN_PATH)
    X_val, y_val_labels = load_dataset(VAL_PATH)

    test_datasources = False

    if test_datasources:
        print("Loading test datasets for different sources...")
        X_test_lb, y_test_lb_labels = load_dataset(TEST_LB)
        X_test_github, y_test_github_labels = load_dataset(TEST_GITHUB)
        X_test_reddit, y_test_reddit_labels = load_dataset(TEST_REDDIT)

    test_diff_labels = True

    if test_diff_labels:
        print("Loading test dataset with different amount labels...")
        # Load test dataset with different labels
        X_test_multi, y_test_multi_labels = load_dataset(MULTI_LABEL_TEST_PATH)
        X_test_single, y_test_single_labels = load_dataset(SINGLE_LABEL_TEST_PATH)

  
    X_test, y_test_labels = load_dataset(TEST_PATH)

    # Create MultiLabelBinarizer and fit on all labels
    print("Processing labels...")
    all_labels = y_train_labels + y_val_labels + y_test_labels
    mlb = MultiLabelBinarizer()
    mlb.fit(all_labels)
    
    # Transform labels to multi-hot encoding
    y_train = mlb.transform(y_train_labels)
    y_val = mlb.transform(y_val_labels)
    y_test = mlb.transform(y_test_labels)

    if test_datasources:
        y_test_lb = mlb.transform(y_test_lb_labels)
        y_test_github = mlb.transform(y_test_github_labels)
        y_test_reddit = mlb.transform(y_test_reddit_labels)

    if test_diff_labels:
        y_test_multi = mlb.transform(y_test_multi_labels)
        y_test_single = mlb.transform(y_test_single_labels)

    
    num_classes = y_train.shape[1]
    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")
    print(f"Number of unique labels: {num_classes}")
    print(f"Label classes: {mlb.classes_}")

    tokenizer = RobertaTokenizer.from_pretrained(CODEBERT_MODEL_NAME)

    skip_contrastive_training = True

    # Contrastive pre-training with validation
    print("\n1: Contrastive Pre-training with Validation")

    contrastive_model = ContrastiveCodeModel(CODEBERT_MODEL_NAME, PROJECTION_DIM).to(device)
    contrastive_model_path = 'full_contrastive_encoder_model_with_validation_128_64.pth'

    if not skip_contrastive_training:
        # Create datasets and dataloaders
        contrastive_train_dataset = ContrastiveCodeDataset(X_train, y_train, tokenizer, MAX_TOKEN_LENGTH)
        contrastive_val_dataset = ContrastiveCodeDataset(X_val, y_val, tokenizer, MAX_TOKEN_LENGTH)
        
        contrastive_train_loader = DataLoader(
            contrastive_train_dataset, batch_size=CONTRASTIVE_BATCH_SIZE, shuffle=True, num_workers=0
        )
        contrastive_val_loader = DataLoader(
            contrastive_val_dataset, batch_size=CONTRASTIVE_BATCH_SIZE, shuffle=False, num_workers=0
        )

        contrastive_loss_fn = SupConLossMultiLabel(temperature=TEMPERATURE)
        optimizer_contrastive = torch.optim.AdamW(contrastive_model.parameters(), lr=CONTRASTIVE_LR)
        
        # Training with validation and early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = {'train_loss': [], 'val_loss': []}

        # Load existing model and continue training if available
        if os.path.exists(contrastive_model_path):
            contrastive_model.load_state_dict(torch.load(contrastive_model_path, map_location=device))
            print(f"Loaded pre-trained contrastive model from '{contrastive_model_path}'")
        else:
            print(f"Starting training from scratch...")

        for epoch in range(CONTRASTIVE_EPOCHS):
            epoch_start_time = time.time()
            
            # Training phase
            contrastive_model.train()
            total_train_loss = 0
            for batch_idx, (input_ids, attention_mask, batch_y_labels) in enumerate(contrastive_train_loader):
                input_ids, attention_mask, batch_y_labels = \
                    input_ids.to(device), attention_mask.to(device), batch_y_labels.to(device)

                features = contrastive_model(input_ids, attention_mask)
                loss = contrastive_loss_fn(features, batch_y_labels)

                optimizer_contrastive.zero_grad()
                loss.backward()
                optimizer_contrastive.step()
                total_train_loss += loss.item()

                if (batch_idx + 1) % 20 == 0:
                    print(f"  Epoch {epoch+1}/{CONTRASTIVE_EPOCHS} | Batch {batch_idx+1}/{len(contrastive_train_loader)} | Loss: {loss.item():.4f}")

            avg_train_loss = total_train_loss / len(contrastive_train_loader)
            
            # Validation phase
            avg_val_loss = evaluate_contrastive_model(contrastive_model, contrastive_val_loader, contrastive_loss_fn, device)
            
            # Record history
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(avg_val_loss)
            
            epoch_time = time.time() - epoch_start_time
            print(f"Contrastive Epoch {epoch+1}/{CONTRASTIVE_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {epoch_time:.2f}s")
            
            # Early stopping check
            if avg_val_loss < best_val_loss - MIN_DELTA:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(contrastive_model.state_dict(), contrastive_model_path)
                print(f"  New best validation loss: {best_val_loss:.4f} - Model saved")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{PATIENCE}")
                
                if patience_counter >= PATIENCE:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            # torch.mps.empty_cache()

        # Load best model
        contrastive_model.load_state_dict(torch.load(contrastive_model_path, map_location=device))
        print(f"Loaded best contrastive model with validation loss: {best_val_loss:.4f}")
        
        # Save training history
        with open('contrastive_training_history.pkl', 'wb') as f:
            pickle.dump(training_history, f)
        print("Training history saved to 'contrastive_training_history.pkl'")
        
    else:
        try:
            contrastive_model.load_state_dict(torch.load(contrastive_model_path, map_location=device))
            print(f"Loaded pre-trained contrastive model from '{contrastive_model_path}'")
        except FileNotFoundError:
            print(f"Error: Pre-trained model not found at {contrastive_model_path}. Please train first or check path.")
            return

    # Downstream classification with validation optimization
    print("\n2: Downstream Classification with Validation")
    contrastive_model.eval()

    if test_datasources:
        # Create embedding datasets and dataloaders for test sources
        test_lb_embedding_dataset = ContrastiveCodeDataset(X_test_lb, y_test_lb, tokenizer, MAX_TOKEN_LENGTH)
        test_github_embedding_dataset = ContrastiveCodeDataset(X_test_github, y_test_github, tokenizer, MAX_TOKEN_LENGTH)
        test_reddit_embedding_dataset = ContrastiveCodeDataset(X_test_reddit, y_test_reddit, tokenizer, MAX_TOKEN_LENGTH)

        test_lb_embedding_loader = DataLoader(test_lb_embedding_dataset, batch_size=EMBEDDING_BATCH_SIZE, shuffle=False, num_workers=0)
        test_github_embedding_loader = DataLoader(test_github_embedding_dataset, batch_size=EMBEDDING_BATCH_SIZE, shuffle=False, num_workers=0)
        test_reddit_embedding_loader = DataLoader(test_reddit_embedding_dataset, batch_size=EMBEDDING_BATCH_SIZE, shuffle=False, num_workers=0)
        print("Generating embeddings for test sources...")
        X_test_lb_embedded = get_embeddings(contrastive_model, test_lb_embedding_loader, device)
        X_test_github_embedded = get_embeddings(contrastive_model, test_github_embedding_loader, device)
        X_test_reddit_embedded = get_embeddings(contrastive_model, test_reddit_embedding_loader, device)
        print(f"Generated embeddings - Test LB: {X_test_lb_embedded.shape}, Test GitHub: {X_test_github_embedded.shape}, Test Reddit: {X_test_reddit_embedded.shape}")

    if test_diff_labels:
        # Create embedding datasets and dataloaders for different label tests
        test_multi_embedding_dataset = ContrastiveCodeDataset(X_test_multi, y_test_multi, tokenizer, MAX_TOKEN_LENGTH)
        test_single_embedding_dataset = ContrastiveCodeDataset(X_test_single, y_test_single, tokenizer, MAX_TOKEN_LENGTH)

        test_multi_embedding_loader = DataLoader(test_multi_embedding_dataset, batch_size=EMBEDDING_BATCH_SIZE, shuffle=False, num_workers=0)
        test_single_embedding_loader = DataLoader(test_single_embedding_dataset, batch_size=EMBEDDING_BATCH_SIZE, shuffle=False, num_workers=0)

        print("Generating embeddings for multi-label and single-label tests...")
        X_test_multi_embedded = get_embeddings(contrastive_model, test_multi_embedding_loader, device)
        X_test_single_embedded = get_embeddings(contrastive_model, test_single_embedding_loader, device)
        print(f"Generated embeddings - Test Multi: {X_test_multi_embedded.shape}, Test Single: {X_test_single_embedded.shape}")


    # Create embedding datasets and dataloaders
    train_embedding_dataset = ContrastiveCodeDataset(X_train, y_train, tokenizer, MAX_TOKEN_LENGTH)
    val_embedding_dataset = ContrastiveCodeDataset(X_val, y_val, tokenizer, MAX_TOKEN_LENGTH)
    test_embedding_dataset = ContrastiveCodeDataset(X_test, y_test, tokenizer, MAX_TOKEN_LENGTH)

    train_embedding_loader = DataLoader(train_embedding_dataset, batch_size=EMBEDDING_BATCH_SIZE, shuffle=False, num_workers=0)
    val_embedding_loader = DataLoader(val_embedding_dataset, batch_size=EMBEDDING_BATCH_SIZE, shuffle=False, num_workers=0)
    test_embedding_loader = DataLoader(test_embedding_dataset, batch_size=EMBEDDING_BATCH_SIZE, shuffle=False, num_workers=0)

    # Generate embeddings
    print("Generating embeddings for all datasets...")
    X_train_embedded = get_embeddings(contrastive_model, train_embedding_loader, device)
    X_val_embedded = get_embeddings(contrastive_model, val_embedding_loader, device)
    X_test_embedded = get_embeddings(contrastive_model, test_embedding_loader, device)
    
    print(f"Generated embeddings - Train: {X_train_embedded.shape}, Val: {X_val_embedded.shape}, Test: {X_test_embedded.shape}")

    # Visualization
    run_tsne = False
    run_umap = False
    if run_tsne:
        create_tsne(X_train, X_train_embedded, y_train, mlb)
    elif run_umap:
        create_umap(X_train, X_train_embedded, y_train, mlb)

    # Train downstream classifier with hyperparameter optimization on validation set
    print("Training downstream classifier with validation optimization...")
    
    # Try different hyperparameters
    n_estimators_options = [100]
    max_depth_options = [10]
    min_samples_split_options = [5]

    best_val_accuracy = 0
    best_params = {}
    best_clf = None

    for n_estimators in n_estimators_options:
        for max_depth in max_depth_options:
            for min_samples_split in min_samples_split_options:
                print(f"  Testing n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}")
                
                clf = MultiOutputClassifier(
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        random_state=42,
                        n_jobs=-1
                    )
                )
                
                clf.fit(X_train_embedded, y_train)
                val_metrics = evaluate_downstream_classifier(clf, X_val_embedded, y_val, mlb)
                
                val_accuracy = val_metrics['f1_score']
                print(f"    Validation f1: {val_accuracy:.4f}")
                
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_params = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split
                    }
                    best_clf = clf
                    print(f"    New best f1: {best_val_accuracy:.4f}")

    print(f"\nBest parameters: {best_params}")
    print(f"Best validation f1: {best_val_accuracy:.4f}")

    # Final evaluation on test set
    print("\n3: Final Evaluation on Test Set")
    test_metrics = evaluate_downstream_classifier(best_clf, X_test_embedded, y_test, mlb)

    if test_datasources:
        test_metrics_lb = evaluate_downstream_classifier(best_clf, X_test_lb_embedded, y_test_lb, mlb)
        test_metrics_github = evaluate_downstream_classifier(best_clf, X_test_github_embedded, y_test_github, mlb)
        test_metrics_reddit = evaluate_downstream_classifier(best_clf, X_test_reddit_embedded, y_test_reddit, mlb)

        print("\nTest Metrics for Leaderboard Data:")
        print(f"Exact Match Accuracy: {test_metrics_lb['exact_match_accuracy']:.4f}")
        print(f"At Least One Correct: {test_metrics_lb['at_least_one_correct']:.4f}")
        print(f'Hamming loss: {hamming_loss(y_test_lb, test_metrics_lb["y_pred"]):.4f}')
        print(classification_report(
            y_test_lb,
            test_metrics_lb['y_pred'],
            target_names=mlb.classes_,
            zero_division=0
        ))

        print("\nTest Metrics for GitHub Data:")
        print(f"Exact Match Accuracy: {test_metrics_github['exact_match_accuracy']:.4f}")
        print(f"At Least One Correct: {test_metrics_github['at_least_one_correct']:.4f}")
        print(f'Hamming loss: {hamming_loss(y_test_github, test_metrics_github["y_pred"]):.4f}')
        print(classification_report(
            y_test_github,
            test_metrics_github['y_pred'],
            target_names=mlb.classes_,
            zero_division=0
        ))

        print("\nTest Metrics for Reddit Data:")
        print(f"Exact Match Accuracy: {test_metrics_reddit['exact_match_accuracy']:.4f}")
        print(f"At Least One Correct: {test_metrics_reddit['at_least_one_correct']:.4f}")
        print(f'Hamming loss: {hamming_loss(y_test_reddit, test_metrics_reddit["y_pred"]):.4f}')
        print(classification_report(
            y_test_reddit,
            test_metrics_reddit['y_pred'],
            target_names=mlb.classes_,
            zero_division=0
        ))

    if test_diff_labels:
        test_metrics_multi = evaluate_downstream_classifier(best_clf, X_test_multi_embedded, y_test_multi, mlb)
        test_metrics_single = evaluate_downstream_classifier(best_clf, X_test_single_embedded, y_test_single, mlb)

        print("\nTest Metrics for Multi-Label Data:")
        print(f"Exact Match Accuracy: {test_metrics_multi['exact_match_accuracy']:.4f}")
        print(f"At Least One Correct: {test_metrics_multi['at_least_one_correct']:.4f}")
        print(f'Hamming loss: {hamming_loss(y_test_multi, test_metrics_multi["y_pred"]):.4f}')
        print(classification_report(
            y_test_multi,
            test_metrics_multi['y_pred'],
            target_names=mlb.classes_,
            zero_division=0
        ))

        print("\nTest Metrics for Single-Label Data:")
        print(f"Exact Match Accuracy: {test_metrics_single['exact_match_accuracy']:.4f}")
        print(f"At Least One Correct: {test_metrics_single['at_least_one_correct']:.4f}")
        print(f'Hamming loss: {hamming_loss(y_test_single, test_metrics_single["y_pred"]):.4f}')
        print(classification_report(
            y_test_single,
            test_metrics_single['y_pred'],
            target_names=mlb.classes_,
            zero_division=0
        ))
    
    print(f"Test exact match accuracy: {test_metrics['exact_match_accuracy']:.4f}")
    print(f"Test at least one correct accuracy: {test_metrics['at_least_one_correct']:.4f}")
    print(f"Hamming loss: {hamming_loss(y_test, test_metrics['y_pred']):.4f}")
    
    # Detailed classification report
    print("\nDetailed Classification Report (Test Set)")
    print(classification_report(
        y_test, 
        test_metrics['y_pred'],
        target_names=mlb.classes_,
        zero_division=0
    ))

    # Save results
    results = {
        'best_params': best_params,
        'best_val_accuracy': best_val_accuracy,
        'test_metrics': test_metrics,
        'mlb_classes': mlb.classes_.tolist()
    }
    
    with open('final_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("\nResults saved to 'final_results.pkl'")

    return contrastive_model, best_clf, mlb, results


if __name__ == "__main__":
    main()