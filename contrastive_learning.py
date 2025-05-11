import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
import time
from sklearn.manifold import TSNE
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split


class SupConLossMultiLabel(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, multi_hot_labels):
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
    def __init__(self, codebert_model_name_or_path, projection_dim):
        super().__init__()
        self.codebert = RobertaModel.from_pretrained(codebert_model_name_or_path)
        self.projector = nn.Sequential(
            nn.Linear(self.codebert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, input_ids, attention_mask):
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
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for input_ids, attention_mask, _ in dataloader:
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            batch_embeddings = model(input_ids, attention_mask)
            all_embeddings.append(batch_embeddings.cpu().numpy())
    return np.concatenate(all_embeddings, axis=0)


# --- Main Script ---
def main():
    CODEBERT_MODEL_NAME = 'microsoft/codebert-base'
    DATA_PATH = 'labelled_data.pkl'
    MAX_TOKEN_LENGTH = 256

    # Contrastive Learning Hyperparameters
    PROJECTION_DIM = 128
    CONTRASTIVE_BATCH_SIZE = 32
    CONTRASTIVE_EPOCHS = 10
    CONTRASTIVE_LR = 2e-5
    TEMPERATURE = 0.07

    # Downstream Classifier Hyperparameters
    LOGISTIC_MAX_ITER = 1000
    EMBEDDING_BATCH_SIZE = 64

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Data ---
    print("Loading data...")

    with open(DATA_PATH, 'rb') as f:
        df = pickle.load(f)


    df = df[df['Data'].notnull()]
    df['Data'] = df['Data'].astype(str)

    X_code_snippets = df['Data'].tolist()

    mlb = MultiLabelBinarizer()
    y_multi_hot_labels_all = mlb.fit_transform(df['Labels'])
    num_classes = y_multi_hot_labels_all.shape[1]
    print(f"Loaded {len(X_code_snippets)} code snippets.")
    print(f"Number of unique labels: {num_classes}")
    print(f"Label classes: {mlb.classes_}")

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_code_snippets, y_multi_hot_labels_all, test_size=0.2, random_state=42, stratify=None
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    tokenizer = RobertaTokenizer.from_pretrained(CODEBERT_MODEL_NAME)

    skip_contrastive_training = True

    # Contrasive pre-training
    print("\n=== Phase 1: Contrastive Pre-training ===")

    contrastive_model = ContrastiveCodeModel(CODEBERT_MODEL_NAME, PROJECTION_DIM).to(device)
    contrastive_model_path = 'full_contrastive_encoder_model3.pth'

    if not skip_contrastive_training:
        contrastive_train_dataset = ContrastiveCodeDataset(
            X_train,
            y_train,
            tokenizer,
            MAX_TOKEN_LENGTH
        )
        contrastive_train_loader = DataLoader(
            contrastive_train_dataset,
            batch_size=CONTRASTIVE_BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )

        contrastive_loss_fn = SupConLossMultiLabel(temperature=TEMPERATURE)
        optimizer_contrastive = torch.optim.AdamW(contrastive_model.parameters(), lr=CONTRASTIVE_LR)

        for epoch in range(CONTRASTIVE_EPOCHS):
            epoch_start_time = time.time()
            contrastive_model.train()
            total_contrastive_loss = 0
            for batch_idx, (input_ids, attention_mask, batch_y_labels) in enumerate(contrastive_train_loader):
                input_ids, attention_mask, batch_y_labels = \
                    input_ids.to(device), attention_mask.to(device), batch_y_labels.to(device)

                features = contrastive_model(input_ids, attention_mask)
                loss = contrastive_loss_fn(features, batch_y_labels)

                optimizer_contrastive.zero_grad()
                loss.backward()
                optimizer_contrastive.step()
                total_contrastive_loss += loss.item()

                if (batch_idx + 1) % 20 == 0:
                    print(f"  Epoch {epoch+1}/{CONTRASTIVE_EPOCHS} | Batch {batch_idx+1}/{len(contrastive_train_loader)} | Loss: {loss.item():.4f}")

            avg_epoch_loss = total_contrastive_loss / len(contrastive_train_loader)
            epoch_time = time.time() - epoch_start_time
            print(f"Contrastive Epoch {epoch+1}/{CONTRASTIVE_EPOCHS} | Avg Loss: {avg_epoch_loss:.4f} | Time: {epoch_time:.2f}s")

        torch.save(contrastive_model.state_dict(), contrastive_model_path)
        print(f"Saved fine-tuned contrastive model to '{contrastive_model_path}'")
    else:
        try:
            contrastive_model.load_state_dict(torch.load(contrastive_model_path, map_location=device))
            print(f"Loaded pre-trained contrastive model from '{contrastive_model_path}'")
        except FileNotFoundError:
            print(f"Error: Pre-trained model not found at {contrastive_model_path}. Please train first or check path.")
            return

    # Downstream classification
    print("\n=== Phase 2: Downstream Classification ===")
    contrastive_model.eval()

    train_embedding_dataset = ContrastiveCodeDataset(X_train, y_train, tokenizer, MAX_TOKEN_LENGTH)
    test_embedding_dataset = ContrastiveCodeDataset(X_test, y_test, tokenizer, MAX_TOKEN_LENGTH)

    train_embedding_loader = DataLoader(
        train_embedding_dataset,
        batch_size=EMBEDDING_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    test_embedding_loader = DataLoader(
        test_embedding_dataset,
        batch_size=EMBEDDING_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("Generating embeddings for TRAIN set using the fine-tuned model...")
    X_train_embedded = get_embeddings(contrastive_model, train_embedding_loader, device)
    print(f"Generated TRAIN embeddings of shape: {X_train_embedded.shape}")

    print("Generating embeddings for TEST set using the fine-tuned model...")
    X_test_embedded = get_embeddings(contrastive_model, test_embedding_loader, device)
    print(f"Generated TEST embeddings of shape: {X_test_embedded.shape}")


    run_tsne = True
    if run_tsne:
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
            fig = px.scatter(
                tsne_df_train, x='TSNE-1', y='TSNE-2', color='Label',
                hover_data={'Label': True, 'Code Snippet': True},
                title='Interactive t-SNE of TRAIN Code Embeddings'
            )
            fig.write_html("interactive_tsne_plot_train_embeddings3.html")
            print("Interactive t-SNE plot saved to 'interactive_tsne_plot_train_embeddings3.html'")


    # Train Logistic Regression classifier on TRAIN embeddings
    print("Training downstream Logistic Regression classifier on TRAIN data...")
    clf = MultiOutputClassifier(
        LogisticRegression(max_iter=LOGISTIC_MAX_ITER, solver='liblinear', random_state=42, class_weight='balanced')
    )
    clf.fit(X_train_embedded, y_train) # Fit on TRAIN embeddings and TRAIN labels

    # Evaluate on the TEST data
    print("\n=== Evaluating on TEST Set ===")
    y_pred_test = clf.predict(X_test_embedded)

    print("\n=== Classification Report (on TEST Set) ===")
    print(classification_report(
        y_test, 
        y_pred_test,
        target_names=mlb.classes_,
        zero_division=0
    ))

    accuracy_test = accuracy_score(y_test, y_pred_test)
    print("Exact match accuracy (on TEST set):", accuracy_test)

    correct_predictions_test = np.all(y_test == y_pred_test, axis=1) # Exact match
    at_least_one_correct_test = np.sum(np.sum((y_test == 1) & (y_pred_test == 1), axis=1) > 0) / len(y_test)
    print(f"Accuracy (at least one correct label and no incorrect ones for that sample, more like Jaccard subset): {at_least_one_correct_test:.4f}")


    # Investigate some predictions TEST set
    print("\n--- Sample Predictions (from TEST set) ---")
    num_samples_to_show = min(10, len(X_test))
    for i in range(num_samples_to_show):
        # Reshape y_pred_test[i] to be 2D for inverse_transform
        predicted_labels_for_sample = mlb.inverse_transform(y_pred_test[i].reshape(1, -1))
        true_labels_for_sample = mlb.inverse_transform(y_test[i].reshape(1, -1))

        print(f"\nSample {i} (from test set):")
        print(f"  Code Snippet (first 80 chars): {X_test[i][:80]}...")
        print(f"  Predicted Labels: {predicted_labels_for_sample}")
        print(f"  True Labels:      {true_labels_for_sample}")

    return contrastive_model, clf, mlb


if __name__ == "__main__":
    main()