# Install required libraries
import warnings
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import emoji
import os
import zipfile
import urllib.request
import numpy as np
import re
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import pandas as pd
!pip install torch torchvision torchaudio scikit-learn pandas emoji matplotlib seaborn

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# -------------------------
# Enhanced Data Loading and Preprocessing
# -------------------------


class TextPreprocessor:
    def __init__(self):
        self.url_pattern = re.compile(r"http\S+|www\S+")
        self.email_pattern = re.compile(r"\S+@\S+")
        self.number_pattern = re.compile(r"\d+")
        self.punctuation_pattern = re.compile(r"([!?.])")
        self.repeat_pattern = re.compile(r'(.)\1{2,}')
        self.space_pattern = re.compile(r'\s+')

    def clean_text(self, text):
        """Enhanced text cleaning with better spam indicators"""
        text = str(text).lower()

        # Replace URLs, emails, numbers with specific tokens
        text = self.url_pattern.sub("<URL>", text)
        text = self.email_pattern.sub("<EMAIL>", text)
        text = self.number_pattern.sub("<NUM>", text)

        # Separate punctuation for better tokenization
        text = self.punctuation_pattern.sub(r" \1 ", text)

        # Normalize repeated characters (sooo -> so)
        text = self.repeat_pattern.sub(r'\1\1', text)

        # Convert emojis to text tokens
        text = emoji.demojize(text)

        # Handle common spam patterns
        text = self._enhance_spam_features(text)

        # Remove extra spaces
        text = self.space_pattern.sub(' ', text).strip()

        return text

    def _enhance_spam_features(self, text):
        """Add specific tokens for common spam patterns"""
        spam_indicators = {
            r'\b(urgent|immediately|asap|right away)\b': '<URGENT>',
            r'\b(free|prize|winner|won|award)\b': '<FREE>',
            r'\b(cash|money|price|cost|dollar)\b': '<MONEY>',
            r'\b(click|call|text|reply)\b': '<ACTION>',
            r'\b(guarantee|guaranteed)\b': '<GUARANTEE>',
            r'\b(limited|offer|only|special)\b': '<OFFER>',
        }

        for pattern, replacement in spam_indicators.items():
            text = re.sub(pattern, replacement, text)

        return text

# Load and prepare data


def load_and_prepare_data(file_path="spam.csv"):
    """Load and preprocess the dataset with comprehensive analysis"""
    df = pd.read_csv(file_path)

    print("Dataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Class distribution:\n{df['type'].value_counts()}")
    print(f"Spam percentage: {(df['type'] == 'spam').mean():.2%}")

    # Initialize preprocessor
    preprocessor = TextPreprocessor()

    # Clean texts
    df['clean_text'] = df['text'].apply(preprocessor.clean_text)
    df['clean_text'] = df['clean_text'].fillna("").astype(str)

    # Analyze text lengths
    df['text_length'] = df['clean_text'].apply(len)
    df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))

    print(f"\nText length stats:")
    print(f"Average length: {df['text_length'].mean():.1f} chars")
    print(f"Average words: {df['word_count'].mean():.1f}")

    return df, preprocessor

# -------------------------
# Vocabulary and Tokenization
# -------------------------


class Vocabulary:
    def __init__(self, min_freq=2):
        self.word2idx = {}
        self.idx2word = {}
        self.min_freq = min_freq
        self.vocab_size = 0

    def build_vocab(self, texts):
        """Build vocabulary from texts with frequency filtering"""
        all_words = ' '.join(texts).split()
        word_counts = Counter(all_words)

        # Filter words by minimum frequency
        filtered_words = {word: count for word, count in word_counts.items()
                          if count >= self.min_freq}

        # Reserve indices: 0=PAD, 1=UNK, 2=SOS, 3=EOS
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}

        # Add words to vocabulary
        for i, word in enumerate(filtered_words.keys(), start=4):
            self.word2idx[word] = i

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Most common words: {word_counts.most_common(10)}")

    def encode_text(self, text, max_len=100):
        """Encode text with SOS and EOS tokens"""
        tokens = [self.word2idx.get('<SOS>', 2)]  # Start with SOS

        # Add word tokens
        for w in str(text).split():
            tokens.append(self.word2idx.get(w, 1))  # UNK for unknown words

        tokens.append(self.word2idx.get('<EOS>', 3))  # End with EOS

        # Pad or truncate
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len-1] + [self.word2idx.get('<EOS>', 3)]

        return tokens

# -------------------------
# PyTorch Dataset
# -------------------------


class SMSDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# -------------------------
# Enhanced LSTM Model
# -------------------------


class LSTMSpamClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, embedding_matrix, hidden_dim=128,
                 output_dim=2, n_layers=2, dropout=0.3, bidirectional=True):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=False, padding_idx=0
        )

        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional
        )

        self.dropout = nn.Dropout(dropout)

        # Adjust linear input size based on bidirectional
        linear_input_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(linear_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(x)

        # Use the last hidden state for classification
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        out = self.dropout(hidden)
        out = self.classifier(out)
        return out

# -------------------------
# Training and Evaluation Utilities
# -------------------------


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for texts, labels in self.train_loader:
            texts, labels = texts.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(texts)
            loss = self.criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for texts, labels in self.val_loader:
                texts, labels = texts.to(self.device), labels.to(self.device)
                outputs = self.model(texts)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        return total_loss / len(self.val_loader), accuracy

    def train(self, epochs, patience=5):
        best_val_acc = 0
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss, val_acc = self.evaluate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            print(f'Epoch {epoch+1}/{epochs}:')
            print(
                f'  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                print(f'  ↳ New best model saved!')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        return best_val_acc


def plot_training_history(trainer):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(trainer.train_losses, label='Train Loss')
    ax1.plot(trainer.val_losses, label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(trainer.val_accuracies, label='Val Accuracy', color='green')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def evaluate_model(model, test_loader, device):
    """Comprehensive model evaluation"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification report
    print("Classification Report:")
    print(classification_report(all_labels,
          all_preds, target_names=['ham', 'spam']))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['ham', 'spam'],
                yticklabels=['ham', 'spam'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    return all_preds, all_labels

# -------------------------
# GloVe Embeddings Loader
# -------------------------


def load_glove_embeddings(embedding_dim=100):
    """Load and return GloVe embeddings"""
    glove_url = "http://nlp.stanford.edu/data/glove.6B.zip"
    glove_path = "glove.6B.zip"
    glove_dir = "glove"

    if not os.path.exists(glove_path):
        print("Downloading GloVe embeddings...")
        urllib.request.urlretrieve(glove_url, glove_path)

    if not os.path.exists(glove_dir):
        with zipfile.ZipFile(glove_path, 'r') as zip_ref:
            zip_ref.extractall(glove_dir)

    glove_file = f"glove/glove.6B.{embedding_dim}d.txt"
    embeddings_index = {}

    with open(glove_file, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector

    print(f"Loaded {len(embeddings_index)} word vectors")
    return embeddings_index


def create_embedding_matrix(vocab, embeddings_index, embedding_dim=100):
    """Create embedding matrix from vocabulary and GloVe"""
    embedding_matrix = np.zeros((len(vocab.word2idx), embedding_dim))

    for word, idx in vocab.word2idx.items():
        if word in embeddings_index:
            embedding_matrix[idx] = embeddings_index[word]
        elif word.lower() in embeddings_index:
            embedding_matrix[idx] = embeddings_index[word.lower()]
        else:
            # Initialize with small random numbers for unknown words
            embedding_matrix[idx] = np.random.normal(0, 0.1, embedding_dim)

    # Ensure padding is zeros
    embedding_matrix[0] = np.zeros(embedding_dim)

    return torch.tensor(embedding_matrix, dtype=torch.float)

# -------------------------
# Main Execution
# -------------------------


def main():
    # Configuration
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    BATCH_SIZE = 32
    EPOCHS = 50
    MAX_LEN = 100
    LEARNING_RATE = 0.001

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Step 1: Load and preprocess data
    print("Step 1: Loading and preprocessing data...")
    df, preprocessor = load_and_prepare_data("spam.csv")

    # Encode labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['type'])

    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=42, stratify=train_labels
    )

    print(f"\nData splits:")
    print(
        f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

    # Step 2: Build vocabulary
    print("\nStep 2: Building vocabulary...")
    vocab = Vocabulary(min_freq=2)
    vocab.build_vocab(train_texts)

    # Step 3: Encode texts
    print("\nStep 3: Encoding texts...")
    train_encoded = [vocab.encode_text(text, MAX_LEN) for text in train_texts]
    val_encoded = [vocab.encode_text(text, MAX_LEN) for text in val_texts]
    test_encoded = [vocab.encode_text(text, MAX_LEN) for text in test_texts]

    # Step 4: Create datasets and dataloaders
    print("\nStep 4: Creating datasets...")
    train_dataset = SMSDataset(train_encoded, train_labels.values)
    val_dataset = SMSDataset(val_encoded, val_labels.values)
    test_dataset = SMSDataset(test_encoded, test_labels.values)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Step 5: Load GloVe embeddings
    print("\nStep 5: Loading GloVe embeddings...")
    embeddings_index = load_glove_embeddings(EMBEDDING_DIM)
    embedding_matrix = create_embedding_matrix(
        vocab, embeddings_index, EMBEDDING_DIM)

    # Step 6: Initialize model
    print("\nStep 6: Initializing model...")
    model = LSTMSpamClassifier(
        vocab_size=len(vocab.word2idx),
        embed_dim=EMBEDDING_DIM,
        embedding_matrix=embedding_matrix,
        hidden_dim=HIDDEN_DIM,
        output_dim=2,
        n_layers=2,
        dropout=0.3
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Step 7: Train model
    print("\nStep 7: Training model...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    trainer = Trainer(model, train_loader, val_loader,
                      criterion, optimizer, device)
    best_val_acc = trainer.train(EPOCHS, patience=7)

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    # Plot training history
    plot_training_history(trainer)

    # Step 8: Evaluate model
    print("\nStep 8: Evaluating model...")
    all_preds, all_labels = evaluate_model(model, test_loader, device)

    # Step 9: Interactive prediction
    print("\nStep 9: Interactive prediction...")

    def predict_spam(text):
        model.eval()
        text_clean = preprocessor.clean_text(text)
        text_encoded = vocab.encode_text(text_clean, MAX_LEN)
        tensor = torch.tensor([text_encoded], dtype=torch.long).to(device)

        with torch.no_grad():
            output = model(tensor)
            probas = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()
            confidence = probas[0][pred].item()

        return "spam" if pred == 1 else "ham", confidence

    # Test with known spam patterns
    test_messages = [
        "URGENT: Your Bank account ending **1234** has been locked due to suspicious activity. Verify now: http://verify-secure.example.com to avoid permanent suspension.",
        "Missed delivery: Parcel ID 8A7F has failed to deliver. Re-schedule instantly at http://track.example.com/8A7F or reply RESCH. Fee $2.",
        "Your verification code is 847120. Enter it at http://secure-login.example.com to complete sign-in.",
        "Hey, are we still meeting for lunch tomorrow?",
        "Congratulations! You've won a $1000 Walmart gift card! Click here to claim: http://win.example.com"
    ]

    print("\nTesting with sample messages:")
    for msg in test_messages:
        prediction, confidence = predict_spam(msg)
        print(f"Message: {msg[:80]}...")
        print(f"Prediction: {prediction} (confidence: {confidence:.4f})\n")

    # Interactive loop
    print("Interactive prediction mode (type 'exit' to quit):")
    while True:
        try:
            user_input = input("\nEnter a message: ")
            if user_input.lower() == 'exit':
                break
            prediction, confidence = predict_spam(user_input)
            print(f"Prediction: {prediction} (confidence: {confidence:.4f})")
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
