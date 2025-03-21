import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from nltk.tokenize import wordpunct_tokenize
import nltk
from collections import Counter
from tqdm import tqdm
from functools import partial
import os
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
nltk.download('punkt', quiet=True)

# Model Components

"""Collate function to pad sequences to the same length"""
def collate_fn_transformer(batch, vocab):
    indices, labels = zip(*batch)
    padded_indices = torch.stack(indices)
    labels = torch.stack(labels)
    mask = (padded_indices != vocab['<PAD>']).long()
    return padded_indices, labels, mask

"""Dataset class for the Transformer model"""
class TransformerDataset(Dataset):
    def __init__(self, df, vocab, max_length=None):
        self.labels = torch.tensor(df["label"].values, dtype=torch.long)
        self.token_indices = [torch.tensor(indices, dtype=torch.long) for indices in df["token_indices"].values]
        self.max_length = max_length if max_length is not None else max(len(indices) for indices in self.token_indices)
        self.vocab = vocab

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        indices = self.token_indices[idx]
        if len(indices) < self.max_length:
            padding = torch.zeros(self.max_length - len(indices), dtype=torch.long)
            padded_indices = torch.cat([indices, padding])
        else:
            padded_indices = indices[:self.max_length]
        return padded_indices, self.labels[idx]

"""Encoder block for the Transformer model"""
class Encoder(nn.Module):
    def __init__(self, embedding_dim, n_heads=10, hidden_size=64, dropout=0.1):
        super().__init__()
        if embedding_dim % n_heads != 0:
            raise ValueError("embedding_dim must be divisible by n_heads")
        self.attention = nn.MultiheadAttention(embedding_dim, n_heads, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embedding_dim)
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        key_padding_mask = (mask == 0)
        attention, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        attention_skip = self.dropout(attention) + x
        x = self.norm(attention_skip)
        out = self.feedforward(x)
        out = self.norm(out + x)
        return out

"""Encoder-only Transformer model"""
class Encoder_Only_Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_heads=10, hidden_size=64, dropout=0.1, num_classes=2, num_blocks=3):
        super().__init__()
        # Use padding_idx=0 because <PAD> is set to 0 in our vocabulary
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.blocks = nn.ModuleList([
            Encoder(embedding_dim, n_heads, hidden_size, dropout) for _ in range(num_blocks)
        ])
        
        self.feedforward_classification = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x, mask):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x, mask)
        x = x.mean(dim=1)
        out = self.feedforward_classification(x)
        return out

class TransformerPipeline:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.skip_training = False
        print(f"Using device: {self.device}")
        
        self.train_df = None
        self.vocab = None
        self.dataset_train = None
        self.dataloader_train = None
        self.dataloader_dev = None

    def run(self):
        self.load_data()
        self.preprocess()
        self.build_vocab()
        self.prepare_datasets()
        self.train_model()
        self.test_model()
        self.predictions()
        
    def load_data(self):
        """Load the training data"""
        self.train_df = pd.read_csv('train.tsv', sep='\t', quoting=3)
        self.dev_df = pd.read_csv('dev.tsv', sep='\t', quoting=3)
        self.test_df = pd.read_csv('test.tsv', sep='\t', quoting=3)

    def preprocess(self):
        """Tokenize text data"""
        self.train_df['tokens'] = self.train_df['text'].apply(
            lambda x: [token.lower() for token in wordpunct_tokenize(x)]
        )
        if hasattr(self, 'dev_df'):
            self.dev_df['tokens'] = self.dev_df['text'].apply(
                lambda x: [token.lower() for token in wordpunct_tokenize(x)]
        )

    def build_vocab(self):
        """Build vocabulary from training data"""
        all_tokens = [token for tokens in self.train_df['tokens'] for token in tokens]
        self.vocab = {token: idx+2 for idx, (token, _) in enumerate(Counter(all_tokens).items())}
        self.vocab['<PAD>'] = 0
        self.vocab['<UNK>'] = 1

        def tokens_to_indices(tokens):
            return [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]

        self.train_df['token_indices'] = self.train_df['tokens'].apply(tokens_to_indices)
        if hasattr(self, 'dev_df'):
            self.dev_df['token_indices'] = self.dev_df['tokens'].apply(tokens_to_indices)

    def prepare_datasets(self):
        """Prepare DataLoaders with weighted sampling and consistent max_length"""
        self.dataset_train = TransformerDataset(self.train_df, self.vocab)
        max_length = self.dataset_train.max_length

        class_counts = self.train_df['label'].value_counts()
        total_samples = len(self.train_df)
        class_weights = torch.tensor(
            [total_samples / (len(class_counts) * count) for count in class_counts]
        )
        sample_weights = self.train_df['label'].map(lambda label: class_weights[label]).tolist()
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        self.dataloader_train = DataLoader(
            self.dataset_train,
            batch_size=128,
            collate_fn=partial(collate_fn_transformer, vocab=self.vocab),
            sampler=sampler,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True
        )

        if hasattr(self, 'dev_df'):
            self.dataset_dev = TransformerDataset(self.dev_df, self.vocab, max_length=max_length)
            self.dataloader_dev = DataLoader(
                self.dataset_dev, 
                batch_size=128, 
                collate_fn=partial(collate_fn_transformer, vocab=self.vocab),
                num_workers=8,
                pin_memory=True,
                persistent_workers=True
            )

    def train_model(self):
        """Training loop with early stopping and dynamic LR"""
        vocab_size = len(self.vocab)
        embedding_dim = 100  # Consistent embedding dimension
        encoder = Encoder_Only_Transformer(vocab_size, embedding_dim).to(self.device)
        
        # Hyperparameters
        EPOCHS = 100
        patience = 3  # Early stopping patience
        best_loss = np.inf
        patience_counter = 0
        
        # Optimizer with weight decay
        optimizer = optim.Adam(encoder.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
        criterion = nn.CrossEntropyLoss()

        if 'model.pth' in os.listdir():
            self.skip_training = True
        if not self.skip_training:
            for epoch in range(EPOCHS):
                encoder.train()
                avg_loss = 0
                for i, (embeddings, labels, mask) in tqdm(enumerate(self.dataloader_train)):
                    optimizer.zero_grad()
                    embeddings = embeddings.to(self.device)
                    labels = labels.to(self.device)
                    mask = mask.to(self.device)
                    logits = encoder(embeddings, mask)
                    loss = criterion(logits, labels)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    avg_loss += loss.item()

                # Validation phase
                encoder.eval()
                val_loss = 0
                with torch.no_grad():
                    for embeddings, labels, mask in self.dataloader_dev:
                        embeddings = embeddings.to(self.device)
                        labels = labels.to(self.device)
                        mask = mask.to(self.device)
                        logits = encoder(embeddings, mask)
                        val_loss += criterion(logits, labels).item()
                
                val_loss /= len(self.dataloader_dev)
                scheduler.step(val_loss)  # Update learning rate
                
                print(f"Epoch: {epoch+1}, Train Loss: {avg_loss/len(self.dataloader_train)}, Val Loss: {val_loss}")
                
                # Early stopping
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    torch.save(encoder.state_dict(), 'model.pth')  # Save best model
                else:
                    patience_counter +=1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

    def test_model(self):
        if not hasattr(self, 'dataloader_dev'):
            return

        vocab_size = len(self.vocab)
        embedding_dim = 100  # Match training dimension
        self.test_model = Encoder_Only_Transformer(vocab_size, embedding_dim).to(self.device)
        self.test_model.load_state_dict(torch.load('model.pth', map_location=self.device))
        self.test_model.eval()
        
        correct = 0
        total = 0
        with torch.no_grad():
            for embeddings, labels, mask in tqdm(self.dataloader_dev):
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)
                mask = mask.to(self.device)
                logits = self.test_model(embeddings, mask)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        accuracy = correct / total
        print(f"Validation Accuracy: {accuracy}")

    def predictions(self):
        max_length = self.dataset_train.max_length
        def predict_sentiment(model, sentence, vocab, max_length, device):
            model.eval()
            
            if not sentence.strip():
                return -1

            # Tokenize and convert to indices
            tokens = [token.lower() for token in wordpunct_tokenize(sentence)]
            token_indices = [vocab.get(token, vocab['<UNK>']) for token in tokens]
            
            # Padding/truncating
            if len(token_indices) < max_length:
                token_indices += [vocab['<PAD>']] * (max_length - len(token_indices))
            else:
                token_indices = token_indices[:max_length]

            # Create tensors and move to device
            input_tensor = torch.tensor(token_indices, dtype=torch.long).unsqueeze(0).to(device)
            mask = (input_tensor != vocab['<PAD>']).long().to(device)

            with torch.no_grad():
                output = model(input_tensor, mask)
                prediction = torch.argmax(output, dim=1).item()

            return prediction

        # Load test data
        test_file = 'test.tsv'
        test_data = pd.read_csv(test_file, sep='\t', header=0, quoting=3)
        output_file = 'predictions_new_model.tsv'

        # Load model
        vocab_size = len(self.vocab)
        embedding_dim = 100
        self.test_model = Encoder_Only_Transformer(vocab_size, embedding_dim).to(self.device)
        self.test_model.load_state_dict(torch.load('model.pth', map_location=self.device))
        
        # Get max length from training data
        max_length = self.dataset_train.max_length

        with open(output_file, 'w') as f:
            f.write('id\tpredicted\n')

            for idx, row in test_data.iterrows():
                sentence = row['text']
                
                try:
                    prediction = predict_sentiment(
                        self.test_model, 
                        sentence, 
                        self.vocab, 
                        max_length,
                        self.device
                    )
                    f.write(f"{row['id']}\t{prediction}\n")
                except Exception as e:
                    print(f"Error for row {row['id']}: {e}")
                    f.write(f"{row['id']}\terror\n")

if __name__ == '__main__':
    pipeline = TransformerPipeline()
    pipeline.run()