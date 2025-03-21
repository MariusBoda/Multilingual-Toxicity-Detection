import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from nltk.tokenize import wordpunct_tokenize
import nltk
from collections import Counter
from tqdm import tqdm
from metaflow import FlowSpec, step
from functools import partial
from multi_tokenizer import MultiTokenizer, PretrainedTokenizers

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
    def __init__(self, df, vocab):
        self.labels = torch.tensor(df["label"].values, dtype=torch.long)
        self.token_indices = [torch.tensor(indices, dtype=torch.long) for indices in df["token_indices"].values]
        self.max_length = max(len(indices) for indices in self.token_indices) if self.token_indices else 0
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

# Metaflow Pipeline

class TransformerFlow(FlowSpec):

    @step
    def start(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}", flush=True)
        self.skip_training = False
        self.train_df = pd.read_csv('train.tsv', sep='\t', quoting=3)
        
        # Initialize MultiTokenizer with specified language tokenizers
        self.lang_tokenizers = [
            PretrainedTokenizers.ENGLISH,
            PretrainedTokenizers.FINNISH,
            PretrainedTokenizers.GERMAN,
        ]
        self.tokenizer = MultiTokenizer(self.lang_tokenizers, split_text=True)
        self.next(self.preprocess)

    @step
    def preprocess(self):
        # Tokenize text using MultiTokenizer
        self.train_df['tokens'] = self.train_df['text'].apply(
            lambda x: self.tokenizer.encode(x)[1]  # Extract tokens from encode output
        )
        self.next(self.build_vocab)

    @step
    def build_vocab(self):
        # Use the tokenizer's predefined vocabulary
        self.vocab = self.tokenizer.get_vocab()
        
        # Convert tokens to indices using the tokenizer's vocab
        def tokens_to_indices(tokens):
            return [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        self.train_df['token_indices'] = self.train_df['tokens'].apply(tokens_to_indices)
        self.next(self.prepare)

    @step
    def prepare(self):
        class_counts = self.train_df['label'].value_counts()
        total_samples = len(self.train_df)
        class_weights = torch.tensor(
            [total_samples / (len(class_counts) * count for count in class_counts)]
        )
        sample_weights = self.train_df['label'].map(lambda label: class_weights[label]).tolist()
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        self.dataset_train = TransformerDataset(self.train_df, self.vocab)
        self.dataloader_train = DataLoader(
            self.dataset_train,
            batch_size=64,
            collate_fn=partial(collate_fn_transformer, vocab=self.vocab),
            sampler=sampler,
            num_workers=4
        )
        self.next(self.train)

    @step
    def train(self):
        vocab_size = len(self.vocab)
        embedding_dim = 100
        encoder = Encoder_Only_Transformer(vocab_size, embedding_dim).to(self.device)
        EPOCHS = 5
        num_batches = 30
        optimizer = optim.Adam(encoder.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        if not self.skip_training:
            for epoch in range(EPOCHS):
                avg_loss = 0
                for i, (embeddings, labels, mask) in enumerate(self.dataloader_train):
                    if i >= num_batches:
                        break
                    optimizer.zero_grad()
                    embeddings = embeddings.to(self.device)
                    labels = labels.to(self.device)
                    mask = mask.to(self.device)
                    logits = encoder(embeddings, mask)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                    avg_loss += loss.item()
                    if i % 10 == 0:
                        print(f"Epoch {epoch+1} Batch {i}, Loss: {loss.item()}", flush=True)
                print(f"Epoch: {epoch+1}, Loss: {avg_loss / num_batches}")
            torch.save(encoder.state_dict(), 'model.pth')
        self.next(self.test)

    @step
    def test(self):
        vocab_size = len(self.vocab)
        embedding_dim = 100
        test_model = Encoder_Only_Transformer(vocab_size, embedding_dim).to(self.device)
        test_model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
        test_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for embeddings, labels, mask in tqdm(self.dataloader_dev):
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)
                mask = mask.to(self.device)
                logits = test_model(embeddings, mask)
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy}")
        self.next(self.end)

    @step
    def end(self):
        print("Pipeline completed")

if __name__ == '__main__':
    TransformerFlow()
