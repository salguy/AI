import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

csv_path = "./dataset/full_prompt_dataset.csv"
df_loaded = pd.read_csv(csv_path)

# 하이퍼파라미터 설정
MAX_LEN = 64
BATCH_SIZE = 16
EMBED_DIM = 128
HIDDEN_DIM = 256
NUM_CLASSES = len(df_loaded["label"].unique())
EPOCHS = 100

# 토크나이저 로딩
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Dataset 정의
class PromptDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.texts = dataframe["text"].tolist()
        self.labels = dataframe["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_ids, attention_mask, label

# LSTM 분류기
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])

df_shuffled = df_loaded.sample(frac=1, random_state=42).reset_index(drop=True)
split_idx = int(0.8 * len(df_shuffled))
train_df = df_shuffled[:split_idx]
val_df = df_shuffled[split_idx:]

train_dataset = PromptDataset(train_df, tokenizer, MAX_LEN)
val_dataset = PromptDataset(val_df, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 모델 초기화
model = LSTMClassifier(
    vocab_size=tokenizer.vocab_size,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=NUM_CLASSES,
    pad_idx=tokenizer.pad_token_id
).to(DEVICE)

# 옵티마이저와 손실 함수
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 학습 루프
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for input_ids, attention_mask, labels in train_loader:
        input_ids, attention_mask, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader:
            input_ids, attention_mask, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {accuracy:.2%}")

torch.save(model.state_dict(), './weight/model.pth')