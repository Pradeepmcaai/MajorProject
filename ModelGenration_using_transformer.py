import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
import re

# Step 1: Data Preparation (Assuming you have a pandas DataFrame)
import pandas as pd
df = pd.read_csv("twcs.csv")  # Load your custom data
df = df.iloc[:100000]
print(len(df.iloc[:1000]))
texts = df['text'].tolist()
df["inbound"] = df['inbound'].apply(lambda x : 1 if x==True else 0)
labels = df['inbound'].tolist()

# Step 2: Tokenization
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenized_data = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Step 3: Create a Dataset
dataset = TensorDataset(tokenized_data['input_ids'], tokenized_data['attention_mask'], torch.tensor(labels))

# Step 4: Model Selection
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)  # Adjust 'num_labels' as needed

# Step 5: Fine-Tuning Parameters
optimizer = AdamW(model.parameters(), lr=1e-5)
batch_size = 32
epochs = 3

# Step 6: Split Data
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Step 7: Data Loaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Step 8: Loss Function
loss_fn = torch.nn.CrossEntropyLoss()  # Use CrossEntropyLoss for binary classification

# Step 9: Training Loop
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = loss_fn(logits, labels)  # Use CrossEntropyLoss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Average Training Loss: {total_loss / len(train_dataloader)}")

model.save_pretrained("Twitter_Model_777_1")
