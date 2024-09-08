import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm
import time
import sys

# Load datasets using Hugging Face datasets library
reddit_dataset = load_dataset('reddit', split='train[:1%]')  # Using 1% for testing purposes
openwebtext_dataset = load_dataset('openwebtext', split='train[:1%]')  # Using 1% for testing purposes


# Custom Dataset Class for Hugging Face datasets
class ChatDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # Set the pad_token to eos_token (End Of Sequence)
        self.tokenizer.pad_token = self.tokenizer.eos_token  #Use eos_token as pad_token
        # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})  #Define a new pad token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Adjust based on dataset structure
        text = self.data[idx]['content'] if 'content' in self.data[idx] else self.data[idx]['text']
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors='pt',
            padding='max_length',
            max_length=50,
            truncation=True
        )
        return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze()


# Load datasets
reddit_chat = ChatDataset(reddit_dataset)
open_web_text_chat = ChatDataset(openwebtext_dataset)

# Combine datasets
combined_dataset = torch.utils.data.ConcatDataset([reddit_chat, open_web_text_chat])
data_loader = DataLoader(combined_dataset, batch_size=8, shuffle=True, num_workers=4)

# Model Initialization
model = GPT2LMHeadModel.from_pretrained('gpt2')
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# Training Loop with Gradient Accumulation
model.train()
accumulation_steps = 4  # Number of steps to accumulate gradients
num_epochs = 3  # Number of epochs

for epoch in range(num_epochs):
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")
    for i, batch in progress_bar:
        input_ids, attention_mask = batch
        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss / accumulation_steps

        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()

        # Update progress bar
        progress_bar.set_postfix(loss=loss.item() * accumulation_steps, accuracy=100 - loss.item() * accumulation_steps)

    print(f'\nEpoch {epoch + 1} completed.')

# Save the model
torch.save(model.state_dict(), 'chat_model.pth')