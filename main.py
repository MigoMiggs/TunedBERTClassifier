import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from transformers import BertTokenizer
from transformers import BertModel
from torch.utils.data import DataLoader, TensorDataset
from BertClassifier import BertClassifier
import utilities
import Teso

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

######################## set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############ Create some sample data
# Tokenize the text
status_1 = "We're making good progress on the project, but we're starting to run into some resource constraints."
status_2 = "The project is moving along as planned, and we don't anticipate any major risks in the near future."
status_3 = "We're behind schedule on the project, and we're facing some significant time risks."

encoded_status_1 = tokenizer(status_1, padding='max_length', truncation=True, max_length=32)
encoded_status_2 = tokenizer(status_2, padding='max_length', truncation=True, max_length=32)
encoded_status_3 = tokenizer(status_3, padding='max_length', truncation=True, max_length=32)

# Convert the tokenized input to input IDs and attention masks
input_ids_1 = torch.tensor(encoded_status_1['input_ids']).unsqueeze(0)
attention_mask_1 = torch.tensor(encoded_status_1['attention_mask']).unsqueeze(0)
label_1 = torch.tensor([1])

input_ids_2 = torch.tensor(encoded_status_2['input_ids']).unsqueeze(0)
attention_mask_2 = torch.tensor(encoded_status_2['attention_mask']).unsqueeze(0)
label_2 = torch.tensor([0])

input_ids_3 = torch.tensor(encoded_status_3['input_ids']).unsqueeze(0)
attention_mask_3 = torch.tensor(encoded_status_3['attention_mask']).unsqueeze(0)
label_3 = torch.tensor([2])

# Create a batch of input
train_input_ids = torch.cat([input_ids_1, input_ids_2, input_ids_3], dim=0)
train_attention_masks = torch.cat([attention_mask_1, attention_mask_2, attention_mask_3], dim=0)
train_labels = torch.cat([label_1, label_2, label_3], dim=0)

############# Setup training data

# Create PyTorch tensors for input and output data
input_ids = train_input_ids
attention_masks = train_attention_masks
labels = train_labels

# Create a TensorDataset from input and output data
dataset = TensorDataset(input_ids, attention_masks, labels)

# Create a DataLoader for the training data
batch_size = 3
trainloader = DataLoader(dataset, batch_size, True)

########################## instantiate the Classifier
num_classes = 3
model = BertClassifier(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)
num_epochs = 50

######################### train the model for fine tuning

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        input_ids, input_mask, labels = data
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=input_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('Epoch [%d], loss: %.3f' % (epoch+1, running_loss/len(trainloader)))


####################################### evaluate performance

# Evaluate data for one epoch
utilities.evaluate(model, trainloader)
