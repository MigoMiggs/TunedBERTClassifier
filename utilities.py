import torch
import torch.nn as nn
import numpy as np

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def evaluate(model, val_dataloader):
    model.eval()
    val_loss, val_accuracy = 0, 0
    nb_val_steps, nb_val_examples = 0, 0

    # a couple helpers
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for batch in val_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, labels = batch

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        #logits = outputs[0]
        loss = criterion(outputs, labels)  # Remove .float() conversion

        val_loss += loss.item()
        logits = outputs.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        val_accuracy += flat_accuracy(logits, label_ids)  # No need to calculate argmax for labels

        nb_val_examples += input_ids.size(0)
        nb_val_steps += 1

    avg_val_loss = val_loss / nb_val_steps
    avg_val_accuracy = val_accuracy / nb_val_examples

    print("Validation loss: {0:.2f}".format(avg_val_loss))
    print("Validation Accuracy: {0:.2f}".format(avg_val_accuracy * 100))
    return avg_val_accuracy, avg_val_loss