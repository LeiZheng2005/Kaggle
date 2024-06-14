import torch


from torch.optim import lr_scheduler
import torch.optim as optim
from torch import nn as nn
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
def loss_with_evaluate(model):
    optimizer = optim.Adam(model.parameters(), lr=0.00005)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    #Define loss functions
    Loss_Function = nn.CrossEntropyLoss()
    return optimizer,scheduler,Loss_Function


def Evaluate(class_names,model,device,dataloaders,train_losses,val_losses):
    # Define label names
    # class_names = ['malignant', 'normal', 'benign']
    label_names = class_names

    # Calculate classification report and confusion matrix on unseen test data
    y_true = []
    y_pred = []

    model_fineTuning = model
    model_fineTuning.eval()  # Set model to evaluation mode

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model_fineTuning(inputs)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Generate classification report
    classification_rep = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)

    # Generate confusion matrix
    confusion_mat = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix with label names
    plt.figure(figsize=(5, 3))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=label_names,
                yticklabels=label_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # calculate the percentage
    confusion_mtx_percent = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis] * 100

    f, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(confusion_mtx_percent, annot=True, linewidths=0.01, cmap="BuPu", linecolor="gray", fmt='.1f', ax=ax)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Percentage)")
    plt.show()
    # Convert the classification report to a heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(pd.DataFrame(classification_rep).iloc[:-1, :].T, annot=True, cmap='Blues',
                fmt='.2f')  # Simplify classification report
    plt.title('Classification Report Heatmap')
    plt.show()

    # Print the simplified classification report
    print("Simplified Classification Report:")
    print(pd.DataFrame(classification_rep).iloc[:-1, :])  # Display without support and avg/total rows
    # Plotting the training and validation losses
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
