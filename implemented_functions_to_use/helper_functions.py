"""
A series of helper functions used throughout the course.

If a function gets defined once and could be used over and over, it'll go in here.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np

from torch import nn
import os
import zipfile
from pathlib import Path
import requests
import os

import torch
import torch.nn as nn
import torchmetrics
from torchmetrics.classification import MulticlassAUROC


# Plot linear data or training and test and predictions (optional)
def plot_predictions(
    train_data, train_labels, test_data, test_labels, predictions=None
):
    """
  Plots linear training data and test data and compares predictions.
  """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})


# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


# Plot loss curves of a model
def plot_loss_curves(model, results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title(f'Loss for {model}')
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title(f'Accuracy for {model}')
    plt.xlabel("Epochs")
    plt.legend()

    plt.savefig(fname=fr'graphs/loss_acc_{model}.png',format='png')

# Pred and plot image function from notebook 04
# See creation: https://www.learnpytorch.io/04_pytorch_custom_datasets/#113-putting-custom-image-prediction-together-building-a-function
from typing import List
import torchvision


def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str] = None,
    transform=None,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Makes a prediction on a target image with a trained model and plots the image.

    Args:
        model (torch.nn.Module): trained PyTorch image classification model.
        image_path (str): filepath to target image.
        class_names (List[str], optional): different class names for target image. Defaults to None.
        transform (_type_, optional): transform of target image. Defaults to None.
        device (torch.device, optional): target device to compute on. Defaults to "cuda" if torch.cuda.is_available() else "cpu".
    
    Returns:
        Matplotlib plot of target image and model prediction as title.

    Example usage:
        pred_and_plot_image(model=model,
                            image="some_image.jpeg",
                            class_names=["class_1", "class_2", "class_3"],
                            transform=torchvision.transforms.ToTensor(),
                            device=device)
    """

    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.0

    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))

    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(
        target_image.squeeze().permute(1, 2, 0)
    )  # make sure it's the right size for matplotlib
    if class_names :
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Path {image_path} | Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)

def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)

def plot_roc_auc(model: str, results, num_classes, class_names,trained):
    """
    Plots ROC curves and calculates AUC for multi-class classification.

    Args:
        results (dict): A dictionary containing 'y_true' and 'y_pred' as keys.
                        "train_loss": [...],
                        "train_acc": [...],
                        "test_loss": [...],
                        "test_acc": [...]},
                        'y_true',
                        'y_pred'
                        )
        num_classes (int): Number of classes.
        class_names (list): A list of class names.
    """
    if trained == False:
        # non trained
        y_true = results["y_true"][0]
        y_pred = results["y_pred"][0]
        txt = 'Not trained'
    else:
        # After training
        y_true = results["y_true"][-1]
        y_pred = results["y_pred"][-1]
        txt = 'trained'
    # Binarize the true labels for multi-class classification
    auroc = MulticlassAUROC(num_classes=num_classes)

    # Compute ROC-AUC for multi-class classification
    auroc_value = auroc(torch.tensor(y_pred), torch.tensor(y_true))

    
    # print(f'Multi-class AUROC is : {txt}: {auroc_value.item()}')

    # Compute ROC curve for each class
    fpr, tpr, thresholds_nt = torchmetrics.functional.roc(torch.tensor(y_pred), torch.tensor(y_true), num_classes=num_classes, task="multiclass")

    
    
    # Plot ROC curves for each class
    plt.figure(figsize=(10, 7))
    
     
    for i in range(num_classes):
        plt.plot(fpr[i].numpy(), tpr[i].numpy(), label=f'{class_names[i]} ROC curve')
        
        # plt.plot(thresholds[i].numpy(), label=f'{class_names[i]} Thresholds')
    
    # Plot the random classifier as a dotted line
    plt.plot([0, 1], [0, 1], label='Random Classifier (AUROC = 0.5)', linestyle='--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Tumor class ROC Curves {txt} for {model} | Multi-class AUROC is : {txt}: {auroc_value.item()}')
    plt.legend(loc="best")  
    
    plt.savefig(fname =fr'graphs/roc_auc_{model}_{txt}_model.png',  format='png')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(model: str,results, class_names,trained):
    """
    Plots a confusion matrix using true labels and predicted labels.

    Args:
    results (dict): A dictionary containing 'y_true' and 'y_pred' as keys.
                        "train_loss": [...],
                        "train_acc": [...],
                        "test_loss": [...],
                        "test_acc": [...]},
                        'y_true',
                        'y_pred'
                        )
    class_names: List of class names for labeling the confusion matrix.

    Returns:
    None (displays confusion matrix).
    """
    
    if trained == False:
        # non trained
        y_true = results["y_true"][0]
        y_pred = np.argmax(results["y_pred"][0], axis=1)
        txt = 'Not trained'
    else:
        # After training
        y_true = results["y_true"][-1]
        y_pred = np.argmax(results["y_pred"][-1], axis=1)
        txt = 'trained'
    
    # y_true = results["y_true"][0]
    # y_pred = np.argmax(results["y_pred"][0], axis=1)  
    
    # Compute the confusion matrix
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create a heatmap for the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)

    # Add labels and title
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Normalized Confusion Matrix for {txt} model : {model}')
    plt.tight_layout()
    plt.savefig(fname=fr'graphs/confusion_matrix_{model}_{txt}.png',format='png')
    plt.show()
