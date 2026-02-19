import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from utils.representation_loading import batch_representation_loader
from tqdm import tqdm
import numpy as np

def test (model, test_loader, names_of_categories, loss_fn, device, representation_config):
    model.eval()
    test_losses = []
    epoch_labels = []
    epoch_preds = []
    accuracy_per_class = {}
    for batch in tqdm(test_loader):
        input_names, targets = batch['file_names'], batch['labels'].long().to(device)
        inputs = batch_representation_loader(input_names, representation_config["layer"], representation_config["path"], representation_config["model"], representation_config["data_set"]).to(device)
        output = model(inputs)
        output =  output.logits
        loss = loss_fn(output, targets)
        
        preds = torch.argmax(torch.softmax(output, dim=-1), dim=-1).to('cpu').detach().numpy().tolist()
        epoch_preds = epoch_preds + preds
        labels = targets.to('cpu').detach().numpy().tolist()
        epoch_labels = epoch_labels + labels
        test_losses.append(loss.data.item())
        inputs, targets, output = inputs.to('cpu'), targets.to('cpu'), output.to('cpu') # Return tensors to cpu after each iteration


    #######################################################################################################
    unique_classes = set(epoch_labels)
    for class_label in unique_classes:
        indices = [i for i, y in enumerate(epoch_labels) if y == class_label]
        class_accuracy = accuracy_score([epoch_labels[i] for i in indices], [epoch_preds[i] for i in indices])
        accuracy_per_class[class_label] = class_accuracy
    #######################################################################################################


    test_loss = np.mean(test_losses)
    w_acc = accuracy_score(epoch_labels, epoch_preds)
    u_acc = balanced_accuracy_score(epoch_labels, epoch_preds)
    precision = precision_score(epoch_labels, epoch_preds, average='weighted')
    recall = recall_score(epoch_labels, epoch_preds, average='weighted')
    f1 = f1_score(epoch_labels, epoch_preds, average='weighted')
    cm = confusion_matrix(epoch_labels, epoch_preds)
    cm_percent = [a/np.sum(a) for a in cm]
    annotations = []
    for axis, arr in zip(cm_percent, cm):
        annot = [f"{x :.2f}%\n{y}" for x,y in zip(axis, arr)]
        annotations.append(annot)

    sns.heatmap(cm_percent, annot=annotations, fmt='', cmap='Greens', 
                xticklabels=names_of_categories, 
                yticklabels=names_of_categories)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    print(f'Testing Loss:{test_loss} | UnWeighted_Accuracy:{u_acc} | Weighted_Accuracy:{w_acc} | Precision:{precision} | Recall:{recall} | f1:{f1}')

    print("Accuracy per class:")
    for class_label, accuracy in accuracy_per_class.items():
        print(f"Class {class_label}: {accuracy}")

    return {
        "loss": test_loss,
        "w_accuracy": w_acc,
        "u_accuracy": u_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cm": cm,
        "accuracy_per_class": accuracy_per_class
    }

class CrossValidationTestTracker:
    def __init__(self):
        print("Tracker is Initialized with empty memory")
        self.loss = []
        self.accuracy_per_class = []
        self.w_accuracy = []
        self.u_accuracy = []
        self.precision = []
        self.recall = []
        self.f1 = []
        self.cm = []

    def append(self, metrics: dict):

        for metric, value in metrics.items():
            if not hasattr(self, metric):
                raise KeyError(f"Unknown metric '{metric}'")
            lst = getattr(self, metric)
            lst.append(value)