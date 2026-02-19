from utils.representation_loading import batch_representation_loader
from utils.plotting_utils import draw_line_plot
import time
from tqdm import tqdm
import torch
import torch.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
import os

def train(model,
          saving_path,
          model_name,
          optimizer,
          loader,
          epochs,
          metric_for_save_best,
          loss_fn,
          device,
          scheduler,
          representation_config):
  
  train_loader, val_loader = (loader["TrainLoader"], loader["TestLoader"])
  training_losses = []
  valid_losses = []
  training_accuracies = []
  valid_accuracies = []
  durations = []
  best_val_loss = float('inf')
  best_val_accuracy = 0.0

  for epoch in range(epochs):
    training_loss = []
    valid_loss = []
    start_time = time.time()
    train_epoch_labels = []
    train_epoch_preds = []

    model.train()
    for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False): # Each batch, One Iteration
      optimizer.zero_grad()
      input_names, targets = batch['file_names'], batch['labels'].long().to(device)
      inputs = batch_representation_loader(input_names, representation_config["layer"], representation_config["path"], representation_config["model"], representation_config["data_set"]).to(device)
      output = model(inputs)
      output =  output.logits

      loss = loss_fn(output, targets)
      loss.backward()
      optimizer.step()

      preds = torch.argmax(torch.softmax(output, dim=-1), dim=-1).to('cpu').detach().numpy().tolist()
      train_epoch_preds = train_epoch_preds + preds 
      labels = targets.to('cpu').detach().numpy().tolist()
      train_epoch_labels = train_epoch_labels + labels
      inputs, targets, output = inputs.to('cpu'), targets.to('cpu'), output.to('cpu') # Return tensors to cpu after each iteration

      training_loss.append(loss.data.item())
    training_loss = np.mean(training_loss)
    training_losses.append(training_loss)
    train_accuracy = accuracy_score(y_pred=train_epoch_preds, y_true=train_epoch_labels)
    training_accuracies.append(train_accuracy)
    if scheduler:
       scheduler.step()
    
    model.eval()
    valid_epoch_preds = []
    valid_epoch_labels = []
    for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
      input_names, targets = batch['file_names'], batch['labels'].long().to(device)
      inputs = batch_representation_loader(input_names, representation_config["layer"], representation_config["path"], representation_config["model"], representation_config["data_set"]).to(device)
      output = model(inputs)
      output =  output.logits

      loss = loss_fn(output, targets)

      valid_loss.append(loss.data.item())
      preds = torch.argmax(torch.softmax(output, dim=-1), dim=-1).to('cpu').detach().numpy().tolist()
      valid_epoch_preds = valid_epoch_preds + preds
      labels = targets.to('cpu').detach().numpy().tolist()
      valid_epoch_labels = valid_epoch_labels + labels
      inputs, targets, output = inputs.to('cpu'), targets.to('cpu'), output.to('cpu') # Return tensors to cpu after each iteration
      
    valid_loss = np.mean(valid_loss)
    valid_losses.append(valid_loss)
    end_time = time.time()
    duration = end_time - start_time
    durations.append(duration)
    valid_accuracy = accuracy_score(y_true=valid_epoch_labels, y_pred=valid_epoch_preds)
    valid_accuracies.append(valid_accuracy)

    if scheduler:
      print(f'Epoch{epoch+1} {duration:.0f} seconds, Train Loss:{training_loss: .2f}, Train ACC:{train_accuracy: .2f}, Valid Loss:{valid_loss:.2f}, Valid Acc:{valid_accuracy : .2f}, LR:{scheduler.get_last_lr()}')
    else:
      print(f'Epoch{epoch+1} {duration:.0f} seconds, Train Loss:{training_loss: .2f}, Train ACC:{train_accuracy: .2f} Valid Loss:{valid_loss:.2f}, Valid Acc:{valid_accuracy : .2f}')

    #Check for saving path
    if not os.path.exists(saving_path):
      os.makedirs(saving_path)
    
    # Check for improvement
    if valid_loss < best_val_loss:
        best_val_loss = valid_loss
        if metric_for_save_best == "loss":
          torch.save(model.state_dict(), saving_path + "/" + model_name)
          print("model SAVED due to loss record")

    if valid_accuracy > best_val_accuracy:
      best_val_accuracy = valid_accuracy
      if metric_for_save_best == "accuracy":
        torch.save(model.state_dict(), saving_path + "/" + model_name)
        print("model SAVED due to accuracy record")

  print(f"Each epoch took {np.mean(durations):.2f} seconds in average.")

  draw_line_plot(values_list= [training_losses, valid_losses], labels_list=["Training Loss", "Validation Loss"], x_label="Epoch", y_label="Loss")
  draw_line_plot(values_list= [training_accuracies, valid_accuracies], labels_list=["Training Accuracy", "Validation Accuracy"], x_label="Epoch", y_label="Accuracy")

  return {
     "training_losses": training_losses,
     "validation_losses": valid_losses,
     "tarining_accuracies": training_accuracies,
     "validation_accuracies": valid_accuracies
  }