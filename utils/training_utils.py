from utils.representation_loading import batch_representation_loader
from utils.plotting_utils import draw_line_plot
import time
from tqdm import tqdm
import torch
import torch.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
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
          representation_config,
          writer=None,
          fold_idx=0):

  train_loader, val_loader = (loader["TrainLoader"], loader["TestLoader"])
  training_losses = []
  valid_losses = []
  training_accuracies = []
  valid_accuracies = []
  durations = []
  best_val_loss = float('inf')
  best_val_accuracy = 0.0
  fold_tag = f"fold_{fold_idx + 1}"

  for epoch in range(epochs):
    training_loss = []
    valid_loss = []
    start_time = time.time()
    train_epoch_labels = []
    train_epoch_preds = []

    model.train()
    for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
      optimizer.zero_grad()
      input_names, targets = batch['file_names'], batch['labels'].long().to(device)
      inputs = batch_representation_loader(input_names, representation_config["layer"], representation_config["path"], representation_config["model"], representation_config["data_set"]).to(device)
      output = model(inputs)
      output = output.logits

      loss = loss_fn(output, targets)
      loss.backward()
      optimizer.step()

      preds = torch.argmax(torch.softmax(output, dim=-1), dim=-1).to('cpu').detach().numpy().tolist()
      train_epoch_preds = train_epoch_preds + preds
      labels = targets.to('cpu').detach().numpy().tolist()
      train_epoch_labels = train_epoch_labels + labels
      inputs, targets, output = inputs.to('cpu'), targets.to('cpu'), output.to('cpu')

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
    with torch.no_grad():
      for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
        input_names, targets = batch['file_names'], batch['labels'].long().to(device)
        inputs = batch_representation_loader(input_names, representation_config["layer"], representation_config["path"], representation_config["model"], representation_config["data_set"]).to(device)
        output = model(inputs)
        output = output.logits

        loss = loss_fn(output, targets)
        valid_loss.append(loss.data.item())
        preds = torch.argmax(torch.softmax(output, dim=-1), dim=-1).to('cpu').detach().numpy().tolist()
        valid_epoch_preds = valid_epoch_preds + preds
        labels = targets.to('cpu').detach().numpy().tolist()
        valid_epoch_labels = valid_epoch_labels + labels
        inputs, targets, output = inputs.to('cpu'), targets.to('cpu'), output.to('cpu')

    valid_loss = np.mean(valid_loss)
    valid_losses.append(valid_loss)
    end_time = time.time()
    duration = end_time - start_time
    durations.append(duration)
    valid_accuracy = accuracy_score(y_true=valid_epoch_labels, y_pred=valid_epoch_preds)
    valid_accuracies.append(valid_accuracy)

    if scheduler:
      current_lr = scheduler.get_last_lr()[0]
      print(f'Epoch{epoch+1} {duration:.0f}s  Train Loss:{training_loss:.2f}  Train ACC:{train_accuracy:.2f}  Valid Loss:{valid_loss:.2f}  Valid Acc:{valid_accuracy:.2f}  LR:{scheduler.get_last_lr()}')
    else:
      current_lr = optimizer.param_groups[0]['lr']
      print(f'Epoch{epoch+1} {duration:.0f}s  Train Loss:{training_loss:.2f}  Train ACC:{train_accuracy:.2f}  Valid Loss:{valid_loss:.2f}  Valid Acc:{valid_accuracy:.2f}')

    if writer is not None:
      # Scalars: loss, accuracy, learning rate, epoch duration
      writer.add_scalars(f"{fold_tag}/loss",     {"train": training_loss, "val": valid_loss},      epoch + 1)
      writer.add_scalars(f"{fold_tag}/accuracy", {"train": train_accuracy, "val": valid_accuracy}, epoch + 1)
      writer.add_scalar(f"{fold_tag}/lr",                  current_lr, epoch + 1)
      writer.add_scalar(f"{fold_tag}/epoch_duration_sec",  duration,   epoch + 1)

      # Weight and gradient histograms for all trainable parameters
      for name, param in model.named_parameters():
        if param.requires_grad:
          writer.add_histogram(f"{fold_tag}/weights/{name}", param.data, epoch + 1)
          if param.grad is not None:
            writer.add_histogram(f"{fold_tag}/gradients/{name}", param.grad, epoch + 1)

    # Check for saving path
    if not os.path.exists(saving_path):
      os.makedirs(saving_path)

    # Check for improvement and save best model
    if valid_loss < best_val_loss:
      best_val_loss = valid_loss
      if metric_for_save_best == "loss":
        torch.save(model.state_dict(), saving_path + "/" + model_name)
        print("  model SAVED (best loss)")

    if valid_accuracy > best_val_accuracy:
      best_val_accuracy = valid_accuracy
      if metric_for_save_best == "accuracy":
        torch.save(model.state_dict(), saving_path + "/" + model_name)
        print("  model SAVED (best accuracy)")

  print(f"Each epoch took {np.mean(durations):.2f} seconds on average.")

  # Training curve figures
  loss_fig = draw_line_plot(
    values_list=[training_losses, valid_losses],
    labels_list=["Training Loss", "Validation Loss"],
    x_label="Epoch", y_label="Loss"
  )
  acc_fig = draw_line_plot(
    values_list=[training_accuracies, valid_accuracies],
    labels_list=["Training Accuracy", "Validation Accuracy"],
    x_label="Epoch", y_label="Accuracy"
  )

  if writer is not None:
    writer.add_figure(f"{fold_tag}/curves/loss",     loss_fig, epochs)
    writer.add_figure(f"{fold_tag}/curves/accuracy", acc_fig,  epochs)

  plt.close(loss_fig)
  plt.close(acc_fig)

  return {
    "training_losses": training_losses,
    "validation_losses": valid_losses,
    "tarining_accuracies": training_accuracies,
    "validation_accuracies": valid_accuracies
  }
