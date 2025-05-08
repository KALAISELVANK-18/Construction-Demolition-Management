import torch

# Load the checkpoint file
checkpoint = torch.load('checkpoints/execute/Main_CP27.pth',map_location=torch.device('cpu'))

# Fetch the training history from the checkpoint
if 'history' in checkpoint:
    history = checkpoint['history']
    print("Training history loaded successfully!")

    # Access the lists of metrics
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    train_precision = history['train_precision']
    val_precision = history['val_precision']

    # Example: Print the last epoch's training and validation precision
    print(f"Last epoch's training precision: {train_precision}")
    print(f"Last epoch's validation precision: {val_precision}")
    print(f"Last epoch's validation loss: {val_loss}")
    print(f"Last epoch's train loss: {train_loss}")
else:
    print("No history found in the checkpoint.")


import matplotlib.pyplot as plt

# Plot the precision
plt.plot(history['train_precision'], label='Training Precision')
plt.plot(history['val_precision'], label='Validation Precision')
plt.title('Training vs Validation Precision U-NET')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()
plt.show()

# Plot the loss
plt.plot(history['train_loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
