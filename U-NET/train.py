import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from tensorboardX import SummaryWriter

from model import UNet, make_dataloaders, eval_net_loader, make_checkpoint_dir
from lib import plot_net_predictions


def train_epoch(epoch, train_loader, criterion, optimizer, batch_size, scheduler, net, writer, device):
    net.train()
    epoch_loss = 0
    correct = 0
    total = 0
    num_batches = len(train_loader)

    print(f'Starting Epoch {epoch + 1}/{scheduler.last_epoch + 1}...')

    for i, sample_batch in enumerate(train_loader):
        imgs = sample_batch['image']
        true_masks = sample_batch['mask']

        # Move data to the proper device (CPU/GPU)
        imgs = imgs.to(device)
        true_masks = true_masks.to(device)

        # Forward pass
        outputs = net(imgs)
        probs = torch.softmax(outputs, dim=1)
        masks_pred = torch.argmax(probs, dim=1)

        # Calculate loss
        loss = criterion(outputs, true_masks)
        epoch_loss += loss.item()

        # Calculate accuracy
        correct += (masks_pred == true_masks).sum().item()
        total += true_masks.numel()

        # Print progress every few batches
        if i % 10 == 0:
            print(
                f'Epoch [{epoch + 1}/{scheduler.last_epoch + 1}], Step [{i + 1}/{num_batches}], Loss: {loss.item():.4f}')

        # Log data for TensorBoard
        if i % 100 == 0:
            writer.add_scalar('train_loss_iter', loss.item(), i + num_batches * epoch)
            writer.add_figure('predictions vs actuals',
                              plot_net_predictions(imgs, true_masks, masks_pred, batch_size),
                              global_step=i + num_batches * epoch)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Average loss and accuracy for the epoch
    avg_loss = epoch_loss / num_batches
    accuracy = correct / total

    # Print epoch summary
    print(
        f'Epoch [{epoch + 1}/{scheduler.last_epoch + 1}] finished! Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%')

    scheduler.step()

    # Log epoch-level loss and accuracy to TensorBoard
    writer.add_scalar('train_loss_epoch', avg_loss, epoch)
    writer.add_scalar('train_acc_epoch', accuracy, epoch)

    return avg_loss, accuracy


def validate_epoch(epoch, train_loader, val_loader, net, criterion, device):
    avg_val_loss, class_iou, mean_iou = eval_net_loader(net, val_loader, n_classes=2, criterion=criterion,
                                                        device=device)
    print(f'Epoch {epoch + 1} Validation - Loss: {avg_val_loss:.4f}, Mean IoU: {mean_iou:.4f}')
    print('Class IoU:', ' '.join(f'{x:.3f}' for x in class_iou))

    # Save to summary
    writer.add_scalar('mean_iou', mean_iou, len(train_loader) * (epoch + 1))
    writer.add_scalar('val_loss', avg_val_loss, len(train_loader) * (epoch + 1))

    return avg_val_loss, mean_iou


def train_net(train_loader, val_loader, net, device, epochs=5, batch_size=1, lr=0.1, save_cp=True, dir_checkpoint='./checkpoints/'):
    print(f'''
    Starting training:
        Epochs: {epochs}
        Batch size: {batch_size}
        Learning rate: {lr}
        Training size: {len(train_loader.dataset)}
        Validation size: {len(val_loader.dataset)}
        Checkpoints: {str(save_cp)}
        Device: {str(device)}
    ''')

    # Optimizer and scheduler
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(0.3 * epochs), gamma=0.1)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Dictionary to store history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_precision': [],
        'val_precision': []
    }

    # Best validation precision initialization
    best_precision = 0

    # Loop through epochs
    for epoch in range(epochs):
        print(f'Starting epoch {epoch + 1}/{epochs}.')

        # Training phase
        train_loss, train_precision = train_epoch(
            epoch=epoch,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            batch_size=batch_size,
            scheduler=scheduler,
            net=net,
            writer=writer,   # Assuming writer is already defined
            device=device
        )

        # Validation phase
        val_loss, val_precision = validate_epoch(epoch, train_loader, val_loader, net,criterion,device)

        # Scheduler step for learning rate adjustment
        scheduler.step()

        # Save history for this epoch
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_precision'].append(train_precision)
        history['val_precision'].append(val_precision)

        # Checkpoint saving (if it's the best precision so far)
        if save_cp:
            state_dict = net.state_dict()
            if device == "cuda":
                state_dict = net.module.state_dict()

            # Save model checkpoint and history
            checkpoint = {
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1,
                'best_precision': best_precision,
                'history': history
            }

            torch.save(checkpoint, dir_checkpoint + f'CP{epoch + 1}.pth')
            print(f'Checkpoint {epoch + 1} saved!')

            # Update best precision
            best_precision = val_precision

    # Save the complete history at the end
    torch.save(history, dir_checkpoint + 'training_history.pth')
    print('Training history saved!')

    # Close the TensorBoard writer
    writer.close()

    return history


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=50, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=8,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.001,
                      type='float', help='learning rate')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-f', '--folder', dest='folder',
                      default='image_train_main', help='folder name')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    dir_data = f'./data/{args.folder}'
    dir_checkpoint = f'./checkpoints/{args.folder}_b{args.batchsize}/'
    dir_summary = f'./runs/{args.folder}_b{args.batchsize}'
    params = {'batch_size': args.batchsize, 'shuffle': True, 'num_workers': 4}

    make_checkpoint_dir(dir_checkpoint)
    writer = SummaryWriter(dir_summary)

    val_ratio = 0.2
    train_loader, val_loader = make_dataloaders(dir_data, val_ratio, params)

    net = UNet(n_channels=3, n_classes=2)
    net.to(device)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    # train model in parallel on multiple-GPUs
    if torch.cuda.device_count() > 1:
        print("Model training on", torch.cuda.device_count(), "GPUs")
        net = nn.DataParallel(net)

    try:
        train_net(train_loader, val_loader, net, device, epochs=args.epochs, batch_size=args.batchsize, lr=args.lr)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
