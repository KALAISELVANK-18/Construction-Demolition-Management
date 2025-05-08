import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
def compute_IoU(cm):
    '''
    Adapted from:
        https://github.com/davidtvs/PyTorch-ENet/blob/master/metric/iou.py
        https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/metrics.py#L2716-L2844
    '''
    
    sum_over_row = cm.sum(axis=0)
    sum_over_col = cm.sum(axis=1)
    true_positives = np.diag(cm)

    # sum_over_row + sum_over_col = 2 * true_positives + false_positives + false_negatives.
    denominator = sum_over_row + sum_over_col - true_positives

    iou = true_positives / denominator

    return iou, np.nanmean(iou)


def eval_net_loader(net, val_loader, n_classes, criterion, device='cpu'):
    net.eval()  # Set the model to evaluation mode
    labels = np.arange(n_classes)
    cm = np.zeros((n_classes, n_classes))
    val_loss = 0.0
    num_batches = len(val_loader)

    # Disable gradient calculation for validation
    with torch.no_grad():
        for i, sample_batch in enumerate(val_loader):
            imgs = sample_batch['image'].to(device)
            true_masks = sample_batch['mask'].to(device)

            outputs = net(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            # Calculate loss for this batch
            loss = criterion(outputs, true_masks)
            val_loss += loss.item()

            # Accumulate confusion matrix for IoU calculation
            for j in range(len(true_masks)):
                true = true_masks[j].cpu().detach().numpy().flatten()
                pred = preds[j].cpu().detach().numpy().flatten()
                cm += confusion_matrix(true, pred, labels=labels)

    # Compute IoU from the accumulated confusion matrix
    class_iou, mean_iou = compute_IoU(cm)

    # Calculate the average validation loss
    avg_val_loss = val_loss / num_batches
    avg_val_loss = float(avg_val_loss)
    return avg_val_loss, class_iou, mean_iou


def IoU(mask_true, mask_pred, n_classes=2):
        
        labels = np.arange(n_classes)
        print(labels)
        cm = confusion_matrix(mask_true.flatten(), mask_pred.flatten(), labels=labels)

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)

        return compute_IoU(cm)