import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
import random
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import train_test_split

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

DATA_DIR = '../rois/'
CHECKPOINTS_DIR = '../checkpoints/classifier/'
BEST_MODEL_DIR = '../models/classification/'

MODEL_NAME = 'efficientnet-b2'  # MODIFIED: Use B2 for larger input size
NUM_CLASSES = 4
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 3e-4  # MODIFIED: Higher learning rate
VALIDATION_SPLIT = 0.2
PATIENCE = 10  # MODIFIED: Early stopping patience
DEVICE = torch.device('cuda')

def adjust_learning_rate(optimizer, epoch):
    modellrnew = LEARNING_RATE * (0.2 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew
    return modellrnew

def train(model, device, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def val(model, device, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    corrects = 0
    class_correct = [0] * NUM_CLASSES
    class_total = [0] * NUM_CLASSES
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if preds[i] == label:
                    class_correct[label] += 1
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = corrects.double() / len(val_loader.dataset)
    class_acc = [class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(NUM_CLASSES)]
    logging.info(f"Class accuracies: {class_acc}")
    return epoch_loss, epoch_acc

def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f"Using device: {DEVICE}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(260),  # MODIFIED: Match B2 input size
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),  # MODIFIED: Add rotation
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # MODIFIED: Add color jitter
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(260),  # MODIFIED: Match B2 input size
            transforms.CenterCrop(260),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load the full dataset
    full_dataset = datasets.ImageFolder(root=DATA_DIR)

    # Stratified split for train and validation
    indices = list(range(len(full_dataset)))
    labels = [full_dataset.targets[idx] for idx in indices]
    train_indices, val_indices = train_test_split(indices, test_size=VALIDATION_SPLIT, stratify=labels, random_state=42)

    # Create train and validation subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # Apply transforms
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Log class distribution
    train_class_counts = [0] * NUM_CLASSES
    for idx in train_indices:
        _, target = full_dataset.imgs[idx]
        train_class_counts[target] += 1
    val_class_counts = [0] * NUM_CLASSES
    for idx in val_indices:
        _, target = full_dataset.imgs[idx]
        val_class_counts[target] += 1
    logging.info(f"Train class counts: {train_class_counts}")
    logging.info(f"Val class counts: {val_class_counts}")

    logging.info(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images.")

    model = EfficientNet.from_pretrained(MODEL_NAME, num_classes=NUM_CLASSES)
    model.to(DEVICE)

    # MODIFIED: Use inverse class frequency for weights
    class_counts = [len([i for i, t in enumerate(full_dataset.targets) if t == c]) for c in range(NUM_CLASSES)]
    weights = [1.0 / count for count in class_counts]
    weights = torch.tensor(weights).to(DEVICE)
    weights = weights / weights.sum() * NUM_CLASSES
    logging.info(f"Class weights: {weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0
    best_val_loss = float('inf')
    early_stop_counter = 0
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        current_lr = adjust_learning_rate(optimizer, epoch)
        train_loss = train(model, DEVICE, train_loader, criterion, optimizer)
        val_loss, val_acc = val(model, DEVICE, val_loader, criterion)

        logging.info(f"Epoch {epoch}/{EPOCHS} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        checkpoint_path = os.path.join(CHECKPOINTS_DIR, f'{MODEL_NAME}_epoch{epoch}.pth')
        torch.save(model.state_dict(), checkpoint_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_acc = val_acc
            best_model_path = os.path.join(BEST_MODEL_DIR, f'best_{MODEL_NAME}.pth')
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"🐈👌🏻New model saved to {best_model_path} with accuracy: {best_acc:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= PATIENCE:
            logging.info(f"Early stopping at epoch {epoch}")
            break

    logging.info(f"Finished training. Best validation accuracy: {best_acc:.4f}")

if __name__ == '__main__':
    main()