import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from efficientnet_pytorch import EfficientNet

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# pip install efficientnet_pytorch

DATA_DIR = '../data_classifier/'
CHECKPOINTS_DIR = '../checkpoints/classifier/'
BEST_MODEL_DIR = '../models/classification/'

MODEL_NAME = 'efficientnet-b0'
NUM_CLASSES = 4

EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.2
DEVICE = torch.device('cuda')


# recommend
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
  #      nn.init.xavier_normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight,1)
#        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight,1)
 #       nn.init.constant_(m.bias, 0)

def adjust_learning_rate(optimizer, epoch):
    modellrnew = LEARNING_RATE * (0.2 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew
    return modellrnew


# 定义训练过程

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


# 验证过程
def val(model, device, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    corrects = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = corrects.double() / len(val_loader.dataset)
    return epoch_loss, epoch_acc

def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f"Using device: {DEVICE}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_path = os.path.join(DATA_DIR, 'train')
    val_path = os.path.join(DATA_DIR, 'val')

    train_dataset = datasets.ImageFolder(root=train_path, transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(root=val_path, transform=data_transforms['val'])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    logging.info(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images.")

    model = EfficientNet.from_pretrained(MODEL_NAME, num_classes=NUM_CLASSES)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_acc = 0.0
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    
    for epoch in range(1, EPOCHS + 1):
        current_lr = adjust_learning_rate(optimizer, epoch)
        
        train_loss = train(model, DEVICE, train_loader, criterion, optimizer)
        val_loss, val_acc = val(model, DEVICE, val_loader, criterion)
        
        logging.info(f"Epoch {epoch}/{EPOCHS} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, f'{MODEL_NAME}_epoch{epoch}.pth')
        torch.save(model.state_dict(), checkpoint_path)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_path = os.path.join(BEST_MODEL_DIR, f'best_{MODEL_NAME}.pth')
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"🐈👌🏻New model saved to {best_model_path} with accuracy: {best_acc:.4f}")

    logging.info(f"Finished training. Best validation accuracy: {best_acc:.4f}")

if __name__ == '__main__':
    main()