import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
import time
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet

from torch.utils.data import Dataset
from PIL import Image
import glob


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class FuzzyAttentionGate(nn.Module):
    def __init__(self, input_features, n_fuzzy_sets=3):
        super(FuzzyAttentionGate, self).__init__()
        self.input_features = input_features
        self.n_fuzzy_sets = n_fuzzy_sets

        self.centers = nn.Parameter(torch.linspace(0.0, 1.0, n_fuzzy_sets).unsqueeze(0).repeat(input_features, 1))
        self.sigmas = nn.Parameter(torch.ones(input_features, n_fuzzy_sets) * 0.4)

        self.rule_net = nn.Sequential(
            nn.Linear(input_features * n_fuzzy_sets, input_features // 2),
            nn.ReLU(),
            nn.Linear(input_features // 2, input_features),
            nn.Sigmoid()
        )

    def _gaussian_mf(self, x, centers, sigmas):
        return torch.exp(-((x - centers)**2) / (2 * sigmas**2 + 1e-8))

    def forward(self, x):
        # x: (batch_size, input_features)

        # Fuzzification
        x_expanded = x.unsqueeze(-1)
        memberships = self._gaussian_mf(x_expanded, self.centers, self.sigmas)
        
        # Rule Inference
        memberships_flat = memberships.view(x.size(0), -1)
        attention_weights = self.rule_net(memberships_flat)

        gated_features = x * attention_weights
        
        return gated_features

class EfficientNetWithFuzzyGate(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetWithFuzzyGate, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained(MODEL_NAME)
        num_base_features = self.efficientnet._fc.in_features
        
        # freeze EfficientNet parameters
        # avoid modifying the pre-trained weights
        # avoid overfitting
        for param in self.efficientnet.parameters():
            param.requires_grad = False
        
        self.fuzzy_gate = FuzzyAttentionGate(input_features=num_base_features)
        self.classifier = nn.Linear(num_base_features, num_classes)


    def forward(self, x, return_gated_features=False):
        with torch.no_grad():
            x = self.efficientnet.extract_features(x)
            x = self.efficientnet._avg_pooling(x)
            x = x.flatten(start_dim=1)

        x_normalized = torch.sigmoid(x)

        gated_x = self.fuzzy_gate(x_normalized)
        
        if return_gated_features:
            return gated_x
        
        output = self.classifier(gated_x)
        
        return output

DATA_DIR = '../data_classifier/'
CHECKPOINTS_DIR = '../checkpoints/classifier/'
BEST_MODEL_DIR = '../models/classification/'
MODEL_NAME = 'efficientnet-b0'
NUM_CLASSES = 4
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3 
DEVICE = torch.device('cuda')

class ThreeViewDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        
        self.front_image_paths = sorted(glob.glob(os.path.join(root_dir, '**', '*_front.jpg'), recursive=True))
        
        class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
        
        print(f"found {len(self.front_image_paths)} samples in {root_dir}")
        print(f"Type: {self.class_to_idx}")

    def __len__(self):
        return len(self.front_image_paths)

    def __getitem__(self, idx):
        front_path = self.front_image_paths[idx]
        
        base_path = front_path.replace('_front.jpg', '')
        left_path = base_path + '_left.jpg'
        right_path = base_path + '_right.jpg'
        
        class_name = os.path.basename(os.path.dirname(front_path))
        label = self.class_to_idx[class_name]

        try:
            front_img = Image.open(front_path).convert('RGB')
            left_img = Image.open(left_path).convert('RGB')
            right_img = Image.open(right_path).convert('RGB')
        except FileNotFoundError:
            front_img = Image.open(front_path).convert('RGB')
            left_img = front_img.copy()
            right_img = front_img.copy()

        if self.transform:
            front_img = self.transform(front_img)
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
            
        return (front_img, left_img, right_img), label

def train_fusion(model, device, train_loader, criterion, optimizer, weights={'front': 0.6, 'left': 0.2, 'right': 0.2}):
    model.train()
    running_loss = 0.0

    for (front_view, left_view, right_view), labels in train_loader:
        front_view = front_view.to(device)
        left_view = left_view.to(device)
        right_view = right_view.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()

        gated_front = model(front_view, return_gated_features=True)
        gated_left = model(left_view, return_gated_features=True)
        gated_right = model(right_view, return_gated_features=True)

        fused_features = (weights['front'] * gated_front + 
                          weights['left'] * gated_left + 
                          weights['right'] * gated_right)

        outputs = model.classifier(fused_features)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * front_view.size(0)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


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
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, return_gated_features=False)
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
    
    train_dataset = ThreeViewDataset(root_dir=train_path, transform=data_transforms['train'])
    
    val_dataset = datasets.ImageFolder(root=val_path, transform=data_transforms['val'])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    logging.info(f"Use fusion training: {len(train_dataset)} evaluate with single image: {len(val_dataset)}")



    # model = EfficientNet.from_pretrained(MODEL_NAME, num_classes=NUM_CLASSES)

    logging.info("Initializing model: EfficientNet with Fuzzy Attention Gate")
    model = EfficientNetWithFuzzyGate(num_classes=NUM_CLASSES)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    best_acc = 0.0
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    
    for epoch in range(1, EPOCHS + 1):
        current_lr = adjust_learning_rate(optimizer, epoch)
        
        train_loss = train_fusion(model, DEVICE, train_loader, criterion, optimizer)
        
        val_loss, val_acc = val(model, DEVICE, val_loader, criterion)
        
        logging.info(f"Epoch {epoch}/{EPOCHS} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, f'{MODEL_NAME}_fuzzy_epoch{epoch}.pth')
        torch.save(model.state_dict(), checkpoint_path)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_path = os.path.join(BEST_MODEL_DIR, f'best_{MODEL_NAME}_fuzzy.pth')
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"🐈👌🏻New model saved to {best_model_path} with accuracy: {best_acc:.4f}")

    logging.info(f"Finished training. Best validation accuracy: {best_acc:.4f}")

if __name__ == '__main__':
    main()