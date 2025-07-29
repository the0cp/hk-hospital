import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, matthews_corrcoef)
import numpy as np
import cv2


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_DIR = '../data_classifier/train'
VAL_DIR = '../data_classifier/val'
NUM_CLASSES = 2
EPOCHS = 100 
BATCH_SIZE = 4
LEARNING_RATE = 5e-5 
EARLY_STOP_PATIENCE = 20


class TargetedFeatureEnhancement:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8), lab_a_factor=1.2):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self.lab_a_factor = lab_a_factor
    def __call__(self, img):
        img_cv = np.array(img)
        img_rgb = img_cv
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l_enhanced = self.clahe.apply(l)
        merged_lab = cv2.merge((l_enhanced, a, b))
        enhanced_img_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced_img_rgb)


data_transforms = {
    'train': transforms.Compose([
        TargetedFeatureEnhancement(),
        transforms.RandomResizedCrop(260, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'train_g3_aggressive': transforms.Compose([
        transforms.RandAugment(),
        transforms.RandomResizedCrop(260, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        TargetedFeatureEnhancement(),
        transforms.Resize(280),
        transforms.CenterCrop(260),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class MultiViewDataset(Dataset):
    def __init__(self, root_dir, transform=None, transform_g3=None, is_multiview=True):
        self.samples = []
        self.transform = transform
        self.transform_g3 = transform_g3
        self.is_multiview = is_multiview
        self.class_to_idx = {'G0': 0, 'G1': 1, 'G2': 1, 'G3': 1}
        
        for cls in ['G0', 'G1', 'G2', 'G3']:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir): continue
            if self.is_multiview:
                front_imgs = [f for f in os.listdir(cls_dir) if f.endswith('_front.jpg')]
                for front in front_imgs:
                    base_name = front.replace('_front.jpg', '')
                    left, right = base_name + '_left.jpg', base_name + '_right.jpg'
                    if os.path.exists(os.path.join(cls_dir, left)) and os.path.exists(os.path.join(cls_dir, right)):
                        self.samples.append((os.path.join(cls_dir, front), os.path.join(cls_dir, left), os.path.join(cls_dir, right), cls))
            else:
                for img_file in os.listdir(cls_dir):
                    if img_file.lower().endswith('.jpg'):
                        self.samples.append((os.path.join(cls_dir, img_file), None, None, cls))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        front_path, left_path, right_path, cls = self.samples[idx]
        label = self.class_to_idx[cls]

        if cls == 'G3' and self.transform_g3 is not None:
            transform_to_apply = self.transform_g3
        else:
            transform_to_apply = self.transform

        front_img_pil = Image.open(front_path).convert('RGB')
        
        if transform_to_apply:
            front_img = transform_to_apply(front_img_pil)
        else:
            front_img = transforms.ToTensor()(front_img_pil)
        
        if self.is_multiview and left_path and right_path:
            left_img_pil = Image.open(left_path).convert('RGB')
            right_img_pil = Image.open(right_path).convert('RGB')
            if transform_to_apply:
                left_img = transform_to_apply(left_img_pil)
                right_img = transform_to_apply(right_img_pil)
            else:
                left_img = transforms.ToTensor()(left_img_pil)
                right_img = transforms.ToTensor()(right_img_pil)
        else:
            left_img = torch.zeros_like(front_img)
            right_img = torch.zeros_like(front_img)
            
        return front_img, left_img, right_img, label


class AttentionFusionModel(nn.Module):
    def __init__(self):
        super(AttentionFusionModel, self).__init__()
        self.base_model = timm.create_model('efficientnet_b2', pretrained=True, num_classes=0, drop_path_rate=0.2)
        feature_dim = self.base_model.num_features
        self.attention_net = nn.Sequential(nn.Linear(feature_dim * 3, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 3))
        self.classifier = nn.Linear(feature_dim, NUM_CLASSES)

    def forward(self, front, left=None, right=None):
        if self.training:
            if left is None or right is None: raise ValueError("In training mode, all three views must be provided.")
            f_front = nn.functional.adaptive_avg_pool2d(self.base_model.forward_features(front), 1).view(front.size(0), -1)
            f_left = nn.functional.adaptive_avg_pool2d(self.base_model.forward_features(left), 1).view(left.size(0), -1)
            f_right = nn.functional.adaptive_avg_pool2d(self.base_model.forward_features(right), 1).view(right.size(0), -1)
            all_features_concat = torch.cat([f_front, f_left, f_right], dim=1)
            attention_scores = self.attention_net(all_features_concat)
            attention_weights = torch.softmax(attention_scores, dim=1)
            features_stack = torch.stack([f_front, f_left, f_right], dim=1)
            weights_reshaped = attention_weights.unsqueeze(2)
            fused_feature = torch.sum(features_stack * weights_reshaped, dim=1)
        else:
            fused_feature = nn.functional.adaptive_avg_pool2d(self.base_model.forward_features(front), 1).view(front.size(0), -1)
        output = self.classifier(fused_feature)
        return output

def train_epoch(model, loader, criterion, optimizer):
    model.train(); running_loss = 0.0
    for front, left, right, labels in loader:
        front, left, right, labels = front.to(DEVICE), left.to(DEVICE), right.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(front, left, right)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * front.size(0)
    return running_loss / len(loader.dataset)

def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_trues, all_preds, all_probs = [], [], []

    if not loader.dataset.samples: return 0.0, {}
    
    with torch.no_grad():
        for front, _, _, labels in loader:
            front, labels = front.to(DEVICE), labels.to(DEVICE)
            outputs = model(front)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            probs = torch.softmax(outputs, dim=1)
            
            running_loss += loss.item() * front.size(0)
            all_trues.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    if not all_trues: return 0.0, {}
    
    epoch_loss = running_loss / len(loader.dataset)
    
    # --- 计算所有指标 ---
    metrics = {}
    cm = confusion_matrix(all_trues, all_preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)

    metrics['accuracy'] = accuracy_score(all_trues, all_preds)
    metrics['precision'] = precision_score(all_trues, all_preds, zero_division=0)
    metrics['recall'] = recall_score(all_trues, all_preds, zero_division=0)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['f1_score'] = f1_score(all_trues, all_preds, zero_division=0)
    metrics['mcc'] = matthews_corrcoef(all_trues, all_preds)
    metrics['confusion_matrix'] = cm

    try:
        metrics['auc'] = roc_auc_score(all_trues, all_probs)
    except ValueError:
        metrics['auc'] = 0.5

    return epoch_loss, metrics

def main():
    print(f"Using device: {DEVICE}")

    train_dataset = MultiViewDataset(TRAIN_DIR, transform=data_transforms['train'], transform_g3=data_transforms['train_g3_aggressive'], is_multiview=True)
    val_dataset = MultiViewDataset(VAL_DIR, transform=data_transforms['val'], is_multiview=False)
    
    if not train_dataset.samples: print("Error: Training dataset is empty."); return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    print("Calculating class weights for 2-class problem (G0 vs Others)...")
    class_counts = [0] * NUM_CLASSES
    for *_, cls_name in train_dataset.samples:
        class_counts[train_dataset.class_to_idx[cls_name]] += 1
    weights = [1.0 / count if count > 0 else 0 for count in class_counts]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    print(f"Calculated class weights: {class_weights.cpu().numpy()}")

    model = AttentionFusionModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights) 
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_acc = 0.0
    early_stop_counter = 0
    class_names = ["G0 (Neg)", "Others (Pos)"]

    print("Starting training...")
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        
        old_lr = optimizer.param_groups[0]['lr']
        val_loss, metrics = validate(model, val_loader, criterion)
        
        val_acc = metrics.get('accuracy', 0.0)
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {old_lr:.6f}")
        
        print(f"    ├─ Val Metrics: Acc: {metrics.get('accuracy', 0):.2%}, "
              f"AUC: {metrics.get('auc', 0):.4f}, "
              f"F1: {metrics.get('f1_score', 0):.4f}, "
              f"MCC: {metrics.get('mcc', 0):.4f}")
        
        print(f"    ├─ P-R-S: Precision: {metrics.get('precision', 0):.2%}, "
              f"Recall: {metrics.get('recall', 0):.2%}, "
              f"Specificity: {metrics.get('specificity', 0):.2%}")
        
        cm = metrics.get('confusion_matrix', np.zeros((2,2)))
        print(f"    └─ Confusion Matrix (P/T):")
        print(f"            Predicted Neg | Predicted Pos")
        print(f"  True Neg:      {cm[0,0]:<12} | {cm[0,1]:<12}")
        print(f"  True Pos:      {cm[1,0]:<12} | {cm[1,1]:<12}")

        if new_lr < old_lr:
            print(f"    *** Learning rate reduced to {new_lr:.6f} ***")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_counter = 0
            torch.save(model.state_dict(), 'best_binary_model_full_metrics.pth')
            print(f"    🚀 Validation accuracy improved! New best model saved.")
        else:
            early_stop_counter += 1

        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print("🛑 Early stopping triggered."); break

    print(f"Finished training. Best model saved with validation accuracy: {best_val_acc:.2%}")

if __name__ == "__main__":
    main()