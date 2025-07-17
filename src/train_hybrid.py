import torch
import torch.nn as nn
import torch.optim as optim
import os
import logging
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from efficientnet_pytorch import EfficientNet
from dtsk import D_TSK_FC

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


def extract_features(model, data_loader, device):
    features_list = []
    labels_list = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            features = model.extract_features(inputs)
            features = nn.AdaptiveAvgPool2d(1)(features)
            features = features.view(features.size(0), -1)
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    return np.concatenate(features_list), np.concatenate(labels_list)


def train_dtsk_fc(train_features, train_labels, num_classes):
    logging.info("Initializing and training D-TSK-FC model...")
    
    encoder = OneHotEncoder(sparse_output=False, categories=[range(num_classes)])
    train_labels_one_hot = encoder.fit_transform(train_labels.reshape(-1, 1))

    model_params = {
        'DP': 3,
        'K_dp_list': [80, 50, 30],
        'alpha': 0.03,
        'C': 1000
    }
    
    model = D_TSK_FC(**model_params)
    model.fit(train_features, train_labels_one_hot)
    
    return model

def validate_dtsk_fc(model, val_features, val_labels):
    logging.info("Evaluating the model on the validation set...")
    predictions = model.predict(val_features)
    accuracy = accuracy_score(val_labels, predictions)
    logging.info(f"Validation Accuracy: {accuracy:.4f}")
    return accuracy


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        logging.error(f"Data directory not found. Please check the path: {DATA_DIR}")
        return

    train_dataset = datasets.ImageFolder(root=train_path, transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(root=val_path, transform=data_transforms['val'])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    logging.info(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images.")

    logging.info("Loading pre-trained EfficientNet for feature extraction...")
    feature_extractor = EfficientNet.from_pretrained(MODEL_NAME)
    feature_extractor.to(DEVICE)
    
    logging.info("Extracting features from training and validation sets...")
    train_features, train_labels = extract_features(feature_extractor, train_loader, DEVICE)
    val_features, val_labels = extract_features(feature_extractor, val_loader, DEVICE)
    logging.info(f"Feature extraction complete. Train features shape: {train_features.shape}, Val features shape: {val_features.shape}")
    
    trained_model = train_dtsk_fc(train_features, train_labels, NUM_CLASSES)

    validate_dtsk_fc(trained_model, val_features, val_labels)

    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    model_path = os.path.join(BEST_MODEL_DIR, 'dtsk_fc_model.pkl')
    trained_model.save(model_path)
    logging.info(f"Final trained D-TSK-FC model saved to {model_path}")

if __name__ == '__main__':
    main()