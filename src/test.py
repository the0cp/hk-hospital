import torch
import torch.nn as nn
import os
import logging
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet

MODEL_PATH = '../models/classification/best_efficientnet-b0.pth'
TEST_DIR = '../data_classifier/val/'
MODEL_NAME = 'efficientnet-b0'
NUM_CLASSES = 4
BATCH_SIZE = 16     # For data loader


data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device("cuda")

    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model not found: '{MODEL_PATH}'")
        return
    if not os.path.exists(TEST_DIR):
        logging.error(f"Test folder not found: '{TEST_DIR}'")
        return

    logging.info(f"Loading model: {MODEL_NAME}")
    model = EfficientNet.from_name(MODEL_NAME)
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, NUM_CLASSES)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
        logging.error(f"{e}")
        return
        
    model.to(device)
    model.eval()
    
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=data_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    corrects = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

    accuracy = corrects.double() / total
    logging.info(f'\n--- Test Result --- \nAccuracy: {corrects}/{total} ({accuracy:.4f})')

if __name__ == '__main__':
    main()