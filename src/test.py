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

# classes = ('G0', 'G1','G2','G3')
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = torch.load('F:/mycode/Pytorch-UNet-master/model126.pth')
# model.eval()
# model.to(DEVICE)

#dataset_test = datasets.ImageFolder('F:/RDphotos-2021-12-5/test_set', transform_test)
# dataset_test = datasets.ImageFolder('F:/RDphotos-2021-12-5/interesting_area_manualmask', transform_test)
# dataset_test = datasets.ImageFolder('F:/RDphotos-2021-12-14_selectdata/select_generatemask', transform_test)
# dataset_test = datasets.ImageFolder('F:/RDphotos-2021-12-14_selectdata/select_predictmask', transform_test)
# print(len(dataset_test))
# # 对应文件夹的label
# correct = 0
# for index in range(len(dataset_test)):
#     item = dataset_test[index]
#     img, label = item
#     img.unsqueeze_(0)
#     data = Variable(img).to(DEVICE)
#     output = model(data)
#     _, pred = torch.max(output.data, 1)
# #    print(label,pred.data.item)
# #    if int(dataset_test.imgs[index][0][37]) == pred.data.item():
#   #  print(dataset_test.imgs[index][0][56:60])
#     class3 = int(dataset_test.imgs[index][0][59])
#     predictv = pred.data.item()
#     if class3 == 3:
#         class3 =2
#     if pred.data.item() == 3:
#         predictv=2
# #    elif class3 == pred.data.item():
#     if class3 == predictv:
#         correct=correct+1
#     else:
#         print('Image Name:{},predict:{}'.format(dataset_test.imgs[index][0], classes[pred.data.item()]))
#     index += 1
# print('acc = ',correct/len(dataset_test))

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