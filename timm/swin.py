import torch
import timm
import torchvision.transforms as transforms
from timm.data import resolve_data_config
from torch.utils.data import DataLoader
from tqdm import tqdm

from dateLoader import getDataLoader
import torch.nn as nn
import torch.utils as utils
import torch.optim as optim
from timm.data.transforms_factory import create_transform

img_dir = '../checkpoint/ILSVRC2012_img_val'
labels_file = '../checkpoint/ILSVRC2012_validation_ground_truth.txt'
custom_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 根据Swin Transformer的模型选择适当的输入尺寸
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)

# Create Transform
transform = create_transform(**resolve_data_config(model.default_cfg, model=model))
train_dataLoader, test_dataLoader = getDataLoader(img_dir, labels_file, transform)
criterion = nn.CrossEntropyLoss()  # 选择适当的损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.9)  # 选择适当的优化器和学习率
num_epochs = 10  # 设置适当的训练轮数


def train():
    # for epoch in range(num_epochs):
    #     model.train()
    #     running_loss = 0.0
    #
    #     for images, labels in tqdm(train_dataLoader):
    #         optimizer.zero_grad()
    #         outputs = model(images)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         running_loss += loss.item()
    #
    #     # 打印每个训练轮次的损失
    #     print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(train_dataLoader)}')
    validate()

    print('Training Finished')


def validate():
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_dataLoader):
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on the test dataset: {accuracy:.2f}%')


def main():
    train()


if __name__ == main():
    main()
