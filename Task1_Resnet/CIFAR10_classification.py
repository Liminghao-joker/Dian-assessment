import torch
import torchvision as tv
from torch.utils.data import DataLoader
from torch import optim, nn
from torchvision.transforms import transforms
from Basic_Resnet18 import resnet18_cifar as res_net18

# set parameters
config = {
    'seed': 42,
    'batch_size': 128,
    'learning_rate': 0.01,
    'n_epochs': 10,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'num_classes': 10,
    'num_workers': 2,
}

# prepare dataset
def build_transformer():
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

def get_dataloaders():
    transform = build_transformer()
    train_set = tv.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True)
    test_set = tv.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)
    return train_loader, test_loader

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def resnet18_cifar_model(num_classes=10):
    return res_net18(num_classes=num_classes)

def train(train_loader, model, config, device):
    model.train()
    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config['momentum'])
    num_epochs = config['n_epochs']

    running_loss = 0.0
    running_correct = 0
    running_samples = 0

    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_correct += (outputs.argmax(1) == labels).sum().item()
            running_samples += labels.size(0)

            if (i + 1) % 100 == 0:  # print every 100 mini_batches
                avg_loss = running_loss / 100
                avg_acc = running_correct / running_samples * 100
                print(f'[Epoch {epoch + 1}, Iter {i + 1}] loss: {avg_loss:.4f}, Accuracy:{avg_acc:.2f}%')
                running_loss = 0.0
                running_samples = 0
                running_correct = 0

def evaluate(test_loader, model, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')

if __name__ == '__main__':
    train_loader, test_loader = get_dataloaders()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = resnet18_cifar_model(num_classes=config['num_classes'])
    model.to(device)
    train(train_loader, model, config, device)
    evaluate(test_loader, model, device)
