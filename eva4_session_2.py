from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # 28x28x16
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.01)
        )

        # CONV Block 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),  # 28x28x32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.01)
        )

        # Transition Block 1
        self.trans1 = nn.Sequential(
            nn.MaxPool2d(2, 2)  # 14x14x32
        )

        # CONV Block 2
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 16, 1),  # Channel reduction
            nn.Conv2d(16, 32, 3, padding=1),  # 14x14x32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.01)
        )

        # Transition Block 2
        self.trans2 = nn.Sequential(
            nn.MaxPool2d(2, 2)  # 7x7x32
        )

        # Output Block
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),  # 7x7x16
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.01)
        )
        
        # Final Layer
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 1x1x16
            nn.Conv2d(16, 10, 1)  # 1x1x10
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.trans1(x)
        x = self.conv3(x)
        x = self.trans2(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
summary(model, input_size=(1, 28, 28))

torch.manual_seed(1)
batch_size = 128

# Enhanced data augmentation
train_transforms = transforms.Compose([
    transforms.RandomRotation((-10.0, 10.0), fill=(1,)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-7, 7)),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.2, contrast=0.2)
    ], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                    transform=train_transforms),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=test_transforms),
    batch_size=batch_size, shuffle=True, **kwargs)

from tqdm import tqdm
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        
        pbar.set_description(desc=f'Epoch={epoch} Loss={loss.item():.4f} Batch={batch_idx} Accuracy={100*correct/processed:0.2f}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=True)
total_epochs = 20  # Increased epochs

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,
    epochs=total_epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.2,
    anneal_strategy='cos',
    div_factor=25,
    final_div_factor=1000
)

best_acc = 0
target_acc = 99.4  # Target accuracy

for epoch in range(1, total_epochs + 1):
    print(f'Epoch {epoch}/{total_epochs}:')
    train(model, device, train_loader, optimizer, epoch)
    acc = test(model, device, test_loader)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'best_model.pth')
    
    if acc >= target_acc:
        print(f'\nReached target accuracy of {target_acc}% in epoch {epoch}!')
        break

print(f'Best Test Accuracy: {best_acc:.2f}%')
