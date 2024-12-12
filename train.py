from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from model import Net

def train(model, device, train_loader, optimizer, epoch, scheduler):
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

def get_transforms():
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
    
    return train_transforms, test_transforms

def get_dataloaders():
    batch_size = 128
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_transforms, test_transforms = get_transforms()
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                      transform=train_transforms),
        batch_size=batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=test_transforms),
        batch_size=batch_size, shuffle=True, **kwargs)
        
    return train_loader, test_loader

def main():
    torch.manual_seed(1)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    train_loader, test_loader = get_dataloaders()
    model = Net().to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=True)
    total_epochs = 20
    
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
    target_acc = 99.4
    
    for epoch in range(1, total_epochs + 1):
        print(f'Epoch {epoch}/{total_epochs}:')
        train(model, device, train_loader, optimizer, epoch, scheduler)
        acc = test(model, device, test_loader)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        if acc >= target_acc:
            print(f'\nReached target accuracy of {target_acc}% in epoch {epoch}!')
            break
    
    print(f'Best Test Accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    main() 