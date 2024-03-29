from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn import functional as F
import torch 

class Loss:
    def __init__(self, dataset_name, device):
        self.device = device

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        transform2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)                              
        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform2)

        kwargs = {'num_workers': 8, 'pin_memory': True}
        self.train_loader = DataLoader(train_dataset, batch_size=10000, shuffle=True, **kwargs)
        self.test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False, **kwargs)

        self.criterion = torch.nn.MSELoss()


    def get_loss(self, dnn, loss_name, whichloss):
        if whichloss=="mse":
            loss = 0
            loader = self.test_loader if loss_name == "test_loss" else self.train_loader
            with torch.no_grad():
                for data, target in loader:
                    output = dnn(data.to(self.device))
                    loss += F.nll_loss(output, target.to(self.device), reduction='sum').item()  # Sum up batch loss
            loss /= len(loader.dataset)
            return torch.tensor(loss)
        else:
            raise "loss not defined"


    