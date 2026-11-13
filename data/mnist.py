from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_mnist_loaders(batch_size):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    full = datasets.MNIST("data", train=True, download=True, transform=tf)
    test = datasets.MNIST("data", train=False, download=True, transform=tf)
    train, val = random_split(full, [54000, 6000])
    return (DataLoader(train, batch_size, shuffle=True),
            DataLoader(val, batch_size),
            DataLoader(test, batch_size),
            [str(i) for i in range(10)])