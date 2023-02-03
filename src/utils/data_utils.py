import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

#
def get_dataset(dataroot):

    train_dataset = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root=dataroot, train=False, download=True)

    return train_dataset, test_dataset


class Custom_dataset(Dataset):

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    def __init__(self, data, targets, transform=transform_test):
        self.data = data
        self.targets = targets
        self.n = len(list(targets))
        self.index = list(range(self.n))
        self.transform = transform

    def __getitem__(self, i):
        img = self.data[i]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[i], self.index[i]

    def __len__(self):
        return self.n

    def update_label(self, noise_label):
        self.targets[:] = noise_label[:]


if __name__ == "__main__":

    a = 0
    # train_dataset, test_dataset = get_dataset('./')
    # a = train_dataset.data[0]
    # print(a[0, :, 0])
    #
    # train_dataset = torchvision.datasets.CIFAR10(root='./', train=True, download=True)
    # a = train_dataset.data[0]
    # print(a[0, :, 0])
    #
    # b = pk.load(open('first_image.pk', 'rb'))
    # print(b[0, :, 0])

