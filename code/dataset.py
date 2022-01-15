import numpy as np
import torch
import torchvision as tv
import torchvision.transforms as transforms


class MyRegression(torch.utils.data.Dataset):
    def __init__(self, size, dim):
        self.X = torch.normal(0, 10, size=(size, dim))
        self.y = torch.norm(self.X, dim=1).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        feature = self.X[index]
        label = self.y[index]
        return feature, label


class MyClassification(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        feature = self.X[index]
        label = self.y[index]
        return feature, label


def make_regression_dataset(size, dim):
    # (trainsize : validsize : testsize) = (5 : 1 : 1)
    trainsize = max(size * 5 // 7, 1)
    validsize = max(size // 7, 1)
    testsize = max(size - trainsize - validsize, 1)

    trainset = MyRegression(trainsize, dim)
    validset = MyRegression(validsize, dim)
    testset = MyRegression(testsize, dim)

    return trainset, validset, testset


def make_classification_dataset(size):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    mnist_train = tv.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_x, train_y = torch.empty([60000, 784]), torch.empty([60000], dtype=torch.long)
    for i, t in enumerate(list(mnist_train)):
        train_x[i], train_y[i] = t[0].reshape((-1)), t[1]
    trainset = MyClassification(train_x, train_y)

    mnist_test = tv.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_x, test_y = torch.empty([10000, 784]), torch.empty([10000], dtype=torch.long)
    for i, t in enumerate(list(mnist_test)):
        test_x[i], test_y[i] = t[0].reshape((-1)), t[1]
    testset = MyClassification(test_x, test_y)

    return trainset, testset, testset


def make_classification_dataset2(size):
    # (trainsize : validsize : testsize) = (5 : 1 : 1)
    trainsize = max(size * 5 // 7, 1)
    validsize = max(size // 7, 1)
    testsize = max(size - trainsize - validsize, 1)
    assert (trainsize + validsize <= 60000) and (testsize <= 10000), f"datasize : {size} ?"

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])

    mnist_train = tv.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_x, train_y = torch.empty([trainsize, 784]), torch.empty([trainsize], dtype=torch.long)
    for i, t in enumerate(list(mnist_train)[:trainsize]):
        train_x[i], train_y[i] = t[0].reshape((-1)), t[1]
    trainset = MyClassification(train_x, train_y)

    valid_x, valid_y = torch.empty([validsize, 784]), torch.empty([validsize], dtype=torch.long)
    for i, t in enumerate(list(mnist_train)[trainsize:trainsize + validsize]):
        valid_x[i], valid_y[i] = t[0].reshape((-1)), t[1]
    validset = MyClassification(valid_x, valid_y)

    mnist_test = tv.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_x, test_y = torch.empty([testsize, 784]), torch.empty([testsize], dtype=torch.long)
    for i, t in enumerate(list(mnist_test)[:testsize]):
        test_x[i], test_y[i] = t[0].reshape((-1)), t[1]
    testset = MyClassification(test_x, test_y)

    return trainset, validset, testset
