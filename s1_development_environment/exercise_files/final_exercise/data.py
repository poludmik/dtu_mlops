import torch


def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    # train = torch.randn(50000, 784)
    # test = torch.randn(10000, 784)

    # Load data from the data/corruptmnist/ folder for every 'train_images_0.pt' to 'train_images_5.pt' and store to train variable
    data = [torch.load(f'data/corruptmnist/train_images_{i}.pt') for i in range(6)]
    # for i in data:
    #     print(i.shape)
    train = torch.cat(data)
    train_labels = torch.cat([torch.load(f'data/corruptmnist/train_target_{i}.pt') for i in range(6)]) 

    test = torch.load('data/corruptmnist/test_images.pt')
    # print(test.shape)
    test_labels = torch.load('data/corruptmnist/test_target.pt')
    # create a histogram of the labels and show it
    # import matplotlib.pyplot as plt
    # plt.hist(test_labels)
    # plt.show()

    # Create dataloaders for train and test
    train = torch.utils.data.TensorDataset(train, train_labels)
    test = torch.utils.data.TensorDataset(test, test_labels)

    return train, test

def vizualize_random_image(dataloader):
    import matplotlib.pyplot as plt
    import random
    # get a random image from the dataloader
    image, label = random.choice(dataloader)
    plt.imshow(image.reshape(28, 28))
    plt.show()


if __name__ == '__main__':
    tr, tst = mnist()
    vizualize_random_image(tr)
