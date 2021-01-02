import os
import torchvision.transforms as transforms
import torch




def get_dataLoader(dataDir, batch_size=32, workers=1):

    trainDir = os.path.join(dataDir, 'train', 'images')
    testDir = os.path.join(dataDir, 'test', 'images')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])


    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(trainDir, transforms.Compose([
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(testDir, transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True)
    return train_loader, test_loader