import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
from torchsampler import ImbalancedDatasetSampler



def get_dataLoader(dataDir, batch_size=32, workers=1):
    trainDir = os.path.join(dataDir, 'train')
    valDir = os.path.join(dataDir, 'val',)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])


    train_dataset = datasets.ImageFolder(trainDir, transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                 ]))

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=ImbalancedDatasetSampler(train_dataset),
            batch_size=batch_size,
            num_workers=workers, pin_memory=True)


    val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valDir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True)


    return train_loader, val_loader




class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path





def get_testLoader(dataDir):
        batch_size = 2
        workers = 1

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        test_loader = torch.utils.data.DataLoader(
            ImageFolderWithPaths(dataDir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True)

        return test_loader
