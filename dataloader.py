import torch
from config import cfg
from torchvision import transforms, datasets

# part 0: parameter
input_size = cfg.INPUT_SIZE
batch_size = cfg.BATCH_SIZE

# part 1: transforms
train_transforms = transforms.Compose([
    transforms.RandomRotation(5),
    transforms.RandomResizedCrop(input_size[0]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

valid_transforms = transforms.Compose([
    transforms.Resize(input_size),
    transforms.RandomResizedCrop(input_size[0]),
    transforms.ToTensor(),
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))
])

# part 2: dataset
train_dataset = datasets.ImageFolder(root=cfg.TRAIN_DATASET_DIR,
                                     transform=train_transforms)
valid_dataset = datasets.ImageFolder(root=cfg.VALID_DATASET_DIR,
                                     transform=valid_transforms)

# part 3: dataloader
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=1)
valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=1)

# part 4: test
if __name__ == "__main__":

    for image, label in train_dataloader:
        print(image.shape, label.shape, len(train_dataloader))