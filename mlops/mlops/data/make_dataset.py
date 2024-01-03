import torch
import os
import glob

if __name__ == '__main__':
    # Get the data and process it
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    train_images_list = glob.glob("data/raw/train_images_*.pt")
    train_images_list = [torch.load(x).unsqueeze(1) for x in train_images_list]
    train_images = torch.cat(train_images_list, dim=0)
    print('Train set:', train_images.shape)

    train_target_list = glob.glob("data/raw/train_target_*.pt")
    train_target_list = [torch.load(x) for x in train_target_list]
    train_target = torch.cat(train_target_list, dim=0)


    test_images = torch.load("data/raw/test_images.pt").unsqueeze(1)
    print('Test set:', test_images.shape)
    test_target = torch.load("data/raw/test_target.pt")

    # normalizing values both using the normalized train set values
    train_images = (train_images - train_images.mean()) / train_images.std()
    test_images = (test_images - train_images.mean()) / train_images.std()

    torch.save(train_images, "data/processed/train_images.pt")
    torch.save(test_images, "data/processed/test_images.pt")
    torch.save(test_target, "data/processed/test_target.pt")
    torch.save(train_target, "data/processed/train_target.pt")

    #train = TensorDataset(train_images, train_target)
    #test = TensorDataset(test_images, test_target)
