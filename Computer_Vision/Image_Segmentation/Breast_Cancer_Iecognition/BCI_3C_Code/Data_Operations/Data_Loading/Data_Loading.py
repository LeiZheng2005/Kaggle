import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from Data_Operations.Data_Augmentation.Data_Transforms import data_transforms
def data_loading(data_dir):
    # data_dir = r"D:\Final_Result"

    # Create datasets for train, validation, and test
    image_datasets = {
        x: ImageFolder(
            root=os.path.join(data_dir, x),
            transform=data_transforms(x)
        )
        for x in ['train', 'validation', 'test']
    }

    # Specify batch size for dataloaders
    batch_size = 32  # You can adjust this based on your hardware and preferences

    # Create dataloaders for train, validation, and test
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
                   for x in ['train', 'validation', 'test']}

    # Calculate dataset sizes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation', 'test']}

    # Get class labels
    class_names = image_datasets['train'].classes

    # Print dataset sizes and class labels
    print("Dataset Sizes:", dataset_sizes)
    print("Class Labels:", class_names)
    return image_datasets,dataloaders,dataset_sizes

