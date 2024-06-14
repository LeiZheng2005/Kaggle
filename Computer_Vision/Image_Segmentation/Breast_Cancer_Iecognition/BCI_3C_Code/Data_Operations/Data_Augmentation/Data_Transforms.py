from torchvision import transforms
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ColorJitter
def data_transforms(datasets_class):

    # Define the minority classes in your dataset
    class_names = ['malignant', 'normal','benign']
    minority_classes = ['malignant', 'normal']

    # Define custom data transformations for minority classes
    minority_class_transforms = transforms.Compose([
        RandomHorizontalFlip(p=0.9),  # Apply with 90% probability
        RandomRotation(15, expand=False, center=None),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])
    if datasets_class == 'train':
        f = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # Apply custom augmentations to minority classes
            transforms.RandomApply([minority_class_transforms], p=0.5) if any(cls in minority_classes for cls in class_names) else transforms.RandomApply([], p=0.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        f = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return f
    # # Define data transformations for train, validation, and test sets
    # data_transforms = {
    #     'train': transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         # Apply custom augmentations to minority classes
    #         transforms.RandomApply([minority_class_transforms], p=0.5) if any(cls in minority_classes for cls in class_names) else transforms.RandomApply([], p=0.0),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    #     'validation': transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    #     'test': transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    # }