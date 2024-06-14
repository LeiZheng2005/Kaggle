#Transfer Learning by fineTuning the pretrained Resnet101 Model
#Load Resnet101 pretained Model
#If pretained is not working, you can also use weights instead.

import torch.nn as nn
from torchvision.models import resnet50
def Resnet50(labels):
    # labels = ['malignant', 'normal','benign']
    resnet50_model= resnet50(pretrained=True)

    # print(resnet50)

    for param in resnet50_model.parameters():
        param.requires_grad = True

    #Get the number of Input features of Resnet last fully connected layer
    #because we are going to replace it with new fully connected layer.
    in_features = resnet50_model.fc.in_features

    #Reset the final fully connected layer of the of the pre-trained Resnet.
    resnet50_model.fc = nn.Linear(in_features, len(labels))
    print('Resnet50  finished!')
    return resnet50_model