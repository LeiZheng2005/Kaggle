import os
import torch
import warnings
import shutil
from multiprocessing import freeze_support

###
from Data_Operations.Data_Loading.Date_read_and_crop import data_read_and_crop
from Data_Operations.Data_Loading.Dataset_Divide import dataset_divide
from Data_Operations.Data_Loading.Data_Loading import data_loading
###

###
from Model_with_Network.train_model import train_model_with_early_stopping
from Model_with_Network.Set_Resnet50 import Resnet50
###

###
from Loss_with_Evaluate.Loss_with_Evaluate import loss_with_evaluate, Evaluate
###



if __name__ == '__main__':
    freeze_support()

    shutil.rmtree(r"D:\Final_Result")
    os.makedirs(r"D:\Final_Result",exist_ok=True)
    print('clear finish~')

    ###
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=ResourceWarning)
    ###

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    labels = ['malignant', 'normal','benign']
    data_dir = r"D:\Final_Result"

    ####加载数据集
    print('加载数据集:呜呜呜~')
    # crop第一次处理
    input_dir = r"D:\Dataset\Dataset_BUSI_with_GT"
    output_dir = r"D:\Final_Result\OverlayedImages"
    result_dir = r"D:\Final_Result"
    data_read_and_crop(labels=labels,input_dir=input_dir,output_dir=output_dir)
    # divide_with_transforms第二次处理
    dataset_divide(input_dir=output_dir,output_dir=result_dir)


    save_model_pkl_path = r"D:\Final_Result\Resnet50_pkl"
    os.makedirs(save_model_pkl_path, exist_ok=True)
    save_model_pkl_path = os.path.join(save_model_pkl_path,'resnet50_fineTuning.h5')

    resnet50=Resnet50(labels=labels)
    resnet50.to(device=device)
    optimizer,scheduler,Loss_Function = loss_with_evaluate(model=resnet50)
    image_datasets,dataloaders,dataset_sizes = data_loading(data_dir=data_dir)

    train_losses = []
    val_losses = []
    print('Model train~')
    BCI_Model = train_model_with_early_stopping(
        model=resnet50,
        lossFunction=Loss_Function,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        class_names=labels,
        device=device,
        train_losses=train_losses,
        val_losses=val_losses,
        num_epochs=200,
        patience=10
    )
    print('torch.save~')
    torch.save(BCI_Model, save_model_pkl_path)
    print('Evaluate~')
    Evaluate(
        class_names=labels,
        model=resnet50,
        device=device,
        dataloaders=dataloaders,
        train_losses=train_losses,
        val_losses=val_losses
    )
