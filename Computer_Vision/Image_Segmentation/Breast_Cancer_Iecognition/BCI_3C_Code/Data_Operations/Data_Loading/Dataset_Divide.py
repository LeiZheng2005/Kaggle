import os
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
def dataset_divide(input_dir,output_dir):
    print('dataset_divide start~')
    # Set the path to your input folder
    # labels = ['malignant', 'normal','benign']
    # input_dir = r"D:\Final_Result\OverlayedImages"
    # output_dir = r"D:\Final_Result"
    # Create a list to store file paths and labels
    labels = []
    file_paths = []
    # Loop through the subdirectories (benign, malignant, normal)
    for label in os.listdir(input_dir):
        print('labels:',label)
        label_dir = os.path.join(input_dir, label)
        if os.path.isdir(label_dir):
            for image_file in tqdm(os.listdir(label_dir)):
                if image_file.endswith('.png') and not (image_file.endswith('_mask.png') or
                                                         image_file.endswith('_mask_1.png') or
                                                         image_file.endswith('_mask_2.png')):
                    image_path = os.path.join(label_dir, image_file)
                    labels.append(label)
                    file_paths.append(image_path)

    # Create a DataFrame to store the file paths and labels
    data = pd.DataFrame({'Image_Path': file_paths, 'Label': labels})

    # Split the dataset into train, validation, and test sets
    train_data, test_data = train_test_split(data, test_size=0.15, random_state=42, stratify=data['Label'])
    train_data, val_data = train_test_split(train_data, test_size=0.15, random_state=42, stratify=train_data['Label'])

    # Define the paths for the train, validation, and test directories

    train_dir = os.path.join(output_dir,'train')
    val_dir = os.path.join(output_dir,'validation')
    test_dir = os.path.join(output_dir,'test')

    # Create the train, validation, and test directories and subdirectories
    for label in os.listdir(input_dir):
        os.makedirs(os.path.join(train_dir, label), exist_ok=True)
        os.makedirs(os.path.join(val_dir, label), exist_ok=True)
        os.makedirs(os.path.join(test_dir, label), exist_ok=True)

    # Copy the images to the corresponding directories
    for _, row in train_data.iterrows():
        image_path = row['Image_Path']
        label = row['Label']
        shutil.copy(image_path, os.path.join(train_dir, label))

    for _, row in val_data.iterrows():
        image_path = row['Image_Path']
        label = row['Label']
        shutil.copy(image_path, os.path.join(val_dir, label))

    for _, row in test_data.iterrows():
        image_path = row['Image_Path']
        label = row['Label']
        shutil.copy(image_path, os.path.join(test_dir, label))
    print('Dataset divided over!')