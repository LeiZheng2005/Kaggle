import os
import torch

# Function to count the number of files in a directory
def test_crop_dir(labels,input_dir,output_dir):
    print('test_crop_dir start!')
    def count_files_in_directory(directory):
        return sum(len(files) for _, _, files in os.walk(directory))

    # labels = ['benign', 'malignant', 'normal']
    # input_dir = r"D:\Dataset\Dataset_BUSI_with_GT"
    # output_dir = r"D:\Final_Result\OverlayedImages"

    # Count the files in the input and output directories
    input_counts = {}
    output_counts = {}

    # Count files in input directory
    for label in labels:
        label_dir = os.path.join(input_dir, label)
        if os.path.isdir(label_dir):
            input_counts[label] = count_files_in_directory(label_dir)

    # Count files in output directory
    for label in labels:
        label_dir = os.path.join(output_dir, label)
        if os.path.isdir(label_dir):
            output_counts[label] = count_files_in_directory(label_dir)

    # Print file counts
    print("File Counts Before Overlay-includes masks:")
    for label, count in input_counts.items():
        print(f"{label}: {count} files")

    print("\nFile Counts After Overlay:")
    for label, count in output_counts.items():
        print(f"{label}: {count} files")
    print('test_crop_dir over!')
    
def test_dataset_divide(data_dir):
    print('test_dataset_divide start!')

    train_dir = os.path.join(data_dir,'train')

    # List the subdirectories (benign, malignant, normal)
    subdirectories = ['benign', 'malignant', 'normal']

    # Dictionary to store file counts
    file_counts = {}

    # Loop through the subdirectories and count files in each
    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(train_dir, subdirectory)
        if os.path.exists(subdirectory_path):
            file_count = len(os.listdir(subdirectory_path))
            file_counts[subdirectory] = file_count

    # Print the file counts
    for category, count in file_counts.items():
       # print("Train folder counts including masks:")
        print(f"Train {category}: {count}")

    val_dir = os.path.join(data_dir,'validation')

    # List the subdirectories (benign, malignant, normal)
    subdirectories = ['benign', 'malignant', 'normal']

    # Dictionary to store file counts
    file_counts = {}

    # Loop through the subdirectories and count files in each
    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(val_dir, subdirectory)
        if os.path.exists(subdirectory_path):
            file_count = len(os.listdir(subdirectory_path))
            file_counts[subdirectory] = file_count

    # Print the file counts
    for category, count in file_counts.items():
        #print("Validation folder counts including masks:")
        print(f"Validation {category}: {count}")


    test_dir = os.path.join(data_dir,'test')

    # List the subdirectories (benign, malignant, normal)
    subdirectories = ['benign', 'malignant', 'normal']

    # Dictionary to store file counts
    file_counts = {}

    # Loop through the subdirectories and count files in each
    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(test_dir, subdirectory)
        if os.path.exists(subdirectory_path):
            file_count = len(os.listdir(subdirectory_path))
            file_counts[subdirectory] = file_count

    # Print the file counts
    for category, count in file_counts.items():
        #print("test folder counts including masks:")
        print(f"test {category}: {count}")
    print('test_dataset_divide over!')


from torch.utils.data import DataLoader
def test_val_with_acc(device,model,image_datasets):

    # Set the number of images to display
    num_images_to_display = 15

    # Create a DataLoader for the test dataset
    test_dataloader = DataLoader(image_datasets['test'], batch_size=num_images_to_display, shuffle=True, num_workers=4)

    # Get a batch of test data
    inputs, labels = next(iter(test_dataloader))

    # Move inputs to the device
    inputs = inputs.to(device)

    # Convert images to grayscale
    grayscale_images = inputs.cpu().numpy().mean(axis=1)  # Convert RGB to grayscale

    # Get model predictions

    model_fineTuning = model
    with torch.no_grad():

        model_fineTuning.eval()
        outputs = model_fineTuning(inputs)
        _, preds = torch.max(outputs, 1)