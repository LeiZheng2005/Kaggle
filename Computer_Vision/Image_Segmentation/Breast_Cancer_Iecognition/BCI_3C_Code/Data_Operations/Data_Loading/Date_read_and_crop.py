import os
from PIL import Image
from tqdm import tqdm
def data_read_and_crop(labels,input_dir,output_dir):
    # labels = ['benign', 'malignant', 'normal']
    # input_dir = r"D:\Dataset\Dataset_BUSI_with_GT"
    # output_dir = r"D:\Final_Result\OverlayedImages"
    print('data_read_and_crop start!')
    os.makedirs(output_dir, exist_ok=True)
    for label in labels:
        os.makedirs(os.path.join(output_dir, label), exist_ok=True)

    # Function to overlay images and masks, resize if needed, and save the result
    def overlay_and_save(image_path, mask_path, output_path):
        try:
            # Check if both image and mask files exist
            if os.path.exists(image_path) and os.path.exists(mask_path):
                # Open the actual image and mask image
                image = Image.open(image_path)
                mask = Image.open(mask_path)

                # Ensure both images have the same color mode
                if image.mode != mask.mode:
                    mask = mask.convert(image.mode)

                # Resize the images if their sizes don't match
                if image.size != mask.size:
                    image = image.resize(mask.size)

                # Overlay the image with the mask
                overlayed = Image.blend(image, mask, alpha=0.5)

                # Save the overlayed image to the appropriate label folder
                label = os.path.basename(os.path.dirname(image_path))
                output_path = os.path.join(output_dir, label, os.path.basename(image_path))
                overlayed.save(output_path)
            else:
                #print(f"File not found for: {image_path} or {mask_path}. Skipping...")
                pass
        except Exception as e:
            print(f"An error occurred for: {image_path} or {mask_path}. Error: {str(e)}")

    # Iterate through the subdirectories (benign, malignant, normal)
    for label in labels:
        print('labels:',label)
        label_dir = os.path.join(input_dir, label)
        if os.path.isdir(label_dir):
            for image_filename in tqdm(os.listdir(label_dir)):
                if image_filename.endswith('.png'):
                    image_path = os.path.join(label_dir, image_filename)
                    # Construct the mask file path based on the naming convention
                    mask_filename = image_filename.replace('.png', '_mask.png')
                    mask_path = os.path.join(label_dir, mask_filename)
                    overlay_and_save(image_path, mask_path, output_dir)

    print("Overlayed images have been saved to D:\Final_Result\OverlayedImages directory.")


# # Set the path to the Kaggle working directory
# working_dir = r"D:\Final_Result"
#
# # Delete all files and subdirectories in the working directory
# for item in os.listdir(working_dir):
#     item_path = os.path.join(working_dir, item)
#     if os.path.isfile(item_path):
#         os.remove(item_path)
#     elif os.path.isdir(item_path):
#         shutil.rmtree(item_path)
#
# # Confirm that the directory is empty
# print("Final_Result working directory has been cleared.")