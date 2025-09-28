import os
import shutil
import random
from pathlib import Path


def split_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Split dataset into train, validation, and test sets
    """
    # Create output directories
    for split in ['train', 'valid', 'test']:
        for folder in ['images', 'labels']:
            os.makedirs(os.path.join(output_dir, split, folder), exist_ok=True)

    # Get all image files
    images_dir = os.path.join(source_dir, 'images')
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Shuffle the files
    random.shuffle(image_files)

    # Calculate split indices
    total_files = len(image_files)
    train_end = int(total_files * train_ratio)
    val_end = int(total_files * (train_ratio + val_ratio))

    # Split files
    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]

    # Copy files to respective directories
    splits = {
        'train': train_files,
        'valid': val_files,
        'test': test_files
    }

    for split_name, files in splits.items():
        for image_file in files:
            # Copy image
            src_image = os.path.join(source_dir, 'images', image_file)
            dst_image = os.path.join(output_dir, split_name, 'images', image_file)
            shutil.copy2(src_image, dst_image)

            # Copy corresponding label file
            label_file = image_file.rsplit('.', 1)[0] + '.txt'
            src_label = os.path.join(source_dir, 'labels', label_file)
            dst_label = os.path.join(output_dir, split_name, 'labels', label_file)

            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)

    print(f"Dataset split completed:")
    print(f"Train: {len(train_files)} files")
    print(f"Validation: {len(val_files)} files")
    print(f"Test: {len(test_files)} files")


# Set random seed for reproducibility
random.seed(42)

# Define paths
source_directory = r"E:\Workspace\Code\Python\CV_project\Garbage.v3i.yolov8\train"
output_directory = r"E:\Workspace\Code\Python\CV_project\data_v4"

# Split the dataset (70% train, 20% validation, 10% test)
split_dataset(source_directory, output_directory)