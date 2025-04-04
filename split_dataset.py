import os
import random
import shutil

input_dir = "processed_dataset"
output_dir = "split_dataset"
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
classes = ['yes', 'no']

for split in ['train', 'val', 'test']:
    split_path = os.path.join(output_dir, split)
    if not os.path.exists(split_path):
        os.makedirs(split_path)
    for class_name in classes:
        class_split_path = os.path.join(split_path, class_name)
        if not os.path.exists(class_split_path):
            os.makedirs(class_split_path)

for class_name in classes:
    class_input_path = os.path.join(input_dir, class_name)
    images = [f for f in os.listdir(class_input_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)
    total = len(images)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count
    train_files = images[:train_count]
    val_files = images[train_count:train_count + val_count]
    test_files = images[train_count + val_count:]
    
    def copy_files(file_list, split_name):
        for file_name in file_list:
            src_path = os.path.join(class_input_path, file_name)
            dst_path = os.path.join(output_dir, split_name, class_name, file_name)
            shutil.copy(src_path, dst_path)
    
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')
    
    print(f"Classe '{class_name}': Total = {total}, Train = {len(train_files)}, Val = {len(val_files)}, Test = {len(test_files)}")
