import os
import shutil
import random
import yaml

# === Paths ===
BASE = 'datasets/train'
IMG_DIR = os.path.join(BASE, 'images')
LABEL_DIR = os.path.join(BASE, 'labels')
OUTPUT_BASE = 'dataset'  # changed from datasets_split

# === Class names ===
class_names = ['Hello', 'Yes', 'No', 'Thanks', 'ILoveYou', 'Please']

# === Split ratio ===
split_ratios = {'train': 0.7, 'val': 0.2, 'test': 0.1}

# === Create folders ===
def create_dirs():
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_BASE, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_BASE, split, 'labels'), exist_ok=True)

# === Split data ===
def split_data():
    images = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
    random.shuffle(images)

    total = len(images)
    train_end = int(split_ratios['train'] * total)
    val_end = train_end + int(split_ratios['val'] * total)

    split_map = {
        'train': images[:train_end],
        'val': images[train_end:val_end],
        'test': images[val_end:]
    }

    for split, files in split_map.items():
        for img_file in files:
            label_file = img_file.replace('.jpg', '.txt')

            shutil.copy2(os.path.join(IMG_DIR, img_file), os.path.join(OUTPUT_BASE, split, 'images', img_file))
            shutil.copy2(os.path.join(LABEL_DIR, label_file), os.path.join(OUTPUT_BASE, split, 'labels', label_file))

    print("✅ Dataset split complete.")

# === Generate data.yaml ===
def write_yaml():
    data_yaml = {
        'train': os.path.abspath(os.path.join(OUTPUT_BASE, 'train', 'images')),
        'val': os.path.abspath(os.path.join(OUTPUT_BASE, 'val', 'images')),
        'test': os.path.abspath(os.path.join(OUTPUT_BASE, 'test', 'images')),
        'nc': len(class_names),
        'names': class_names
    }

    with open('data.yaml', 'w') as f:
        yaml.dump(data_yaml, f)

    print("✅ data.yaml generated.")

# === Main ===
if __name__ == "__main__":
    create_dirs()
    split_data()
    write_yaml()
