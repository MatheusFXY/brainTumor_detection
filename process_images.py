import os
from PIL import Image

input_dirs = {
    "Yes": "dataset/yes",
    "No": "dataset/no"
}


output_base = "processed_dataset"
output_size = (224, 224)

if not os.path.exists(output_base):
    os.makedirs(output_base)

for label, input_dir in input_dirs.items():
    output_dir = os.path.join(output_base, label)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_dir, filename)
            try:
                img = Image.open(input_path)
                img = img.convert('RGB')
                img_resized = img.resize(output_size, Image.Resampling.LANCZOS)
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_dir, base_name + ".jpg")
                img_resized.save(output_path, format='JPEG', quality=95)
                print(f"Processado: {input_path} -> {output_path}")
            except Exception as e:
                print(f"Erro ao processar {input_path}: {e}")
