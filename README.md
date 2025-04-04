# Brain Tumor Detection

This is a passion project aimed at detecting brain tumors from MRI images using deep learning. I built this project to explore computer vision and neural networks while solving a real-world medical imaging problem. The goal is simple: classify MRI images into two classes – "Yes" (tumor present) and "No" (tumor absent).

## Project Overview

I started by downloading a dataset from Kaggle. Then I pre-processed the images by converting them into a standard format (JPEG), resizing them to 224x224 pixels, and organizing them into folders. Once the data was ready, I split it into training, validation, and test sets so that the model’s performance can be properly evaluated.

For the modeling part, I used MobileNetV2 with transfer learning. The training was done in two phases:

1. **Phase 1:** Train the new classification head on top of the frozen MobileNetV2 base.
2. **Phase 2:** Fine-tune the last 20 layers of the base model with a reduced learning rate for improved performance.

To make the model more robust, I also applied data augmentation, balanced the classes using class weights, and used callbacks like EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau during training.


## How to Run the Project

1. **Pre-process the Images:**  
   Run `process_images.py` to convert and resize all images, ensuring they have a consistent format and size.

2. **Split the Dataset:**  
   Execute `split_dataset.py` to divide the processed images into training, validation, and test sets.

3. **Train the Model:**  
   Run `train_model_improved.py` to train the model in two phases. This script will:
   - Train a classifier on top of MobileNetV2 with the base frozen (Phase 1).
   - Fine-tune the last layers of MobileNetV2 with a lower learning rate (Phase 2).
   - Save the best models and training histories in the `models/` folder.

4. **Analyze the Results:**  
   Open the `analyze_results.ipynb` notebook (located in the `notebooks/` folder) to view the training metrics, confusion matrix, and classification report.

## Final Thoughts

This project is both a learning experience and a step toward practical applications of deep learning in medical imaging. The current results are promising, but there is always room for improvement—whether by fine-tuning the model further, enhancing data augmentation, or even expanding the dataset. I hope you find this project interesting and that it serves as a useful resource for anyone looking to get into deep learning for medical applications.

Happy coding!


