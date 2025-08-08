import os
import numpy as np
import cv2
import albumentations as alb
import random
import pickle

# Path to data Directories
train_dir = os.path.join('dataset', 'training')

# Data Augmentation Pipeline
image_augmentor = alb.Compose([
    alb.Resize(height=224, width=224),
    alb.HorizontalFlip(p=0.5),
    alb.RandomBrightnessContrast(p=0.2),
    alb.Rotate(limit=30, p=0.5),
    alb.RandomGamma(p=0.2),
    alb.RandomShadow(p=0.2)
])

# Dictionary holding labels for different classes
labels = {
    'cristiano_ronaldo' : 0,
    'linonel_messi' : 1,
    'mujtaba_butt': 2,
    'neymar_jr' : 3,
    'abdul_abbasi' : 4
}

# Initialise lists to store images and corresponding class labels
train_images = []
train_class = []
counter = 0

# Iterate over each person's directory in the training dataset
for person in ['cristiano_ronaldo', 'linonel_messi', 'mujtaba_butt', 'neymar_jr', 'abdul_abbasi']:
    # Iterate over each image in the current person's directory
    for image in os.listdir(os.path.join(train_dir, person)):
        # Read the image
        image_path = os.path.join(train_dir, person, image) 
        img = cv2.imread(image_path)

        # Determine the augmentation count based on the person
        if person == 'mujtaba_butt':
            counter = 9
        else:
            counter = 8
        
        # Apply augmentation to each image and store the augmented images along with their label
        for i in range(counter):
            aug_img = image_augmentor(image = np.array(img))
            train_images.append(aug_img['image'])
            train_class.append(labels[person])

# Shuffle the dataset
indices = list(range(len(train_images)))
random.shuffle(indices)

train_images = [train_images[i] for i in indices]
train_class = [train_class[i] for i in indices]

# Zip the images and class labels together
train = zip(train_images, train_class)

# Save the zipped dataset to a pickle file
pickle_out = open('Train.pickle', 'wb')
pickle.dump(train, pickle_out)
pickle_out.close()

