import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

from keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, GlobalAveragePooling2D, Dense, add

# Function to create inverted residual blocks
def inverted_residual_block(x, expansion_rate, out_channels, strides):
    input_channels = x.shape[-1]
    
    input_data = x
    
    # Expansion Layer
    x = Conv2D(filters = input_channels * expansion_rate, kernel_size = (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU(6)(x)
    
    # Depthwise Convolution Layer
    x = DepthwiseConv2D(kernel_size = (3, 3), strides = strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU(6)(x)
    
    # Projection Layer
    x = Conv2D(filters = out_channels, kernel_size = (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    
    # Shortcut Connection|
    if x.shape[-1] == input_data.shape[-1]:
        x = add([x, input_data])
    
    return x

# Function to create MobileNetV2 feature extraction model
def FeatureExtraction (input_shape):
    model_input = Input(input_shape)

    # Initial Convolution
    x = Conv2D(filters = 32, kernel_size = (3, 3), strides = (2, 2), padding='same')(model_input)
    x = BatchNormalization()(x)
    x = ReLU(6)(x)

    # Inverted Residual Blocks
    x = inverted_residual_block(x, 1, 16, (1, 1))

    x = inverted_residual_block(x, 6, 24, (2, 2))
    x = inverted_residual_block(x, 6, 24, (1, 1))

    x = inverted_residual_block(x, 6, 32, (2, 2))
    x = inverted_residual_block(x, 6, 32, (1, 1))
    x = inverted_residual_block(x, 6, 32, (1, 1))

    x = inverted_residual_block(x, 6, 64, (2, 2))
    x = inverted_residual_block(x, 6, 64, (1, 1))
    x = inverted_residual_block(x, 6, 64, (1, 1))
    x = inverted_residual_block(x, 6, 64, (1, 1))

    x = inverted_residual_block(x, 6, 96, (1, 1))
    x = inverted_residual_block(x, 6, 96, (1, 1))
    x = inverted_residual_block(x, 6, 96, (1, 1))

    x = inverted_residual_block(x, 6, 160, (2, 2))
    x = inverted_residual_block(x, 6, 160, (1, 1))
    x = inverted_residual_block(x, 6, 160, (1, 1))

    x = inverted_residual_block(x, 6, 320, (1, 1))

    # Final Convolution
    x = Conv2D(filters = 1280, kernel_size = (1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU(6)(x)

    # Pooling Layer
    x = GlobalAveragePooling2D()(x)

    # Dense LayerD
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4, activation='softmax')(x)

    model = Model(inputs = model_input, outputs = x)
    
    return model

class FaceRecognition:

    # Define hyperparameters
    def __init__(self):
        self.EPOCHS = 65
        self.BATCH_SIZE = 32
        self.NUMBER_OF_TRAINING_IMAGES = len(train_images)
        self.IMAGE_HEIGHT = 224
        self.IMAGE_WIDTH = 224
        self.model = model

    @staticmethod
    def plot_training(history):
        # Plot training and validation accuracy
        plot_folder = "plot"
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.1, 1])
        plt.legend(loc='lower right')

        # Save plot as image
        if not os.path.exists(plot_folder):
            os.mkdir(plot_folder)
        plt.savefig(os.path.join(plot_folder, "model_accuracy.png"))

    def training(self):

        # Compile the model
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=optimizers.SGD(learning_rate = 1e-4, momentum=0.9),
            metrics=["accuracy"]
        )

        # Train the model
        history = self.model.fit(
            train_images,
            train_class,
            steps_per_epoch=self.NUMBER_OF_TRAINING_IMAGES//self.BATCH_SIZE,
            epochs=self.EPOCHS,
            validation_split=0.2
        )

        # Plot and save the training history
        FaceRecognition.plot_training(history)

    # Save the trained model
    def save_model(self, model_name):
        self.model.save(model_name)

input_shape = (224, 224, 3)

# Create the model
model = FeatureExtraction(input_shape)
model.summary()

# Load the training data
train = pickle.load(open('/kaggle/input/dataset14/Train.pickle', 'rb'))
train_images, train_class = zip(*train)

# Preprocess the training images
train_images = np.array(train_images)
train_class = np.array(train_class)

train_images = train_images / 255.0

# Define the name of the saved model
model_name = 'MobileNetV2Model.h5'

# Initialize FaceRecognition class and train the model
face_recognition = FaceRecognition()
face_recognition.training()
face_recognition.save_model(model_name)