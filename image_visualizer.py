import numpy as np
import matplotlib.pyplot as plt
import config
from data_handler import DataHandler


def load_data():
    # Load the Fashion MNIST dataset
    data_handler = DataHandler(config.DATA_PATH, config.DATA_PATH_TEST)
    data_handler.load_data()

    return data_handler.X_train, data_handler.y_train, data_handler.X_test, data_handler.y_test


def display_image(images, labels, index):
    # Extract a row and reshape to 28x28 image
    image_array = images.iloc[index].values.reshape(28, 28)
    plt.imshow(image_array, cmap='gray')
    plt.colorbar()
    plt.grid(False)

    # Add the image label as the title
    plt.title(f'Image Label: {labels.iloc[index]}')
    plt.show()


def main():
    train_images, train_labels, test_images, test_labels = load_data()

    # # Display the first image from the training dataset
    display_image(train_images, train_labels, 655)
    # print(train_images)


if __name__ == '__main__':
    main()
