# Fashion MNIST Classification Project

This project is designed to handle, process, and classify images from the Fashion MNIST dataset using machine learning models. It employs a structured approach with separate modules for data handling, model definition, and model training, ensuring that the code is modular, reusable, and easy to understand.

## Project Structure

- `data_handler.py`: Manages the loading and preprocessing of the Fashion MNIST dataset.
- `models.py`: Contains definitions of various machine learning models.
- `trainer.py`: Handles the training and evaluation of models.
- `config.py`: Stores configuration settings such as data paths and model parameters.
- `runner.ipynb`: The entry point of the program, orchestrating the data handling, modeling, and training processes.

## Features

- **Data Handling**: The `DataHandler` class efficiently manages data operations, including loading data from CSV files, normalizing pixel values, and splitting data into training and testing sets.
- **Model Definitions**: The `models.py` file includes several model classes such as `RandomForestModel` and others tailored to handle specific types of machine learning algorithms.
- **Training and Evaluation**: The `Trainer` class in `trainer.py` facilitates the training of models and computes various metrics to evaluate their performance, such as accuracy, precision, and recall.
- **Configuration Management**: `config.py` centralizes the configuration, making the system adaptable and easy to configure without altering the core codebase.
- **Execution**: `runner.ipynb` integrates all components, from data preparation through model training and evaluation.

## Usage

To run the project, follow these steps:

1. **Set up the environment**:
    - Ensure Python and necessary packages (`sklearn`, `pandas`, `numpy`) are installed.
    - Install dependencies: `pip install -r requirements.txt`.

2. **Configure paths and parameters**:
    - Modify `config.py` to point to the correct paths for the dataset and adjust any model parameters as needed.

3. **Run the project**:
    - Execute the script via the command line: `python main.py`.

## Requirements

- Python 3.6+
- scikit-learn
- pandas
- numpy
