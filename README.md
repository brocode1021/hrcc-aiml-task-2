Handwritten Digit Recognition

Welcome to the Handwritten Digit Recognition project. This is a classic and informative introduction to the world of machine learning and computer vision.

This simple Python script trains a neural network to recognize and identify handwritten digits from 0 to 9. It uses the famous MNIST dataset, which is a foundational resource for image recognition tasks.

The entire process runs in your terminal, showing you each step of the journey, from loading the data to seeing the final accuracy score.

What Does This Script Do?

This project follows a clear, step-by-step machine learning workflow:

Loads the Data: It automatically downloads the MNIST dataset, which contains thousands of examples of handwritten digits.

Prepares the Images: It normalizes the images, a simple technique that helps the model learn more effectively.

Builds a Model: It constructs a simple neural network using TensorFlow and Keras. The network has an input layer, a hidden layer to find patterns, and an output layer to make its final guess.

Trains the Model: The script shows the model thousands of examples and teaches it how to associate each image with the correct digit.

Checks Its Work: After training, it tests the model on a new set of images it has never seen before and calculates its accuracy.

Visualizes Predictions: To finish, it displays a few random test images and shows you what the model predicted versus what the real answer was.

Technologies Used

This project is built with a few key, powerful libraries:

Python: The core programming language.

TensorFlow/Keras: For building and training the neural network.

Matplotlib: To create the visual plot of the predictions.

NumPy: For efficient handling of the numerical data.

Getting Started

To run the project yourself, follow these simple steps.

Prerequisites

Make sure you have Python 3 installed on your system. You can check by running python --version or python3 --version in your terminal.

1. Set Up the Project

It is a good practice to run the project in a virtual environment. This keeps all the necessary libraries neatly contained.

code

Bash

# Create a virtual environment

python3 -m venv venv

# Activate it
# On Windows:
# venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

2. Install the Libraries

Install all the required libraries using the requirements.txt file.

code

Bash

pip install -r requirements.txt

3. Run the Script

That completes the setup. Now, you can run the main script.

code

Bash

python recognize_digits.py

What to Expect

When you run the script, you will see a series of messages in your terminal updating you on the progress:

It will confirm that the dataset is loaded.

It will print a summary of the model's architecture.

It will show the training progress for 5 rounds (epochs).

It will display the final Test Accuracy (usually around 97-98%).

Finally, a new window will open, showing 5 sample digits and the model's predictions, so you can see how well it performs.
