import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("--- Handwritten Digit Recognition (Python + TensorFlow/Keras) ---")

    print("\n[Step 1] Loading the MNIST dataset from Keras...")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("Dataset loaded successfully.")
    print(f"   - Training images: {x_train.shape[0]}")
    print(f"   - Testing images:  {x_test.shape[0]}")

    print("\n[Step 2] Normalizing pixel values to improve model training...")
    x_train, x_test = x_train / 255.0, x_test / 255.0
    print("Normalization complete.")

    print("\n[Step 3] Building the neural network model...")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    print("Model built and compiled successfully.")
    model.summary()

    print("\n[Step 4] Training the model on the training set...")
    print("This will take a moment...")
    model.fit(x_train, y_train, epochs=5)
    print("Training finished.")

    print("\n[Step 5] Evaluating the model's accuracy on the test set...")
    val_loss, val_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\n   >>> Test Accuracy: {val_acc * 100:.2f}% <<<")
    
    print("\n[Step 6] Making predictions on a few test images...")
    print("A plot will appear showing the digit and the model's prediction.")
    
    num_predictions = 5
    random_indices = np.random.choice(x_test.shape[0], num_predictions, replace=False)
    
    for i, index in enumerate(random_indices):
        image = x_test[index]
        true_label = y_test[index]
        
        prediction_result = model.predict(np.expand_dims(image, axis=0))
        predicted_label = np.argmax(prediction_result)
        
        plt.subplot(1, num_predictions, i + 1)
        plt.imshow(image, cmap=plt.cm.binary)
        plt.title(f"Pred: {predicted_label}\nTrue: {true_label}")
        plt.axis('off')
        
    plt.suptitle("Model Predictions vs. True Labels", fontsize=16)
    plt.show()


if __name__ == "__main__":
    main()