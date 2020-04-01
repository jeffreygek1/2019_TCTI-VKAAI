
"""
Algebraic representation
Used the Keras MMIST example as inspiration: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
The network should exists out of:
- 784 inputs (in case of a 28 by 28 pixel image)
- 10 outputs (one of each digit/class)
The network may be used without hidden layers.
We've tested various type of layers, dropouts and activation functions as shown in the table below:
+----------------------------+-----------------+------------------+-------+----------+
| Layers (excl. input layer) |     Dropout     | Layer Activation | Loss  | Accuracy |
+----------------------------+-----------------+------------------+-------+----------+
| 3 (MLP)                    | 0.25, 0.5, 0.75 | ReLU             | 2.658 |    0.834 |
| 2 (MLP)                    | 0.25, 0.5       | ReLU             | 1.178 |    0.921 |
| 2 (MLP)                    | 0.25, 0.25      | ReLU             | 1.104 |    0.930 |
| 1 (SLP)                    | 0.25            | sigmoid          | 0.185 |    0.944 | <-- BEST PERFORMANCE/ACCURACY RATIO
| 2 (MLP)                    | 0.25, 0.25      | tanh             | 0.171 |    0.947 |
| 2 (MLP)                    | 0.25, 0.25      | sigmoid          | 0.163 |    0.949 | <-- BEST ACCURACY
+----------------------------+-----------------+------------------+-------+----------+
We can conclude that the use of the Sigmoid activation in combination with one hidden layer and a dropout of gives the best accuracy.
However, one may consider using a single layer network for better performance and slightly worse accuracy. Note that in all tests softmax is used
as activation function for the output layer. Note that this implementation will most likely not be the most optimal solution to the problem
(if this would even be possible), since we used a limited amount of test variable combinations. (Amt. of layers, different activation functions etc.)
to test this solution.
Note: RMSprop is used for optimalisation and resulted in a time decrease higher than 90%.
"""
from __future__ import print_function
import keras

# Import dependencies.

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras import backend as K

batchSize = 128
numClasses = 10
epochs = 12
layerActivation = 'sigmoid'  # Other options are for example relu and sigmoid

# Input image dimensions.
imgRows, imgCols = 28, 28

# Split data between train and test sets
(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

# Reshape to the amount of pixels, for example: 28 * 28 = 784. These are the input neurons.
xTrain = xTrain.reshape(len(xTrain), imgRows * imgCols)
xTest = xTest.reshape(len(xTest), imgRows * imgCols)

yTrain = keras.utils.to_categorical(yTrain, numClasses)
yTest = keras.utils.to_categorical(yTest, numClasses)

# Initialize the sequential neural network.
model = Sequential()
model.add(Dense(128, activation=layerActivation,
                input_shape=((imgRows * imgCols),)))
model.add(Dropout(0.25))
model.add(Dense(numClasses, activation='softmax'))

# Generate the underlying TensorFlow model.
model.compile(loss=keras.losses.categorical_crossentropy,
              # Use RMSprop optimizer for faster computation.
              optimizer=RMSprop(),
              metrics=['accuracy'])

# Train the model.
model.fit(xTrain, yTrain,
          batch_size=batchSize,
          epochs=epochs,
          verbose=1,
          validation_data=(xTest, yTest))

# Display the results upon the user's screen.
score = model.evaluate(xTest, yTest, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])