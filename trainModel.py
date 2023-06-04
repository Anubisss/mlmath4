import numpy as np
import tensorflow as tf
import pandas as pd

operators = ['+', '-', '/', '*']
operatorsDict = {operator: i for i, operator in enumerate(operators)}

trainData = pd.read_excel('./trainData/train.xlsx').dropna()
trainRows = trainData.iloc[:, 0:10000].values
trainInputData = trainRows[:, 0:3]
trainInputData[:, 2] = np.array([operatorsDict[string] for string in trainInputData[:, 2]])
trainOutputData = trainRows[:, 3]

testData = pd.read_excel('./trainData/test.xlsx').dropna()
testRows = testData.iloc[:, 0:3000].values
testInputData = testRows[:, 0:3]
testInputData[:, 2] = np.array([operatorsDict[string] for string in testInputData[:, 2]])
testOutputData = testRows[:, 3]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(500, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(np.array(trainInputData, dtype=np.float32), np.array(trainOutputData, dtype=np.float32), epochs=200)

loss = model.evaluate(np.array(testInputData, dtype=np.float32), np.array(testOutputData, dtype=np.float32))
print('Test loss:', loss)

model.save('./model')
