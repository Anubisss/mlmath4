import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('./model')
model.summary()

operators = ['+', '-', '/', '*']
operatorsDict = {operator: i for i, operator in enumerate(operators)}

inputs = np.array([
  [4, 5, operatorsDict['+']],
  [0, 3, operatorsDict['+']],
  [-2, -5, operatorsDict['+']],
  [13, 9, operatorsDict['-']],
  [-6, 5, operatorsDict['-']],
  [-6, -8, operatorsDict['-']],
  [10, 10, operatorsDict['*']],
  [0, 10, operatorsDict['*']],
  [-3, -4, operatorsDict['*']],
  [-2, 1, operatorsDict['*']],
  [10, 1, operatorsDict['/']],
  [0, 10, operatorsDict['/']],
  [4, 4, operatorsDict['/']],
  [8, 2, operatorsDict['/']],
  [9, 3, operatorsDict['/']],
  [10, 4, operatorsDict['/']]
])
print(np.round(model.predict(inputs)))
