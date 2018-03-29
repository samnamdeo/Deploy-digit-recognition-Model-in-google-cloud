import numpy as np
import keras.models
from scipy.misc import imread, imresize

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = keras.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded Model from disk")

# compile and evaluate loaded model
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# loss,accuracy = model.evaluate(X_test,y_test)
# print('loss:', loss)
# print('accuracy:', accuracy)
x = imread('output.png', mode='L')
x = np.invert(x)
x = imresize(x, (28, 28))
import matplotlib.pyplot as plt

plt.imshow(x)
plt.show()
# imshow(x)
x = x.reshape(1, 28, 28, 1)

out = loaded_model.predict(x)
print(out)
print(np.argmax(out, axis=1))
