from tensorflow import keras
import numpy as np

model = keras.models.load_model('my_model_NAND.h5')


#print(model.predict([[0, 0]]))
#print(model.predict([[1, 0]]))
#print(model.predict([[0, 1]]))
#print(model.predict([[1, 1]]))

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[1],[1],[1],[0]])

#print(model.predict([[1, 1]])[0])

#print(int(np.round(model.predict([[0,1]])[0],0)))

#loss = 100 - (model.evaluate(x,y) * 10)
print(model.evaluate(x,y))

#score = model.evaluate(X_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
