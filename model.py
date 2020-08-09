import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout,BatchNormalization
from sklearn.model_selection import train_test_split
try:
	import cPickle as pickle
except:
	import pickle

data = np.load('training_images.npz', allow_pickle=True)
images = data['images']
pts = data['points']
pts = np.reshape(pts,[2811,136,-1])
pts = np.squeeze(pts)
X_train,X_val,Y_train,Y_val = train_test_split(images,pts, test_size = 0.1)

def model_face():
	model = Sequential()
	# l1
	model.add(Conv2D(32,(3,3), strides = (1,1), padding = 'valid', activation = 'relu', input_shape = (250,250,3)))
	model.add(MaxPool2D(pool_size = (2,2), strides = (2,2), padding = 'valid'))
	# l2
	model.add(Conv2D(64,(3,3), strides = (1,1), padding = 'valid', activation = 'relu'))
	model.add(Conv2D(64,(3,3), strides = (1,1), padding = 'valid', activation = 'relu'))
	model.add(MaxPool2D(pool_size = (2,2), strides = (2,2), padding = 'valid'))
	# l3
	model.add(Conv2D(64,(3,3), strides = (1,1), padding = 'valid', activation = 'relu'))
	model.add(Conv2D(64,(3,3), strides = (1,1), padding = 'valid', activation = 'relu'))
	model.add(MaxPool2D(pool_size = (2,2), strides = (2,2), padding = 'valid'))
	# l4
	model.add(Conv2D(128,(3,3), strides = (1,1), padding = 'valid', activation = 'relu'))
	model.add(Conv2D(128,(3,3), strides = (1,1), padding = 'valid', activation = 'relu'))
	model.add(MaxPool2D(pool_size = (2,2), strides = (1,1), padding = 'valid'))
	# l5
	model.add(Conv2D(256,(3,3), strides = (1,1), padding = 'valid', activation = 'relu'))
	# l6
	model.add(Flatten())
	model.add(Dense(1024,activation = "relu",use_bias=True))
	model.add(Dense(68*2,use_bias=True))

	return model

model = model_face()
model.compile(optimizer='adam',loss = 'mean_squared_error', metrics = ['mse'])
hist = model.fit(X_train,Y_train,verbose = 1,epochs = 500,validation_data=(X_val,Y_val))
with open('trainHistoryDict', 'wb') as file_pi:
	pickle.dump(hist.history, file_pi)
hist_df = pd.DataFrame(hist.history)
hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
	hist_df.to_csv(f)

acc = model.evaluate(X_val,Y_val)
print("Loss is {}".format(acc[0]))
model.save('kk')