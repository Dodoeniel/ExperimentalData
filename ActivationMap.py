import numpy as np
import pickle
from keras.models import load_model
from keras.models import Model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import Dropout
from keras.layers import Input
from Libraries import data_preprocessing as pp

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.cm as cmx


name = 'WindowedData_Set1_Model_2_model.h5'
dataName = 'w11s_11hs_1051.p'
PATH = '/media/computations/DATA/ExperimentalData/Runs/116926/'
model = load_model(PATH + name)
layer = model.get_layer('lstm_2')
weights = layer.get_weights()

inputs1 = Input(shape=(None, 8))
lstm1, state_h, state_c = LSTM(8, return_sequences=True, return_state=True)(inputs1)
model = Model(inputs=inputs1, outputs=[lstm1, state_h, state_c])
model.set_weights(weights)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


Data = pickle.load(open(PATH + dataName, 'rb'))
dropChannels = ['time', 'stopId', 'trg1', 'n1', 'trot1', 'tlin1', 'tlin2', 'tamb1']
X = pp.shape_Data_to_LSTM_format(Data[0], dropChannels)
#yt, ht, ct = model.predict(X)

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
plt.ylim((0, 4))
plt.xlim((0, 9))
cm = plt.get_cmap('viridis')
cNorm  = colors.Normalize(vmin=0, vmax=1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

currX = X[0]
def animate(i):
    # get cell state at that time step (keras only gives the last one
    X_timestep = currX[0:i+1]
    yt, ht, ct = model.predict(X_timestep.reshape((1, X_timestep.shape[0], X_timestep.shape[1])))

    # get value from 0th data set, ith row (time step) and 0th feature
    for k in range(len(yt[0][i])):

        colorVal = scalarMap.to_rgba(yt[0][-1][k])
        circle1 = plt.Circle((1+k, 3), 0.2, color=colorVal)
        ax1.add_artist(circle1)
        colorVal = scalarMap.to_rgba(ct[0][k])
        circle1 = plt.Circle((1 + k, 2), 0.2, color=colorVal)
        ax1.add_artist(circle1)
        colorVal = scalarMap.to_rgba(ht[0][k])
        circle1 = plt.Circle((1 + k, 1), 0.2, color=colorVal)
        ax1.add_artist(circle1)


anim = animation.FuncAnimation(fig, animate, frames=np.arange(0, len(currX)), interval=10)
anim.save(PATH+'ActivationMap.gif', dpi=80, writer='imagemagick')
#def animate(i):

