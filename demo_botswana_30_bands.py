import random
import scipy.io as sio
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Nadam
import tensorflow as tf
from scipy.spatial.distance import euclidean, correlation
from scipy.stats import entropy

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)


### #0. hyper-parameters & data
nb_bands = 145
nb_exp_bands = 30
state_size = nb_bands
action_size = nb_bands
learning_rate = 0.0001

data = sio.loadmat('data4drl/data_botswana_drl.mat')
x = np.float32(data['x'])

IEs = np.zeros((nb_bands,))
for i in range(nb_bands):
    IEs[i] = entropy(x[:, i])
IEs = (IEs-np.min(IEs))/(np.max(IEs)-np.min(IEs))

### step #1. defining the agent (dqn)
class dqn4hsi(object):
    def __init__(self, state_size, action_size, learning_rate):
        self.state_size = state_size
        self.action_size = action_size
	self.memory = deque(maxlen = 50000)
	self.gamma = 0.99
	self.epsilon = 1.0
	self.epsilon_min = 0.01
	self.epsilon_decay = 0.995
	self.learning_rate = learning_rate
	self.train_batch = 100
	self._model = self._createModel()

    @property
    def model(self):
        return self._model

    def _createModel(self):
        model = Sequential()
	model.add(Dense(2*self.state_size, input_shape = (self.state_size,), activation = 'relu'))
        model.add(Dense(2*self.state_size, activation = 'relu'))
	model.add(Dense(self.action_size, activation = 'linear'))
	model.compile(loss = 'mse', optimizer = Nadam(lr = self.learning_rate))
		
	return model
	
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def predictAction(self, state):
        return self.model.predict(state)

    def act(self, state):
        state = state.reshape((self.state_size,))
	if random.random()<self.epsilon:
	    action_set = np.squeeze(np.argwhere(state==0))
	    return random.sample(action_set, 1)[0]
        else:
	    invalid_set = np.squeeze(np.argwhere(state>0))
	    state = state.reshape((1, self.state_size))
	    prob = self.predictAction(state)[0]
	    prob[invalid_set] = -99999
	    return np.argmax(prob)

    def replay(self):
        if len(self.memory)>=self.train_batch:
	    minibatch = random.sample(self.memory, self.train_batch)
	    state_batch = np.zeros((self.train_batch, self.state_size))
	    target_batch = np.zeros((self.train_batch, self.action_size))
	    for i, (state, action, reward, next_state, done) in enumerate(minibatch):
	        state_batch[i, :] = state
		target_batch[i, :] = self.predictAction(state)
		if done:
		    target_batch[i, action] = reward
		else:
		    next_prob = self.predictAction(next_state)[0]
		    next_state = next_state.reshape((self.state_size,))
		    invalid_set = np.squeeze(np.argwhere(next_state>0))
		    next_prob[invalid_set] = -99999
		    target_batch[i, action] = reward+self.gamma*np.argmax(next_prob)
	    history = self.model.fit(state_batch, target_batch, epochs = 1, verbose = 0)
	    loss = history.history['loss'][0]
			
	    if self.epsilon>self.epsilon_min:
	        self.epsilon*=self.epsilon_decay
				
	    return loss
	else:
	    print('just a moment...')
			
    def loadWeights(self, name):
        self.model.load_weights(name)

    def saveWeights(self, name):
        self.model.save_weights(name)
	

### step #2. inference
agent = dqn4hsi(state_size, action_size, learning_rate)
agent.loadWeights('models/qnet_botswana_30_bands.h5')

selected_bands = np.zeros((nb_exp_bands,))
state = np.float32(np.zeros((1, state_size)))
for t in range(nb_exp_bands):
    state = state.reshape((state_size,))
    invalid_set = np.squeeze(np.argwhere(state>0))
    state = state.reshape((1, state_size))
    pred = agent.predictAction(state)[0]
    pred[invalid_set] = -99999
    action = np.argmax(pred)
    selected_bands[t] = action
    state[0, action] = 1

    print("{}/{}".format(t, nb_exp_bands))

print selected_bands

sio.savemat('results/drl_30_bands_botswana.mat', {'selected_bands': selected_bands})
















