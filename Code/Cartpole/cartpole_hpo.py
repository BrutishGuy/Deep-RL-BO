from __future__ import print_function

import numpy as np
import gym

from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers import  Convolution1D, MaxPooling1D, Flatten, LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
from keras.optimizers import Adam

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


def data():
    """
    Data providing function:

    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.

    Return:
        x_train, y_train, x_test, y_test
    """
    return [],[],[],[]


def model(x_train, y_train, x_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    ENV_NAME = 'CartPole-v0'


    # Get the environment and extract the number of actions.
    env = gym.make(ENV_NAME)
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n

    model = Sequential()
    #model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(LSTM({{choice([16,32,48,64])}}, input_shape=(1,) + env.observation_space.shape))
    model.add(Dropout({{uniform(0, 1)}})) # if shit commment this out
    model.add(Activation('tanh'))
    model.add(Dense({{choice([16,32,48,64])}}))
    model.add(Dropout({{uniform(0, 1)}})) # if shit comment this out
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    # comment this out
    if conditional({{choice(['three', 'four'])}}) == 'four':
        model.add(Dense({{choice([16,32,48,64])}}))
        model.add({{choice([Dropout(0.5), Activation('linear')])}})
        model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    if conditional({{choice(['four', 'five'])}}) == 'five':
        model.add(Dense({{choice([16,32,48,64])}}))
        model.add({{choice([Dropout(0.5), Activation('linear')])}})
        model.add(Activation({{choice(['relu', 'sigmoid'])}}))
        
    model.add(Dense(nb_actions))
    model.add(Activation({{choice(['relu', 'sigmoid','linear'])}}))
    #print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr={{uniform(1e-4,1e-2)}}), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    number_tasks = 1000#0 # number of steps change me please
    dqn.fit(env, nb_steps=number_tasks, visualize=False, verbose=2)

    hist = dqn.test(env, nb_episodes=5, visualize=True)
    rewards = hist.history['episode_reward']

    loss = 1.0/len(rewards) * np.sum([(90.0 - reward)**2 for reward in rewards])

    print('Test Loss:', loss)
    return {'loss': -loss, 'status': STATUS_OK, 'model': dqn}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=50,
                                          trials=Trials())
    #X_train, Y_train, X_test, Y_test = data()
    #print("Evalutation of best performing model:")
    #print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
