import random
from random import randint
import neuralNet
import gym
from statistics import median, mean
import numpy as np
import utils
env = gym.make('CartPole-v1')

# Variables
RANDOM_EPISODES = 2000
RANDOM_MAX_TIME = 400
RANDOM_CUTOFF = 50

TEST_EPISODES = 50
TEST_MAX_TIME = 2000



# generate Random Training set
training_set = []
scores = []
fairScores = []
current_best = 10
observations_data = [] # list of all observations

def invertData(arr):
    arr2 = []
    for a in arr:
        if a == 0:
            arr2.append(1)
        else:
            arr2.append(0)
    return arr2

def formTrainX(prev_observation_old,prev_observation):
    # print("TEST prev_observation_old",prev_observation_old)
    # print("TEST prev_observation", prev_observation)
    return np.concatenate((prev_observation_old,prev_observation))

for i_episode in range(RANDOM_EPISODES):
    play_without_learn = i_episode%100 == 0 # every 100th game play with perfect model
    observation = env.reset()
    prev_observation = observation
    prev_observation2 = observation
    train_new = []

    for t in range(RANDOM_MAX_TIME):
        prev_observation2 = prev_observation
        prev_observation = observation

        if play_without_learn:
            random_action = neuralNet.predict(formTrainX(prev_observation2, prev_observation))[0]
        else:
            if i_episode<500:
                if (t > 0.5 * current_best):
                    epsilon = 0.95
                else:
                    epsilon = 0.8
            elif i_episode<2000:
                if( t > 0.5*current_best):
                    epsilon = 0.8
                else:
                    epsilon = 0.6
            else:
                if (t > 0.5 * current_best):
                    epsilon = 0.8
                else:
                    epsilon = 0.1
            if random.random()>epsilon:
                random_action = neuralNet.predict(formTrainX(prev_observation2, prev_observation))[0]
            else:
                random_action = env.action_space.sample()
        observation, reward, done, info = env.step(random_action)
        if random_action == 0:
            output = [1,0]
        else:
            output = [0,1]
        train_new.append([formTrainX(prev_observation2,prev_observation),output])


        if done:
            scores.append(t)
            if play_without_learn:
                fairScores.append(t)
            if t>(current_best*0.6):
                if current_best < t:
                    current_best = t
                    print("Current best",current_best)
                print("I:Finished in {} steps after {} episodes".format(t,i_episode))
                for k in train_new:
                    training_set.append(k)
                neuralNet.learn(np.array([i[0] for i in train_new]), np.array([i[1] for i in train_new]))
                if t > (current_best * 0.7):
                    neuralNet.learn(np.array([i[0] for i in train_new]), np.array([i[1] for i in train_new]))
                if t > (current_best * 0.9):
                    neuralNet.learn(np.array([i[0] for i in train_new]), np.array([i[1] for i in train_new]))

            # if current_best>100 and t<(current_best*0.3):
            #     # unlearn
            #     print("U:Finished in {} steps after {} episodes".format(t, i_episode))
            #     neuralNet.learn(np.array([i[0] for i in train_new]), np.array([invertData(i[1]) for i in train_new]))
            #
            # observations_data.append(train_new)






            break

# utils.plotSeries(scores)
# utils.plotSeries(fairScores)
utils.plotMultiSeries([scores,fairScores])
print("Done")
print("Scores",scores)
print("Average score",mean(scores))
print("Median score",median(scores))
print("Training Set size",len(training_set))


# Test if model is working
for i_episode in range(TEST_EPISODES):
    observation = env.reset()
    prev_observation = observation
    prev_observation2 = observation
    train_new = []
    for t in range(TEST_MAX_TIME):
        env.render()
        prev_observation2 = prev_observation
        prev_observation = observation
        # print("Prediction",dql.predict(observation))
        if random.random()>0.1:
            random_action = neuralNet.predict(formTrainX(prev_observation2, prev_observation))[0]
        else:
            random_action = env.action_space.sample()
        # print("random_action",random_action)
        observation, reward, done, info = env.step(random_action)
        if done:
            print("Finished in {} steps after {} episodes".format(t, i_episode))
            break


