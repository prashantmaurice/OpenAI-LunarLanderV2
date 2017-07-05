import random
from random import randint
from neuralNet import TwoLayerNeuralNet
import gym
from statistics import median, mean
import numpy as np
import utils

env = gym.make('LunarLander-v2')
neuralNet = TwoLayerNeuralNet(8,20,6,4)

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
observations_data = []  # list of all observations


def invertData(arr):
    arr2 = []
    for a in arr:
        if a == 0:
            arr2.append(1)
        else:
            arr2.append(0)
    return arr2


def formTrainX(prev_observation_old, prev_observation):
    # print("TEST prev_observation_old",prev_observation_old)
    # print("TEST prev_observation", prev_observation)
    return np.concatenate((prev_observation_old, prev_observation))


def play_game(play_with_mutation=True):
    observation = env.reset()
    prev_observation = observation
    all_moves = []
    pos_moves = []
    mut_moves = []
    total_reward = 0
    mutation = random.random()
    # print("mutation",mutation)
    for t in range(RANDOM_MAX_TIME):
        should_mutate = play_with_mutation and (random.random() < mutation)
        # should_mutate = should_mutate or (random.random() > 0.9)
        prev_observation = observation

        if should_mutate:
            random_action = env.action_space.sample()
        else:
            random_action = neuralNet.predict(prev_observation)[0]
        observation, reward, done, info = env.step(random_action)
        total_reward += reward
        output = {
            0: [1, 0, 0, 0],
            1: [0, 1, 0, 0],
            2: [0, 0, 1, 0],
            3: [0, 0, 0, 1]
        }[random_action]

        all_moves.append([prev_observation, output])
        if reward>0: pos_moves.append([prev_observation, output])
        if should_mutate: mut_moves.append([prev_observation, output])

        if done:
            return ( True, total_reward, all_moves, pos_moves, mut_moves)
    return ( False, total_reward-400, all_moves, pos_moves, mut_moves)


def learn_from_episodes(num_episodes):
    current_best = -10000
    average_reward = 0
    average_reward_count = 0

    average_reward_moving100 = 0
    moving_average_delay = 100

    all_result = []
    for i_episode in range(num_episodes):
        play_with_learn = i_episode % 100 != 0  # every 100th game play with perfect model
        result = play_game(play_with_learn)
        done = result[0]
        total_reward = result[1]
        all_moves = result[2]
        pos_moves = result[3]
        mut_moves = result[4]
        all_result.append(result)

        # reset average every 200 batch
        if i_episode%100==0:
            average_reward = 0
            average_reward_count = 0

        # Calculate average reward
        scores.append(total_reward)
        average_reward = ((average_reward*average_reward_count)+total_reward)/(average_reward_count+1)
        average_reward_count = average_reward_count + 1
        # if i_episode>moving_average_delay:
        #     average_reward_moving100 = average_reward_moving100 + total_reward - scores[i_episode-moving_average_delay]
        print("average_reward",i_episode,average_reward,average_reward_moving100,total_reward)
        if done:
            # print("total_reward", total_reward, current_best, len(pos_moves))
            if not play_with_learn:
                fairScores.append(total_reward)
            if total_reward > average_reward:
                # neuralNet.learn(np.array([i[0] for i in all_moves]), np.array([i[1] for i in all_moves]))
                neuralNet.learn(np.array([i[0] for i in pos_moves]), np.array([i[1] for i in pos_moves]))
                # neuralNet.learn(np.array([i[0] for i in mut_moves]), np.array([i[1] for i in mut_moves]))

            # if i_episode%200 == 0:
            #     print("Retraining network")
            #     # reset network
            #     neuralNet = TwoLayerNeuralNet(8, 20, 6, 4)
            #
            #     # retrain from top x percentile
            #     all_result.sort(key=lambda x: x[1],reverse=True)
            #     print("RELEARN",[x[1] for x in all_result])
            #     for k in range(int(len(all_result)/3)):
            #         all_moves_temp = all_result[k][2]
            #         pos_moves_temp = all_result[k][3]
            #         neuralNet.learn(np.array([i[0] for i in all_moves_temp]), np.array([i[1] for i in all_moves_temp]))



    # Plot scores
    utils.plotMultiSeries([scores, fairScores])
    print("Done")
    print("Scores", scores)
    print("Average score", mean(scores))
    print("Median score", median(scores))
    print("Training Set size", len(training_set))
    return average_reward


def mutate(random_action):
    return random_action


def test_already_trained_model(num_episodes):
    # Test if model is working
    for i_episode in range(num_episodes):
        observation = env.reset()
        prev_observation = observation
        train_new = []
        for t in range(TEST_MAX_TIME):
            env.render()
            prev_observation = observation
            # print("Prediction",dql.predict(observation))
            if random.random() > 0.1:
                random_action = neuralNet.predict(prev_observation)[0]
            else:
                random_action = env.action_space.sample()
            # print("random_action",random_action)
            observation, reward, done, info = env.step(random_action)
            if done:
                print("Finished in {} steps after {} episodes".format(t, i_episode))
                break


def random_without_training():
    observation = env.reset()
    for _ in range(1000):
        env.render()
        action = env.action_space.sample()  # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        print(observation)
        if done:
            env.reset()

    env.close()


def learn_from_episodes2(num_episodes):
    observation = env.reset()
    for _ in range(1000):
        env.render()
        action = env.action_space.sample()  # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        print(observation)
        if done:
            env.reset()

    env.close()

# MAIN PROCESS
# random_without_training()
learn_episodes = 1000
# neuralNet.loadState("ep{}_reward{}".format(learn_episodes,-240))
average_reward = learn_from_episodes(learn_episodes)
neuralNet.saveState("ep{}_reward{}".format(learn_episodes,int(average_reward)))
# learn_from_episodes2(10)
test_already_trained_model(10)
