"""

Excercise 5: reinforcement learning

-----

Actions:
- 0: move south
- 1: move north
- 2: move east
- 3: move west
- 4: pick up passenger
- 5: drop off passenger

-----

Algorithm (lecture slides):

Do forever:
• Select an action a and execute it
• Receive immediate reward r
• Observe the new state s0
• Update the table entry for Qˆ (s, a)
• s ← s0


"""

###########################################
# Preset
###########################################

import gym
import random
import numpy as np
import time

env = gym.make("Taxi-v2")
next_reward = 0 * np.ones((500, 6))

###########################################
# Training
###########################################

# https://en.wikipedia.org/wiki/Q-learning#Algorithm

rounds = 1000
gamma = 0.6   # discount factor for future rewards

for episode in range(rounds):

    state = env.reset()
    done = False

    while not done:
        action = np.argmax(next_reward[state])
        new_state, reward, done, info = env.step(action)

        # Q(s,a) = r + gamma * max_{a'}[Q_{n-1}(s',a')]
        next_reward[state, action] = reward + gamma * np.max(next_reward[new_state])

        state = new_state

###########################################
# Testing
###########################################

rewards = [10]
actions = [10]

for run in range(1, 11):
    test_tot_reward = 0
    test_tot_actions = 0
    past_observation = -1

    observation = env.reset()

    for t in range(50):
        test_tot_actions = test_tot_actions + 1
        action = np.argmax(next_reward[observation])

        if observation == past_observation:
            # This is done only if gets stuck
            action = random.sample(range(0, 6), 1)
            action = action[0]

        past_observation = observation
        observation, reward, done, info = env.step(action)
        test_tot_reward = test_tot_reward + reward

        env.render()
        time.sleep(0.5)

        if done:
            break

    rewards.append(test_tot_reward)
    actions.append(test_tot_actions)

    print("Run " + str(run))
    print("Total reward: ")
    print(test_tot_reward)

    print("Total actions: ")
    print(test_tot_actions)
    print("-----------------")

print("Mean of total rewards: ")
print(np.round(np.mean(rewards), 1))

print("Mean of total actions: ")
print(np.round(np.mean(actions), 1))
print("-----------------")
