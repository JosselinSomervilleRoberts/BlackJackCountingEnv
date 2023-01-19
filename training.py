from playing_env import BlackJackPlayingEnv, OBS_PLAYER_SUM_IDX, OBS_DEALER_CARD_IDX, OBS_USABLE_ACE_IDX, OBS_CAN_SPLIT_IDX, OBS_CAN_DOUBLE_IDX, ACTION_SPLIT
from deck import DecksOfCards
from agent import QLearningAgent
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


def compute_effective_reward(data, gamma):
    """data is a list of step under the format: (state, action, reward, next_state, done)"""
    effective_reward = data[-1]["reward"]
    data[-1]["effective_reward"] = effective_reward
    last_done_rewards = []
    discount = 1
    for i in range(len(data) - 1, -1, -1):
        if data[i]["done"]:
            last_done_rewards.append(data[i]["reward"])
            discount = 1
            data[i]["effective_reward"] = last_done_rewards[-1]
        elif data[i]["action"] != 3: # not a split
            discount *= gamma
            data[i]["effective_reward"] = data[i]["reward"] + last_done_rewards[-1] * discount
        else: # a split
            reward = data[i]["reward"]
            if reward < 0: # illegal split
                data[i]["effective_reward"] = reward
                last_done_rewards.append(reward)
            elif int(reward) == 6: # double blackjack
                data[i]["effective_reward"] = reward
                last_done_rewards.append(reward)
            elif int(reward) == 3: # blackjack
                last_done_rewards[-1] += reward
                data[i]["effective_reward"] = last_done_rewards[-1]
            else: # no blackjack
                last_reward = last_done_rewards.pop(-1)
                last_done_rewards[-1] += last_reward
                data[i]["effective_reward"] = data[i]["reward"] + last_done_rewards[-1] 
            discount = 1

if __name__ == "__main__":
    N_EPISODES = 2500000
    def learning_rate(n):
        return 0.0001 + 0.02 * (max(0, N_EPISODES - n) / N_EPISODES) ** 2
    def epsilon_func(n):
        return 0.3 * (max(0, 0.5*N_EPISODES - n) / N_EPISODES) ** 2

    rules = {
                "double_allowed": False,
                "split_allowed": False,
                "surrender_allowed": True,
                "dealer_hit_soft_17": True,
                "resplit_allowed": True,
                "blackjack_payout": 1.5,
                "floor_finished_reward": True
            }
    decks = DecksOfCards(nb_decks=6, fraction_not_in_play=0.2)
    env = BlackJackPlayingEnv(decks = decks, rules=rules)
    agent = QLearningAgent(rules, epsilon_func=epsilon_func, delta = 0, discount_factor=0.5, learning_rate=learning_rate)

    
    rewards = np.zeros(N_EPISODES, dtype=int)
    prev_cum_reward = 0
    cum_rewards = np.zeros(N_EPISODES, dtype=int)

    for i in tqdm(range(N_EPISODES)):
        state, info = env.reset()

        # The player or the dealer has a natural blackjack
        if "done" in info and info["done"]:
            prev_cum_reward += info["reward"]
            cum_rewards[i] = prev_cum_reward
            rewards[i] = info["reward"]
            continue

        # Play all the actions until the game is finished
        data = []
        sum_reward = 0
        done = False
        while not done:
            action = agent.get_legal_action(state, learning=True)
            new_state, reward, done, info = env.step(action)
            sum_reward += reward

            if reward < -5: # illegal action
                print("ILLEGALE ACTION", action, state, reward)
                agent.Qlearn(state, action, new_state, reward, done)
            if action == ACTION_SPLIT and "other_state" in info: # the two splits are still in play
                agent.Qlearn_split(state, action, new_state, info["other_state"], reward, done)
            else:
                #print(state, new_state)
                agent.Qlearn(state, action, new_state, reward, done)
            state = new_state

        # Update the Q-table
        # compute_effective_reward(data, gamma=0.8)
        # for row in data:
        #    agent.learn(row["state"], row["action"], row["effective_reward"])
        
        # Keep track of the reward
        # sum_reward = sum([row["reward"] for row in data])
        prev_cum_reward += sum_reward
        cum_rewards[i] = prev_cum_reward
        rewards[i] = sum_reward

        # Update the decks
        decks.round_finished()

    plt.plot(cum_rewards)
    plt.show()

    # Plot optimal policy
    action_names = ["H", "S", "D", "P", "R"]
    policy_hard = [[action_names[agent.get_legal_action((player_sum, dealer_card, 0, 0, 0), learning=False)] for dealer_card in range(2,12)] for player_sum in range(4, 21)]
    policy_soft = [[action_names[agent.get_legal_action((player_sum, dealer_card, 1, 0, 0), learning=False)] for dealer_card in range(2,12)] for player_sum in range(13, 21)]

    print("\n\nHard policy:")
    for (i, line) in enumerate(policy_hard):
        print(4+i, ":", line)
    print("\n\nSoft policy:")
    for (i, line) in enumerate(policy_soft):
        print(13+i, ":", line)

    print("\n\n")
    print(agent.state_to_index((15, 3, 0, 0, 0)))
    print(agent.Q[agent.state_to_index((15, 3, 0, 0, 0))])
    print(agent.Q[agent.state_to_index((15, 10, 0, 0, 0))])
    print(agent.Q[agent.state_to_index((18, 10, 0, 0, 0))])
    print(agent.Q[agent.state_to_index((18, 10, 1, 0, 0))])