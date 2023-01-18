import gym
import numpy as np
import random
from deck import DecksOfCards
from playing_env import BlackJackPlayingEnv


class BlackJackBettingEnv(gym.Env):

    def __init__(self, playing_env, player_agent, initial_money: int = 1000, min_bet: int = 1, max_bet: int = 1000, illegal_bet_reward: float = -100):
        self.playing_env = playing_env
        self.player_agent = player_agent
        self.min_bet = min_bet
        self.max_bet = max_bet
        self.current_money = initial_money
        self.illegal_bet_reward = illegal_bet_reward

        # ACTIONS space: the bet to be placed
        self.action_space = gym.paces.Discrete(1 + self.max_bet - self.min_bet)

        # OBSERVATION space: the player's gain
        self.observation_space = gym.spaces.Discrete(2 * initial_money + 1)
    
    def step(self, action: int):
        """Selects the bet to be placed.
        Returns:
            - observation (int, int, int): the player's money, the running count (RC), the true count (TC).
            - reward (float): the reward for the action.
            - done (bool): whether the game is over or not.
            - info (dict): additional information."""

        # place the bet
        bet = action + self.min_bet
        if bet > self.current_money: 
            return 0, self.illegal_bet_reward, False, {"status": "not enough money"}
        if bet > self.max_bet: 
            return 0, self.illegal_bet_reward, False, {"status": "bet too high"}
        self.current_money -= bet

        # play the game
        playing_obs, playing_reward, playing_done, playing_info = None, None, False, {}
        while not playing_done:
            playing_obs, playing_reward, playing_done, playing_info = self.playing_env.step(self.player_agent.act(playing_obs, playing_reward, playing_done, playing_info))

        # update the player's money
        gain = playing_obs * bet
        profit = gain - bet
        self.current_money += gain

        # return the observation, the reward, whether the game is over or not, and additional information
        obs = (self.current_money, self.playing_env.decks.get_running_count(), round(self.playing_env.decks.get_true_count()))
        reward = profit
        status = "ongoing game"
        done = False
        if self.current_money < self.min_bet: 
            done = True
            status = "game over"
        info = {"status": status}
        return obs, reward, done, info




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
            if int(reward) == 6: # double blackjack
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
    decks = DecksOfCards(nb_decks=6, fraction_not_in_play=0.2)
    env = BlackJackPlayingEnv(decks = decks, rules={"double_on": 2, "split": True, "surrender": True, "dealer_hit_soft_17": True, "resplit": True, "blackjack_payout": 1.5, "floor_finished_reward": True})
    data = []
    split, obs = False, None
    while not split:
        obs = env.reset()
        (real_obs, reward, done, info) = obs
        (player_sum, dealer_card, usable_ace, split, double) = real_obs
        if done: split = False
    print(obs, env.player_cards)
    print("SPLITTING")
    prev_obs = obs[0]
    obs, reward, done, info = env.step(3)
    data.append({"state": prev_obs, "action": 3, "reward": reward, "done": done or obs[2]})
    print(obs[0], env.player_cards, env.player_other_cards, reward, done, info)
    print("")
    done = False
    while not done:
        action = env.get_random_legal_action()
        if obs[0][3]: 
            print("CAN RESPLIT")
            action = 3
        print("Action: ", action)
        prev_obs = obs[0]
        obs, reward, done, info = env.step(action)
        data.append({"state": prev_obs, "action": action, "reward": reward, "done": done or obs[2]})
        print(obs[0], env.player_cards, env.player_other_cards, reward, done, info)
    env.close()
    for line in data:
        print(line)
    compute_effective_reward(data, gamma= 0.9)
    print("\n\nDATA")
    for line in data:
        print(line)