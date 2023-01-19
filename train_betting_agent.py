from agent_betting import AgentBetting
from blackjack_env import BlackJackBettingEnv
from playing_env import BlackJackPlayingEnv, ACTION_STAND
from deck import DecksOfCards
from policy_iteration import format_state, policy_and_reward_load
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
import random

class AgentPolicy:

    def __init__(self, name=""):
        if not self.load(name):
            raise Exception("Could not load policy")

    def load(self, name):
        success, self.policy, self.reward, self.state_action_to_idx = policy_and_reward_load(name)
        return success

    def choose_action(self, state, reward, done, info):
        action = ACTION_STAND
        if state in self.policy: action = self.policy[format_state(state)]
        return action



def train(rules, initial_money = 100, N_EPISODES = 100000):
    def learning_rate(n):
        return 0.0001 + 0.02 * (max(0, N_EPISODES - n) / N_EPISODES) ** 2
    def epsilon_func(n):
        return 0.3 * (max(0, 0.5*N_EPISODES - n) / N_EPISODES) ** 2

    decks = DecksOfCards(nb_decks=6, fraction_not_in_play=0.2)
    playing_env = BlackJackPlayingEnv(decks = decks, rules=rules)
    policy_agent = AgentPolicy("save")
    betting_env = BlackJackBettingEnv(playing_env, policy_agent, initial_money=initial_money, min_bet=rules["min_bet"], max_bet=rules["max_bet"])
    betting_agent = AgentBetting(learning_rate=learning_rate,
                                    epsilon_func=epsilon_func,
                                    true_count_step=2,
                                    money_step=initial_money//20,
                                    initial_money = initial_money,
                                    min_bet=rules["min_bet"],
                                    max_bet=rules["max_bet"],
                                    true_count_max=12,
                                    true_count_min=-1, 
                                    discount_factor=0.992)

    done, state, info = True, None, None
    for _ in tqdm(range(N_EPISODES)):
        if done or random.random() < 0.01:
            betting_env.initial_money = random.randint(1, 2*initial_money)
            state, info = betting_env.reset()
            done = False

        action = betting_agent.choose_action(state, info)
        next_state, reward, done, info = betting_env.step(action)
        #print("state: ", state, "action: ", action, "reward: ", reward, "next_state: ", next_state, "done: ", done, "info: ", info)
        betting_agent.Qlearn(state, action, reward, next_state)
        state = next_state
    return betting_agent


def show_betting_agent(betting_agent, initial_money, min_true_count, max_true_count):
    extent = [- 0.5, 2 * initial_money - 1, max_true_count + 0.5, min_true_count - 0.5]

    # Policy
    norm_bet = Normalize(vmin=1, vmax=100)
    data = np.zeros((max_true_count - min_true_count + 1, 2*initial_money))

    for money in range(0, 2*initial_money):
        for true_count in range(min_true_count, max_true_count + 1):
            data[true_count - min_true_count, money] = betting_agent.choose_action((money, true_count))
    im = plt.imshow(data, cmap='hot', interpolation='none', norm=norm_bet, extent=extent, aspect=2*initial_money/(max_true_count - min_true_count + 1))
    plt.colorbar(im)
    plt.show()

if __name__ == "__main__":
    rules = {
                "min_bet": 1,
                "max_bet": 100,
                "double_allowed": True,
                "split_allowed": True,
                "surrender_allowed": True,
                "dealer_hit_soft_17": True,
                "resplit_allowed": True,
                "blackjack_payout": 1.5,
                "floor_finished_reward": True
            }
    betting_agent = train(rules, initial_money = 100, N_EPISODES = 5000000)
    show_betting_agent(betting_agent, 100, -12, 12)