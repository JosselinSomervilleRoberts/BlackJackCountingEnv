from deck import SimulatedCountingDeckOfCards
from playing_env import BlackJackPlayingEnv, OBS_PLAYER_SUM_IDX, ACTION_STAND
import numpy as np
from agent_policy import AgentPolicy, policy_and_reward_load
from policy_iteration_counting import format_state_counting
from policy_iteration import format_state_no_counting
from tqdm import tqdm
import matplotlib.pyplot as plt


def agent_play_one_game(env, agent, reset_on_natural=False):
    state, info = env.reset()
    sum_reward = 0
    while reset_on_natural and "done" in info and info["done"]: # The player or the dealer has a natural blackjack
        state, info = env.reset()

    # The player or the dealer has a natural blackjack
    if "done" in info and info["done"]:
        sum_reward = info["reward"]
    else:
        # Play all the actions until the game is finished
        done = False
        while not done:
            action = agent.choose_action(state)
            state, reward, done, info = env.step(action)
            sum_reward += reward

    # Finish game and return the total reward
    env.decks.round_finished()
    return sum_reward   



def policy_play_one_game(env, policy, format_state=(lambda x: x), reset_on_natural=False):
    state, info = env.reset()
    sum_reward = 0
    while reset_on_natural and "done" in info and info["done"]: # The player or the dealer has a natural blackjack
        state, info = env.reset()

    # The player or the dealer has a natural blackjack
    if "done" in info and info["done"]:
        sum_reward = info["reward"]
    else:
        # Play all the actions until the game is finished
        done = False
        while not done:
            action = None
            state = format_state(state)
            if state[OBS_PLAYER_SUM_IDX] == 21:
                action = ACTION_STAND
            elif state in policy: action = policy[state]
            else:
                state_no_double = (state[0], state[1], state[2], state[3], 0)
                action = policy[state_no_double]
            state, reward, done, info = env.step(action)
            sum_reward += reward

    # Finish game and return the total reward
    env.decks.round_finished()
    return sum_reward

def compute_expected_value_of_agent(rules, agent, list_true_count, n_iter=10000, reset_on_natural=False):
    decks = SimulatedCountingDeckOfCards(true_count=0)
    env = BlackJackPlayingEnv(decks = decks, rules=rules)

    list_rewards = np.zeros(len(list_true_count))
    for idx in tqdm(range(n_iter * len(list_true_count))):
        idx_true_count = idx // n_iter
        true_count = list_true_count[idx_true_count]
        decks.true_count = true_count
        decks.reset()
        reward = agent_play_one_game(env, agent, reset_on_natural=reset_on_natural)
        list_rewards[idx_true_count] += reward / n_iter
    return list_rewards


if __name__ == "__main__":
    rules = {
                "double_allowed": True,
                "split_allowed": True,
                "surrender_allowed": True,
                "dealer_hit_soft_17": True,
                "resplit_allowed": True,
                "blackjack_payout": 1.5,
                "floor_finished_reward": True
            }

    N_ITER = 200000
    agent_counting = AgentPolicy()
    agent_counting.load(name="best_counting", format_state=format_state_counting, default_action=ACTION_STAND)
    agent_no_counting = AgentPolicy()
    agent_no_counting.load(name="best_no_counting", format_state=format_state_no_counting, default_action=ACTION_STAND)
    list_true_count = list(np.arange(-16, 16+1))
    rewards_counting = compute_expected_value_of_agent(rules, agent_counting, list_true_count=list_true_count, n_iter=N_ITER, reset_on_natural=True)
    rewards_no_counting = compute_expected_value_of_agent(rules, agent_no_counting, list_true_count=list_true_count, n_iter=N_ITER, reset_on_natural=True)
    print(rewards_counting)
    print(rewards_no_counting)
    plt.title("Expected value depending on count")
    plt.xlabel("True count (TC)")
    plt.ylabel("Expected value (in %)")
    plt.plot(list_true_count, 100 * rewards_counting, label="Agent counting")
    plt.plot(list_true_count, 100 * rewards_no_counting, label="Agent not counting")
    plt.legend()
    plt.show()