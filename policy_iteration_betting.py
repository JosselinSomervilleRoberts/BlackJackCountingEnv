import random
from playing_env import BlackJackPlayingEnv, OBS_PLAYER_SUM_IDX, OBS_DEALER_CARD_IDX, OBS_USABLE_ACE_IDX, OBS_CAN_SPLIT_IDX, OBS_CAN_DOUBLE_IDX, ACTION_STAND, ACTION_SPLIT, ACTION_SURRENDER, ACTION_DOUBLE
from deck import InfiniteDeckOfCards, DecksOfCards, SimulatedCountingDeckOfCards
from blackjack_env import BlackJackBettingEnv
from train_betting_agent import AgentPolicy
from tqdm import tqdm
import numpy as np
import pickle
import matplotlib.pyplot as plt
from train_betting_agent import show_betting_agent


def format_state(state):
    true_count_rounded = max(-12, min(12, 3 * round(state[5] / 3.0)))
    return (state[OBS_PLAYER_SUM_IDX], state[OBS_DEALER_CARD_IDX], int(state[OBS_USABLE_ACE_IDX]), int(state[OBS_CAN_SPLIT_IDX]), int(state[OBS_CAN_DOUBLE_IDX]), true_count_rounded)


def binary_search_first_greater_or_equal(arr, target):
    left = 0
    right = len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] >= target:
            right = mid - 1
        else:
            left = mid + 1
    return left

def find_closest_abs_lower_or_equal_element(arr, target):
    idx = binary_search_first_greater_or_equal(arr, target)
    if target >= 0:
        if idx == 0:
            return arr[0]
        elif idx == len(arr):
            return arr[-1]
        else:
            if target != arr[idx]:
                return arr[idx - 1]
            else:
                return arr[idx]
    else:
        if idx == arr[-1]:
            return arr[-1]
        elif idx == len(arr):
            return arr[-1]
        else:
            if target != arr[idx]:
                return arr[idx]
            else:
                return arr[idx+1]

def get_legal_actions(state, list_actions):
    money = state[0]
    legal_actions = []
    for action in list_actions:
        if action <= money:
            legal_actions.append(action)
    return legal_actions

def format_betting_state(state, list_money, list_true_count):
    money = find_closest_abs_lower_or_equal_element(list_money, state[0])
    true_count = find_closest_abs_lower_or_equal_element(list_true_count, state[1])
    return (money, true_count)

def train(policy_player_name,
        rules,
        N_ITER = 10000,
        N_POLICY_ITER = 100, 
        discount_factor=0.99,
        out_of_monery_reward=-10000,
        list_true_count = [-12,-8,-5,-3,-2, -1, 0, 1, 2, 3,5,8,12],
        list_actions = [1,2,3,4,6,8,16,32,64],
        list_money = [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,45,50,60,70,80,90,100]):

    state_action_to_idx = {}
    idx_to_state_action = []
    cur_idx = 0

    for true_count in list_true_count:
        for money in list_money:
            for action in list_actions:
                if action <= money:
                    state_action_to_idx[((money, true_count), action)] = cur_idx
                    idx_to_state_action.append(((money, true_count), action))
                    cur_idx += 1

    # Comparison of number of states - actions
    n_pair_states_action = len(idx_to_state_action)
    print("n_pair_states_action:", n_pair_states_action)

    # Initiate policy of betting 1 
    policy = {}
    for idx in range(n_pair_states_action):
        (state, action) = idx_to_state_action[idx]
        if state not in policy:
            policy[state] = 1


    deck = SimulatedCountingDeckOfCards(true_count=0)
    playing_env = BlackJackPlayingEnv(decks = deck, rules=rules)
    player_agent = AgentPolicy()
    player_agent.load(name=policy_player_name, format_state=format_state, default_action=ACTION_STAND)
    betting_env = BlackJackBettingEnv(playing_env, player_agent, initial_money=1, min_bet=rules["min_bet"], max_bet=rules["max_bet"])

    cur_Q = [0] * n_pair_states_action
    cur_reward = [0] * n_pair_states_action
    expected_reward_of_policy = None
    for _ in range(N_POLICY_ITER):
        # Expected reward
        new_Q = [0] * n_pair_states_action
        new_reward = [0] * n_pair_states_action

        n_total = n_pair_states_action * N_ITER
        for i in tqdm(range(n_total)):
            idx = i // N_ITER
            (state, action) = idx_to_state_action[idx]
            (initial_money, true_count) = state
            deck.true_count = true_count
            deck.reset()

            betting_env.initial_money = initial_money
            state, _ = betting_env.reset()
            state = format_betting_state(state, list_money, list_true_count)
            next_state, reward, done, _ = betting_env.step(action)

            reward_discounted = None
            if done:
                reward_discounted = reward + out_of_monery_reward
                new_reward[idx] = reward
            else:
                next_state = format_betting_state(next_state, list_money, list_true_count)
                next_action = policy[next_state]
                idx_next = state_action_to_idx[(next_state, next_action)]
                reward_discounted = reward + discount_factor * cur_Q[idx_next]
                new_reward[idx] = reward + discount_factor * cur_reward[idx_next]
            new_Q[idx] += reward_discounted

        # New policy
        new_policy = {}
        expected_Q_of_policy = {}
        expected_reward_of_policy = {}
        for idx in range(n_pair_states_action):
            (state, action) = idx_to_state_action[idx]
            if state not in new_policy:
                new_policy[state] = action
                expected_Q_of_policy[state] = cur_Q[idx] / N_ITER
                expected_reward_of_policy = new_reward[idx] / N_ITER
            elif cur_Q[idx] / N_ITER > expected_Q_of_policy[state]:
                new_policy[state] = action
                expected_Q_of_policy[state] = cur_Q[idx] / N_ITER
                expected_reward_of_policy = new_reward[idx] / N_ITER

        # Update policy
        policy = new_policy
        cur_Q = new_Q
        cur_reward = new_reward

    return policy, cur_Q, state_action_to_idx, expected_reward_of_policy


def evaluate_policy(policy_player_name, policy, rules, list_money, list_true_count, N_EPISODES=10000, initial_money=100):
    decks = DecksOfCards(nb_decks=6, fraction_not_in_play=0.2)
    playing_env = BlackJackPlayingEnv(decks = decks, rules=rules)
    player_agent = AgentPolicy()
    player_agent.load(name=policy_player_name, format_state=format_state, default_action=ACTION_STAND)
    betting_env = BlackJackBettingEnv(playing_env, player_agent, initial_money=initial_money, min_bet=rules["min_bet"], max_bet=rules["max_bet"])


    rewards = [0]
    prev_cum_reward = 0
    cum_rewards = [0]
    bets = []
    state, _ = betting_env.reset()
    player_money = [state[0]]
    for i in tqdm(range(N_EPISODES)):
        # Place the bet
        state = format_betting_state(state, list_money, list_true_count)
        action = policy[state]
        state, reward, done, _ = betting_env.step(action)

        # Update the rewards
        prev_cum_reward += reward
        cum_rewards.append(prev_cum_reward)
        rewards.append(reward)
        bets.append(action)
        player_money.append(state[0])

        if done:
            break

    plt.plot(player_money, label="Money")
    plt.plot(cum_rewards, label="Cumulative rewards")
    plt.plot(bets, label="Bets")
    plt.legend()
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
                "floor_finished_reward": False
            }

         
    list_true_count = [-10,-7,-4, 0, 4, 7, 10]
    list_actions = [1,2,3,4,5]
    list_money = [1,2,3,4,6,8,10,13,16,20]
    def format_betting(state):
        return format_betting_state(state, list_money, list_true_count)

    name = "save"
    N_ITER = 500
    INITIAL_MONEY = 20
    N_POLICY_ITER = 20 # 3 * INITIAL_MONEY
    success, policy, reward, state_action_to_idx = False, None, None, None #policy_and_reward_load(name)
    if not success:
        policy, reward, state_action_to_idx, exp_reward = train(policy_player_name=name, rules=rules, N_ITER=N_ITER, N_POLICY_ITER=N_POLICY_ITER, discount_factor=0.99, list_true_count=list_true_count, list_actions=list_actions, list_money=list_money)
        if N_POLICY_ITER == 0:
            for elt in policy.keys():
                policy[elt] = 1
    # Show agent
    evaluate_policy(policy_player_name=name, policy=policy, rules=rules, list_money=list_money, list_true_count=list_true_count, N_EPISODES=5000000, initial_money=INITIAL_MONEY)   
    betting_agent = AgentPolicy()
    betting_agent.load_from_policy(policy=policy, format_state=format_betting, default_action=None)
    show_betting_agent(betting_agent, initial_money=INITIAL_MONEY, min_true_count=list_true_count[0], max_true_count=list_true_count[-1])
