import random
from playing_env import BlackJackPlayingEnv, OBS_PLAYER_SUM_IDX, OBS_DEALER_CARD_IDX, OBS_USABLE_ACE_IDX, OBS_CAN_SPLIT_IDX, OBS_CAN_DOUBLE_IDX, ACTION_STAND, ACTION_SPLIT, ACTION_SURRENDER, ACTION_DOUBLE
from deck import DecksOfCards, InfiniteDeckOfCards
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


def get_card_of_value(card_value):
    if card_value == 11 or card_value == 1:
        return 1
    elif card_value == 10:
        return random.randint(10, 14)
    elif card_value >= 2 and card_value <= 9:
        return card_value
    else:
        raise Exception("Invalid card value: {}".format(card_value))

def generate_cards_from_state(state):
    player_sum = state[OBS_PLAYER_SUM_IDX]
    card_value_1 = random.randint(max(2, player_sum - 11), min(11, player_sum - 2))
    card_value_2 = player_sum - card_value_1
    card_1 = get_card_of_value(card_value_1)
    card_2 = get_card_of_value(card_value_2)
    dealer_card_1 = get_card_of_value(state[OBS_DEALER_CARD_IDX])
    dealer_card_2 = random.randint(1, 14)
    return [card_1, card_2], [dealer_card_1, dealer_card_2]

def get_legal_actions(state, rules):
    list_actions = [0, 1]
    if state[OBS_CAN_DOUBLE_IDX]:  list_actions.append(ACTION_DOUBLE)       # doubling
    if state[OBS_CAN_SPLIT_IDX]:   list_actions.append(ACTION_SPLIT)        # splitting
    if rules["surrender_allowed"]: list_actions.append(ACTION_SURRENDER)    # surrendering
    return list_actions

def format_state(state):
    return (state[OBS_PLAYER_SUM_IDX], state[OBS_DEALER_CARD_IDX], int(state[OBS_USABLE_ACE_IDX]), int(state[OBS_CAN_SPLIT_IDX]), int(state[OBS_CAN_DOUBLE_IDX]))

def register_mapping(state, rules, idx, dict_sati, list_itsa):
    list_actions = get_legal_actions(state, rules)
    for action in list_actions:
        dict_sati[(state, action)] = idx
        list_itsa.append((state, action))
        idx += 1
    return idx

if __name__ == "__main__":
    N_ITER = 1000

    rules = {
                "double_allowed": True,
                "split_allowed": True,
                "surrender_allowed": True,
                "dealer_hit_soft_17": True,
                "resplit_allowed": True,
                "blackjack_payout": 1.5,
                "floor_finished_reward": True
            }

    n_actions = 2 + rules["double_allowed"] + rules["split_allowed"] + rules["surrender_allowed"]
    n_dealer_cards = 11

    hards_no_double, softs_no_double, splits_no_double = [], [], []
    hards_double, softs_double, splits_double = [], [], []

    if rules["split_allowed"] and rules["double_allowed"]:
        # WITH SPLITS AND DOUBLING
        # states without doubling
        hards_no_double  = list(range(6, 20 + 1))  # - hard 6 - 20 (no hard 4 as it is split 2, no hard 5 as it can only be 2 and 3 which can be doubled)
        softs_no_double  = list(range(14, 21 + 1)) # - soft 14 - 21 (no soft 12 as soft 12 is split A, no soft 13 as it can only be As and 2 which can be doubled)
        splits_no_double = []                      # - No splits as they can be doubled

        # states with doubling
        hards_double = list(range(5, 19 + 1))   # - all hards can be doubled except 20 cause it's a split 10
        softs_double = list(range(13, 20 + 1))  # - all softs can be doubled except 21 cause it would be a blackjack
        splits_double = list(range(1, 10 + 1))  # - all splits can be doubled

    elif rules["split_allowed"] and not rules["double_allowed"]:
        # WITH SPLITS BUT NO DOUBLING
        # states without doubling
        hards_no_double  = list(range(5, 20 + 1))  # - hard 6 - 20 (no hard 4 as it is split 2)
        softs_no_double  = list(range(13, 21 + 1)) # - soft 13 - 21 (no soft 12 as soft 12 is split A)
        splits_no_double = list(range(1, 10 + 1))  # - all splits

        # no states with doubling
        hards_double, softs_double, splits_double = [], [], []
    
    elif not rules["split_allowed"] and rules["double_allowed"]:
        # NO SPLITS BUT WITH DOUBLING
        # states without doubling
        hards_no_double  = list(range(6, 20 + 1))  # - hard 6 - 20 (no hard 4 nor hard 5 as they can only be 2-2 and 2-3 respectively which can be doubled)
        softs_no_double  = list(range(14, 21 + 1)) # - soft 14 - 21 (no soft 12 nor soft 13 as they can only be As-As and As-2 respectively which can be doubled)
        splits_no_double = []                      # - No splits allowed

        # states with doubling
        hards_double = list(range(5, 20 + 1))   # - all hards can be doubled
        softs_double = list(range(13, 20 + 1))  # - all softs can be doubled except 21 cause it would be a blackjack
        splits_double = []                      # - No splits allowed

    elif not rules["split_allowed"] and not rules["double_allowed"]:
        # NO SPLITS AND NO DOUBLING
        # states without doubling
        hards_no_double  = list(range(4, 20 + 1))  # - hard 4 - 20
        softs_no_double  = list(range(12, 21 + 1)) # - soft 12 - 21
        splits_no_double = []                      # - No splits allowed

        # no states with doubling
        hards_double, softs_double, splits_double = [], [], []

    state_action_to_idx = {}
    idx_to_state_action = []
    cur_idx = 0

    for dealer_card in range(2, 11 + 1):
        for player_sum in hards_no_double:
            cur_idx = register_mapping((player_sum, dealer_card, 0, 0, 0), rules, cur_idx, state_action_to_idx, idx_to_state_action)
        for player_sum in softs_no_double:
            cur_idx = register_mapping((player_sum, dealer_card, 1, 0, 0), rules, cur_idx, state_action_to_idx, idx_to_state_action)
        for card_value in splits_no_double:
            usable_ace = 0
            player_sum = 2 * card_value
            if card_value == 1:
                usable_ace = 1
                player_sum = 12
            cur_idx = register_mapping((player_sum, dealer_card, usable_ace, 1, 0), rules, cur_idx, state_action_to_idx, idx_to_state_action)

        # doubling
        for player_sum in hards_double:
            cur_idx = register_mapping((player_sum, dealer_card, 0, 0, 1), rules, cur_idx, state_action_to_idx, idx_to_state_action)
        for player_sum in softs_double:
            cur_idx = register_mapping((player_sum, dealer_card, 1, 0, 1), rules, cur_idx, state_action_to_idx, idx_to_state_action)
        for card_value in splits_double:
            usable_ace = 0
            player_sum = 2 * card_value
            if card_value == 1:
                usable_ace = 1
                player_sum = 12
            cur_idx = register_mapping((player_sum, dealer_card, usable_ace, 1, 1), rules, cur_idx, state_action_to_idx, idx_to_state_action)

    # Comparison of number of states - actions
    n_pair_states_action = len(idx_to_state_action)
    n_naive_states = (21 - 4 + 1) * (10 - 1 + 1) * 2 * (1 + rules["split_allowed"]) * (1 + rules["double_allowed"])
    n_naive_actions = 2 + rules["split_allowed"] + rules["double_allowed"] + rules["surrender_allowed"]
    n_naive_pair_states_action = n_naive_states * n_naive_actions
    print("n_naive_states:", n_naive_states)
    print("n_naive_actions:", n_naive_actions)
    print("n_naive_pair_states_action:", n_naive_pair_states_action)
    print("n_pair_states_action:", n_pair_states_action)

    # Initiate random policy
    policy = {}
    for idx in range(n_pair_states_action):
        (state, action) = idx_to_state_action[idx]
        if state not in policy:
            policy[state] = np.random.choice(get_legal_actions(state, rules))


    deck = InfiniteDeckOfCards()
    env = BlackJackPlayingEnv(decks = deck, rules=rules)

    # Expected reward
    exp_reward = [0] * n_pair_states_action

    n_total = n_pair_states_action * N_ITER
    for i in tqdm(range(n_total)):
        idx = i // N_ITER
        (state, action) = idx_to_state_action[idx]
        player_cards, dealer_cards = generate_cards_from_state(state)
        env.reset_from_cards(player_cards, dealer_cards)

        # Play the game
        state, reward, done, _ = env.step(action)
        reward_total = reward
        while not done:
            action = ACTION_STAND
            if state in policy: action = policy[format_state(state)]
            state, reward, done, _ = env.step(action)
            reward_total += reward

        # Update expected reward
        exp_reward[idx] += reward_total

    for idx in range(n_pair_states_action):
        (state, action) = idx_to_state_action[idx]
        print("State:", state, "Action:", action, "Expected reward:", exp_reward[idx] / N_ITER)

    # New policy
    new_policy = {}
    expected_reward_of_policy = {}
    for idx in range(n_pair_states_action):
        (state, action) = idx_to_state_action[idx]
        if state not in new_policy:
            new_policy[state] = action
            expected_reward_of_policy[state] = exp_reward[idx] / N_ITER
        elif exp_reward[idx] / N_ITER > expected_reward_of_policy[state]:
            new_policy[state] = action
            expected_reward_of_policy[state] = exp_reward[idx] / N_ITER

    print("\n\n\nNew policy:")
    for state in new_policy.keys():
        print("State:", state, "Action:", new_policy[state], "Expected reward:", expected_reward_of_policy[state])