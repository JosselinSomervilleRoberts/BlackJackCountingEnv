import random
from playing_env import BlackJackPlayingEnv, OBS_PLAYER_SUM_IDX, OBS_DEALER_CARD_IDX, OBS_USABLE_ACE_IDX, OBS_CAN_SPLIT_IDX, OBS_CAN_DOUBLE_IDX, ACTION_STAND, ACTION_SPLIT, ACTION_SURRENDER, ACTION_DOUBLE
from deck import InfiniteDeckOfCards, DecksOfCards
from tqdm import tqdm
import numpy as np
from policy_show import show_policy
import pickle
import matplotlib.pyplot as plt
from agent_policy import policy_and_reward_save, policy_and_reward_load


def get_card_of_value(card_value):
    if card_value == 11 or card_value == 1:
        return 1
    elif card_value == 10:
        return random.randint(10, 14)
    elif card_value >= 2 and card_value <= 9:
        return card_value
    else:
        raise Exception("Invalid card value: {}".format(card_value))

def generate_cards_from_state(state, rules):
    player_sum = state[OBS_PLAYER_SUM_IDX]
    card_1, card_2, card_3, dealer_card_2 = None, None, None, None
    if not state[OBS_CAN_DOUBLE_IDX] and rules["double_allowed"]: # cannot double while the rules allow it. This means that there are at least 3 cards
        card_value_1 = 11 if state[OBS_USABLE_ACE_IDX] else random.randint(2, min(10, player_sum - 4))
        player_sum = player_sum - card_value_1
        card_value_2 = random.randint(max(2 - state[OBS_USABLE_ACE_IDX], player_sum - 10), min(10, player_sum - 2 + state[OBS_USABLE_ACE_IDX]))
        card_value_3 = player_sum - card_value_2
        card_1 = get_card_of_value(card_value_1)
        card_2 = get_card_of_value(card_value_2)
        card_3 = get_card_of_value(card_value_3)
    elif state[OBS_CAN_SPLIT_IDX]:
        if state[OBS_USABLE_ACE_IDX]: # double aces
            card_1, card_2 = 1, 1
        else:
            card_1 = get_card_of_value(player_sum // 2)
            card_2 = card_1
    else:
        # We use min(10, player_sum - 2) and min(10, player_sum - 1) as we want to avoid a value of 1 in the second card.
        # If there is one ace then it will be the first card. The only case where we can have a value of 1 in the second card is when we have two aces.
        # This case is already covered in the plit condition above, so we can forbid having an ace for the second card.
        card_value_1 = 11 if state[OBS_USABLE_ACE_IDX] else random.randint(max(2, player_sum - 10), min(10, player_sum - 2))
        card_value_2 = player_sum - card_value_1
        card_1 = get_card_of_value(card_value_1)
        card_2 = get_card_of_value(card_value_2)

    dealer_card_1 = get_card_of_value(state[OBS_DEALER_CARD_IDX])
    # dealer cannot have a blackjack
    if state[OBS_DEALER_CARD_IDX] == 11: # cannot have a value of 10
        dealer_card_2 = random.randint(1, 10)
    elif state[OBS_DEALER_CARD_IDX] == 10: # cannot have an ace
        dealer_card_2 = random.randint(2, 14)
    else: # can have any card
        dealer_card_2 = random.randint(1, 14)
    player_cards = [card_1, card_2]
    if card_3 is not None: player_cards.append(card_3)
    return player_cards, [dealer_card_1, dealer_card_2]

def get_legal_actions(state, rules):
    list_actions = [0, 1]
    if state[OBS_CAN_DOUBLE_IDX]:  list_actions.append(ACTION_DOUBLE)       # doubling
    if state[OBS_CAN_SPLIT_IDX]:   list_actions.append(ACTION_SPLIT)        # splitting
    if rules["surrender_allowed"]: list_actions.append(ACTION_SURRENDER)    # surrendering
    return list_actions

def format_state_no_counting(state):
    return (state[OBS_PLAYER_SUM_IDX], state[OBS_DEALER_CARD_IDX], int(state[OBS_USABLE_ACE_IDX]), int(state[OBS_CAN_SPLIT_IDX]), int(state[OBS_CAN_DOUBLE_IDX]))

def register_mapping(state, rules, idx, dict_sati, list_itsa):
    list_actions = get_legal_actions(state, rules)
    for action in list_actions:
        dict_sati[(state, action)] = idx
        list_itsa.append((state, action))
        idx += 1
    return idx

def train(rules, N_ITER = 1000, N_POLICY_ITER = 5):
    hards_no_double, softs_no_double, splits_no_double = [], [], []
    hards_double, softs_double, splits_double = [], [], []

    if rules["split_allowed"] and rules["double_allowed"]:
        # WITH SPLITS AND DOUBLING
        # states without doubling
        hards_no_double  = list(range(6, 20 + 1))  # - hard 6 - 20 (no hard 4 as it is split 2, no hard 5 as it can only be 2 and 3 which can be doubled)
        softs_no_double  = list(range(13, 21 + 1)) # - soft 13 - 21 (no soft 12 as soft 12 is split A)
        splits_no_double = []                      # - No splits as they can be doubled

        # states with doubling
        hards_double = list(range(5, 20 + 1))   # - all hards can be doubled
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
        softs_no_double  = list(range(13, 21 + 1)) # - soft 13 - 21 (no soft 12 as it can only be As-As which can be doubled)
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

    for _ in range(N_POLICY_ITER):
        # Expected reward
        exp_reward = [0] * n_pair_states_action

        n_total = n_pair_states_action * N_ITER
        for i in tqdm(range(n_total)):
            idx = i // N_ITER
            (state, action) = idx_to_state_action[idx]
            player_cards, dealer_cards = generate_cards_from_state(state, rules)
            env.reset_from_cards(player_cards, dealer_cards)

            # Play the game
            state, reward, done, _ = env.step(action)
            reward_total = reward
            while not done:
                action = ACTION_STAND
                if state in policy: action = policy[format_state_no_counting(state)]
                state, reward, done, _ = env.step(action)
                reward_total += reward

            # Update expected reward
            exp_reward[idx] += reward_total

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

        # Update policy
        policy = new_policy

    return policy, exp_reward, state_action_to_idx


def evaluate_policy(policy, rules, N_EPISODES=10000):
    decks = DecksOfCards(nb_decks=6, fraction_not_in_play=0.2)
    env = BlackJackPlayingEnv(decks = decks, rules=rules)

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
        else:
            # Play all the actions until the game is finished
            sum_reward = 0
            done = False
            while not done:
                action = None
                state = format_state_no_counting(state)
                if state[OBS_PLAYER_SUM_IDX] == 21:
                    action = ACTION_STAND
                elif state in policy: action = policy[state]
                else:
                    state_no_double = (state[0], state[1], state[2], state[3], 0)
                    action = policy[state_no_double]
                state, reward, done, info = env.step(action)
                sum_reward += reward
            prev_cum_reward += sum_reward
            cum_rewards[i] = prev_cum_reward
            rewards[i] = sum_reward

        # Update the decks
        decks.round_finished()

    plt.plot(cum_rewards)
    plt.show()

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

    name = "save"
    N_ITER = 5000
    success, policy, reward, state_action_to_idx = policy_and_reward_load(name)
    if not success:
        policy, reward, state_action_to_idx = train(rules, N_ITER=N_ITER, N_POLICY_ITER=5)
        policy_and_reward_save(name, policy, reward, state_action_to_idx)
        show_policy(policy, reward, state_action_to_idx, N_ITER)
    else:
        evaluate_policy(policy, rules, N_EPISODES=1000000)
        show_policy(policy, reward, state_action_to_idx, N_ITER)
