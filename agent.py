import numpy as np
from playing_env import ACTION_HIT, ACTION_STAND, ACTION_DOUBLE, ACTION_SPLIT, ACTION_SURRENDER

class QLearningAgent:

    def __init__(self, rules, learning_rate=None, delta=0.1, discount_factor=0.9, epsilon_func=None):
        '''
        Duplicate the rules from the environment.
        RULES list:
        - surrender_allowed (bool, default: False): whether the player can surrender or not.
        - double_after_split_allowed (bool, default: False): whether the player can double after split or not.
        - double_allowed (bool, default: False): whether the player can double or not.
        - split_allowed (bool, default: False): whether the player can split or not.
        - resplit_allowed (bool, default: False): whether the player can resplit or not.
        - dealer_hit_soft_17 (bool, default: True): whether the dealer hits on soft 17 or not.
        - blackjack_payout (float, default: 1.5): the payout for a blackjack.
        - floor_finished_reward (boold, default: True): whether the reward is the floor of the reward or not.
        '''
        # Q-learning parameters
        self.learning_rate = learning_rate
        self.delta = delta
        self.discount_factor = discount_factor
        self.epsilon_func = epsilon_func
        self.learn_index = 0

        # Environment parameters
        self.action_list = [ACTION_HIT, ACTION_STAND]
        self.state_size = 28 * 10 * 2 # default size: player_sum * dealer_card * usable_ace
        self.env_allows_splitting = False
        self.env_allows_doubling = False
        self.env_allows_surrender = False

        # Actions indices
        self.hit_index = 0
        self.stay_index = 1
        self.surrender_index = None
        self.splitting_index = None
        self.doubling_index = None

        if "surrender_allowed" in rules and rules["surrender_allowed"]:
            self.action_list.append(ACTION_SURRENDER)
            self.env_allows_surrender = True
            self.surrender_index = 2
        if "split_allowed" in rules and rules["split_allowed"]:
            self.action_list.append(ACTION_SPLIT)
            self.state_size *= 2
            self.env_allows_splitting = True
            self.splitting_index = len(self.action_list) - 1
        if "double_allowed" in rules and rules["double_allowed"]:
            self.action_list.append(ACTION_DOUBLE)
            self.state_size *= 2
            self.env_allows_doubling = True
            self.doubling_index = len(self.action_list) - 1
        
        # Q-table
        print("State size:", self.state_size)
        self.Q = np.zeros((self.state_size, len(self.action_list)))

        if self.env_allows_splitting: # splitting allowed
            for player_sum in range(4, 22):
                for dealer_card in range(2, 12):
                    for usable_ace in range(2):
                        for double in range(1 + self.env_allows_doubling):
                            index = self.state_to_index((player_sum, dealer_card, usable_ace, 0, double))
                            self.Q[index, self.splitting_index] = -100000

        if self.env_allows_doubling:
            for player_sum in range(4, 22):
                for dealer_card in range(2, 12):
                    for usable_ace in range(2):
                        for split in range(1 + self.env_allows_splitting):
                            index = self.state_to_index((player_sum, dealer_card, usable_ace, split, 0))
                            self.Q[index, self.doubling_index] = -100000


    def state_to_index(self, state):
        (player_sum, dealer_card, usable_ace, split, double) = state
        index = (player_sum - 4) * 20 + (dealer_card - 2) * 2 + usable_ace
        if split: index += 560 # 28 * 10 * 2
        if double:
            if self.env_allows_splitting:
                index += 1120
            else:
                index += 560
        return index

    def action_to_index(self, action):
        return self.action_list.index(action)

    def run_Q_learning_iteration(self, index_state, index_next_state, index_action, reward):
        # Computes best estimated reward
        new_q = reward
        if index_next_state is not None: # if not terminal state
            new_q = reward + self.discount_factor * np.max(self.Q[index_next_state]) + self.delta
        # alpha update Q value
        alpha = self.learning_rate(self.learn_index)
        self.learn_index += 1
        q_before = self.Q[index_state,index_action]
        self.Q[index_state,index_action] += alpha * (new_q - self.Q[index_state,index_action])
        #if index_state == 222: print(reward, new_q, q_before, self.Q[index_state,index_action])

    def run_Q_learning_iteration_split(self, index_state, index_next_state_1, index_next_state_2, index_action, reward):
        # Computes best estimated reward
        new_q = reward + self.delta
        if index_next_state_1 is not None: # if not terminal state
            new_q += self.discount_factor * np.max(self.Q[index_next_state_1])
        if index_next_state_2 is not None: # if not terminal state
            new_q += self.discount_factor * np.max(self.Q[index_next_state_2])
        # alpha update Q value
        alpha = self.learning_rate(self.learn_index)
        self.learn_index += 1
        self.Q[index_state,index_action] += alpha * (new_q - self.Q[index_state,index_action])

    def learn(self, state, action, effective_reward):
        alpha = self.learning_rate(self.learn_index)
        self.learn_index += 1
        index_state = self.state_to_index(state)
        index_action = self.action_to_index(action)
        self.Q[index_state, index_action] += alpha * (effective_reward - self.Q[index_state, index_action])

    def Qlearn(self, state, action, next_state, reward, done):
        index_state = self.state_to_index(state)
        index_action = self.action_to_index(action)
        index_next_state = None
        if not done: index_next_state = self.state_to_index(next_state)
        self.run_Q_learning_iteration(index_state, index_next_state, index_action, reward)

    def Qlearn_split(self, state, action, next_state_1, next_state_2, reward, done):
        index_state = self.state_to_index(state)
        index_action = self.action_to_index(action)
        index_next_state_1 = None
        if not done: index_next_state_1 = self.state_to_index(next_state_1)
        index_next_state_2 = None
        if not done: index_next_state_2 = self.state_to_index(next_state_2)
        self.run_Q_learning_iteration_split(index_state, index_next_state_1, index_next_state_2, index_action, reward)

    def get_legal_action(self, state, learning=False):
        index_state = self.state_to_index(state)
        # epsilon-greedy policy
        epsilon = self.epsilon_func(self.learn_index)
        if learning and np.random.random() < epsilon:
            return self.get_ramdom_legal_action(state)
        else:
            return self.action_list[np.argmax(self.Q[index_state])]

    def get_ramdom_legal_action(self, state):
        actions = [0,1]
        if self.env_allows_surrender: actions.append(ACTION_SURRENDER)
        if state[3]: actions.append(ACTION_SPLIT) # splitting
        if state[4]: actions.append(ACTION_DOUBLE) # doubling
        return np.random.choice(actions)
