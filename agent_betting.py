import math
import numpy as np

class AgentBetting:

    def __init__(self, learning_rate, epsilon_func, true_count_step=1, money_step=1, initial_money=1000, min_bet=1, max_bet=1000, true_count_max=12, true_count_min=-12, discount_factor=0.99):
        self.multiplier = 1
        self.true_count_step = true_count_step
        self.money_step = money_step
        self.initial_money = initial_money
        self.min_bet = min_bet
        self.max_bet = max_bet
        self.true_count_max = true_count_max
        self.true_count_min = true_count_min
        self.discount_factor = discount_factor

        self.learning_rate = learning_rate
        self.epsilon_func = epsilon_func
        self.learn_index = 0
        self.Q = np.zeros((self.get_idx_money(2 * self.initial_money - 1) + 1, self.get_idx_true_count(self.true_count_max) + 1, self.get_idx_from_bet(self.max_bet) + 1))

    def get_idx_true_count(self, true_count):
        true_count = max(self.true_count_min, min(self.true_count_max, round(true_count)))
        return int((true_count - self.true_count_min) // self.true_count_step)

    def get_idx_money(self, money, mult=None):
        if mult is None: mult = self.multiplier
        return int((money//mult) // self.money_step)

    def get_bet_from_idx(self, idx):
        return min(self.max_bet, (self.min_bet + (2**idx-1)) * self.multiplier)

    def get_idx_from_bet(self, bet):
        try:
            return int(math.log2(bet // self.multiplier - self.min_bet + 1))
        except Exception as e:
            print(bet, self.multiplier, self.min_bet)
            raise e

    def get_random_legal_bet(self, money):
        idx_max = self.get_idx_from_bet(money)
        return self.get_bet_from_idx(np.random.randint(0, idx_max + 1))

    def Qlearn(self, observation, action, reward, next_observation):
        (money, true_count) = observation
        (next_money, next_true_count) = next_observation

        idx_money = min(self.get_idx_money(money), self.Q.shape[0] - 1)
        idx_true_count = self.get_idx_true_count(true_count)
        idx_action = self.get_idx_from_bet(action)
        idx_next_money = self.get_idx_money(next_money)
        idx_next_true_count = self.get_idx_true_count(next_true_count)

        alpha = self.learning_rate(self.learn_index)
        self.learn_index += 1
        mult = self.multiplier
        while idx_next_money >= self.Q.shape[0] and mult <= self.max_bet / 2:
            mult *= 2
            idx_next_money = self.get_idx_money(next_money, mult)
        idx_next_money = min(idx_next_money, self.Q.shape[0] - 1)
        next_reward = mult * np.max(self.Q[idx_next_money, idx_next_true_count, :])
        self.Q[idx_money, idx_true_count, idx_action] += alpha * ((reward + self.discount_factor * next_reward) - self.Q[idx_money, idx_true_count, idx_action])

    def choose_action(self, observation, learning=False):
        """
        - observation (int, int): the player's money, the true count (TC).
        Returns:
            - action (int): the bet to be placed.
        """
        (money, true_count) = observation
        while (money >= 2 * self.initial_money * self.multiplier):
            self.multiplier *= 2
        while (money < 0.5 * self.initial_money * self.multiplier) and self.multiplier > 1:
            self.multiplier /= 2
        while self.multiplier > self.max_bet:
            self.multiplier /= 2
        money /= self.multiplier

        idx_money = min(self.get_idx_money(money), self.Q.shape[0] - 1)
        idx_true_count = self.get_idx_true_count(true_count)

        epsilon = self.epsilon_func(self.learn_index)
        if learning and np.random.random() < epsilon:
            return self.get_random_legal_bet(money)
        else:
            #print(idx_money, idx_true_count, self.Q.shape)
            return self.get_bet_from_idx(np.argmax(self.Q[idx_money, idx_true_count]))