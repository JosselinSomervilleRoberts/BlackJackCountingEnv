import numpy as np
import gym
from deck import DecksOfCards

ACTION_HIT = 0
ACTION_STAND = 1
ACTION_DOUBLE = 2
ACTION_SPLIT = 3
ACTION_SURRENDER = 4


class BlackJackPlayingEnv(gym.Env):
    def __init__(self, decks: DecksOfCards, rules: dict = {}, illegal_action_reward: float = -100):
        '''
        RULES list:
        - surrender (bool, default: False): whether the player can surrender or not.
        - double_after_split (bool, default: False): whether the player can double after split or not.
        - double_on (int, default: 2): the number of cards the player can double on.
        - split (bool, default: False): whether the player can split or not.
        - resplit (bool, default: False): whether the player can resplit or not.
        - resplit_aces (bool, default: False): whether the player can resplit aces or not.
        - dealer_hit_soft_17 (bool, default: True): whether the dealer hits on soft 17 or not.
        - blackjack_payout (float, default: 1.5): the payout for a blackjack.
        - floor_finished_reward (boold, default: True): whether the reward is the floor of the reward or not.
        '''
        self.illegal_action_reward = illegal_action_reward
        self.decks = decks
        self.surrender = rules["surrender"] if "surrender" in rules else False
        self.double_after_split = rules["double_after_split"] if "double_after_split" in rules else False
        self.double_on = rules["double_on"] if "double_on" in rules else 2
        self.split = rules["split"] if "split" in rules else False
        self.resplit = rules["resplit"] if "resplit" in rules else False
        self.resplit_aces = rules["resplit_aces"] if "resplit_aces" in rules else False
        self.dealer_hit_soft_17 = rules["dealer_hit_soft_17"] if "dealer_hit_soft_17" in rules else True
        self.blackjack_payout = rules["blackjack_payout"] if "blackjack_payout" in rules else 1.5
        self.floor_finished_reward = rules["floor_finished_reward"] if "floor_finished_reward" in rules else True

        '''
        ACTIONS space:
        - 0: hit
        - 1: stand
        - 2: double
        - 3: split
        - 4: surrender
        '''
        self.action_space = gym.spaces.Discrete(5)

        '''
        OBSERVATION space:
        - player_sum (int): the sum of the player's cards.
        - dealer_card (int): the value of the dealer's card.
        - usable_ace (bool): whether the player has a usable ace or not.
        - split (bool): whether the player has split or not.
        - double (bool): whether the player can double or not.
        '''
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Discrete(28, start=4), # from 4 to 31.
            gym.spaces.Discrete(10, start=2), # from 2 to 11.
            gym.spaces.Discrete(2),
            gym.spaces.Discrete(2),
            gym.spaces.Discrete(2),
        ))
        self.cur_state = None
        self.reset()

    def hit(self, cards: list, sum: int, usable_ace: int):
        card = self.decks.draw()
        cards.append(card)
        usable_aces = usable_ace + (card == 11)
        sum += card
        if usable_aces > 0 and sum > 21:
            sum -= 10
            usable_aces -= 1
        usable_ace = 1 if usable_aces > 0 else 0
        double, split = False, False
        double = 1 if (self.double_on >= len(cards)) and (not self.has_split or self.double_after_split) else 0
        split = 0
        if (len(cards) == 2 and cards[0] == cards[1] and self.split): # Potential split
            if not self.has_split or self.resplit: # check if player can resplit
                if cards[0] != 11 or self.resplit_aces: # check if the player can resplit aces
                    split = 1
        state = (sum, card, usable_ace, split, double)
        return state


    def play_dealer(self):
        dealer_sum = self.dealer_cards[0] + self.dealer_cards[1]
        if dealer_sum == 22: dealer_sum = 12
        usable_ace = 1 if self.dealer_cards[0] == 11 or self.dealer_cards[1] == 11 else 0

        # play the dealer
        while dealer_sum < 17 or (dealer_sum == 17 and usable_ace > 0 and self.dealer_hit_soft_17):
            (dealer_sum, _, usable_ace, _, _) = self.hit(self.dealer_cards, dealer_sum, usable_ace)
        return dealer_sum

    def finish_game(self):
        dealer_sum = self.play_dealer()
        player_sum = self.cur_state[0]
        if player_sum == 21 and len(self.player_cards) == 2:
            reward = 2 * self.blackjack_payout
            reason = "player blackjack"
        elif dealer_sum > 21:
            reward = 2
            reason = "dealer bust"
        elif dealer_sum < player_sum:
            reward = 2
            reason = "player higher than dealer"
        elif dealer_sum == player_sum:
            reward = 0
            reason = "player equal to dealer"
        else:
            reward = -2
            reason = "player lower than dealer"
        return reward, reason, dealer_sum

    def step(self, action: int):
        dealer_card = self.cur_state[1]
        reward, reason, done, info = 0, "", False, {}

        # hit
        if action == ACTION_HIT:
            (player_sum, _, usable_ace, split, double) = self.hit(self.player_cards, self.cur_state[0], self.cur_state[2])
            self.cur_state = (player_sum, dealer_card, usable_ace, split, double)
            if player_sum > 21:
                reward = -2
                reason = "player bust"
                done = True

        # stay
        elif action == ACTION_STAND:
            reward, reason, _ = self.finish_game()
            player_sum = self.cur_state[0]
            done = True

        # double
        elif action == ACTION_DOUBLE:
            if not self.cur_state[4]: # cannot double
                return (self.cur_state, self.illegal_action_reward, True, {}), self.illegal_action_reward, True, {"reason": "illegal action, cannot double"}
            (player_sum, _, usable_ace, split, double) = self.hit(self.player_cards, self.cur_state[0], self.cur_state[2])
            self.cur_state = (player_sum, dealer_card, usable_ace, split, double)
            if player_sum > 21:
                reward = -4
                reason = "player bust"
            else:
                reward, reason, _ = self.finish_game()
                reward *= 2
            done = True

        # split
        elif action == ACTION_SPLIT:
            if not self.cur_state[3]: # cannot split
                return (self.cur_state, self.illegal_action_reward, True, {}), self.illegal_action_reward, True, {"reason": "illegal action, cannot split"}

            # Split the cards
            other_cards = [self.player_cards[1]]
            self.player_cards = [self.player_cards[0]]

            # Game 1 (first card)
            (player_sum, _, usable_ace, split, double) = self.hit(cards = self.player_cards,
                                                                    sum = self.player_cards[0],
                                                                    usable_ace = 1 if self.player_cards[0] == 11 else 0)
            self.cur_state = (player_sum, dealer_card, usable_ace, split, double)

            # Game 2 (second card)
            (player_sum, _, usable_ace, split, double) = self.hit(cards = other_cards,
                                                                    sum = other_cards[0],
                                                                    usable_ace = 1 if other_cards[0] == 11 else 0)
            other_state = (player_sum, dealer_card, usable_ace, split, double)
            self.other_states = [other_state] + self.other_states
            self.player_other_cards = [other_cards] + self.player_other_cards

            # Check if we got some blackjacks
            if self.cur_state[0] == 21 and self.other_states[0][0] == 21: # Double blackjack
                done = True
                reward = 4 * self.blackjack_payout
                if self.floor_finished_reward: reward = 2 * np.floor(2 * self.blackjack_payout)
                reason = "player double blackjack"
                info["obs_of_finished_split"] = self.other_states.pop(0)
                self.player_other_cards.pop(0)
            elif self.cur_state[0] == 21: # Blackjack for first card
                done = True
                reward = 2 * self.blackjack_payout
                if self.floor_finished_reward: reward = np.floor(reward)
                reason = "player blackjack"
            elif self.other_states[0][0] == 21: # Blackjack for second card
                done = False
                reward = 2 * self.blackjack_payout
                if self.floor_finished_reward: reward = np.floor(reward)
                reason = "player blackjack"
                info["obs_of_finished_split"] = self.other_states.pop(0)
                info["additional"] = "playing other game due to split"
                self.player_other_cards.pop(0)

        # surrender
        elif action == ACTION_SURRENDER:
            if not self.surrender: # cannot surrender
                return (self.cur_state, self.illegal_action_reward, True, {}), self.illegal_action_reward, True, {"reason": "illegal action, cannot surrender"}
            reward = -1
            reason = "player surrender"
            done = True

        # Even if the game is done, we still need to play the other games if there are any (due to splits)
        original_done = done
        if done and len(self.other_states) > 0:
            info["obs_of_finished_split"] = self.cur_state
            self.cur_state = self.other_states.pop(0)
            self.player_cards = self.player_other_cards.pop(0)
            done = False
            info["additional"] = "playing other game due to split"

        if len(reason) > 0: info["reason"] = reason
        return (self.cur_state, reward, original_done, info), reward, done, info

    def get_random_action(self):
        return self.action_space.sample()

    def get_random_legal_action(self):
        action = self.get_random_action()
        while not self.is_legal_action(action):
            action = self.get_random_action()
        return action

    def is_legal_action(self, action: int):
        if action == ACTION_HIT:
            return True
        elif action == ACTION_STAND:
            return True
        elif action == ACTION_DOUBLE:
            return self.cur_state[4]
        elif action == ACTION_SPLIT:
            return self.cur_state[3]
        elif action == ACTION_SURRENDER:
            return self.surrender


    def reset(self):
        # We include done and reward in the observation to account for potential naturals (blackjack).
        done, reward, reason, info = False, 0, "", {}

        card1_player = self.decks.draw()
        card1_dealer = self.decks.draw()
        card2_player = self.decks.draw()
        card2_dealer = self.decks.draw()
        self.player_cards = [card1_player, card2_player]
        self.player_other_cards = [] # for splits
        self.other_states = [] # for splits
        self.has_split = False
        self.dealer_cards = [card1_dealer, card2_dealer]

        if card1_dealer + card2_dealer == 21:
            done = True
            if card1_player + card2_player == 21:
                reward = 0
                reason = "dealer and player blackjack"
            else:
                reward = -2
                reason = "dealer blackjack"
        elif card1_player + card2_player == 21: 
            done = True
            reward = 2 * self.blackjack_payout
            if self.floor_finished_reward: reward = np.floor(reward)
            reason = "player blackjack"

        player_sum = card1_player + card2_player
        if player_sum == 22: player_sum = 12
        dealer_card = card1_dealer
        usable_ace = 1 if card1_player == 11 or card2_player == 11 else 0
        split = 1 if card1_player == card2_player and self.split else 0
        double = 1 if self.double_on else 0
        self.cur_state = (player_sum, dealer_card, usable_ace, split, double)

        if len(reason) > 0: info["reason"] = reason
        return (self.cur_state, reward, done, info) # reward and done are only used for naturals


    def render(self, mode='human'):
        pass

    def close(self):
        pass