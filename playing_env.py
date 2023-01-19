from typing import List
import numpy as np
import gym
from deck import DecksOfCards, card_value

ACTION_HIT = 0
ACTION_STAND = 1
ACTION_DOUBLE = 2
ACTION_SPLIT = 3
ACTION_SURRENDER = 4

OBS_PLAYER_SUM_IDX = 0
OBS_DEALER_CARD_IDX = 1
OBS_USABLE_ACE_IDX = 2
OBS_CAN_SPLIT_IDX = 3
OBS_CAN_DOUBLE_IDX = 4


class BlackJackPlayingEnv(gym.Env):
    def __init__(self, decks: DecksOfCards, rules: dict = {}, illegal_action_reward: float = -100):
        '''
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
        self.illegal_action_reward = illegal_action_reward
        self.decks = decks
        self.surrender_allowed = rules["surrender_allowed"] if "surrender_allowed" in rules else False
        self.double_after_split_allowed = rules["double_after_split_allowed"] if "double_after_split_allowed" in rules else False
        self.double_allowed = rules["double_allowed"] if "double_allowed" in rules else 2
        self.split_allowed = rules["split_allowed"] if "split_allowed" in rules else False
        self.resplit_allowed = rules["resplit_allowed"] if "resplit_allowed" in rules else False
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

    def hit(self, cards: List[int]):
        card = self.decks.draw()
        cards.append(card)
        return card


    def play_dealer(self):
        (dealer_sum, usable_ace) = self.compute_hand_value(self.dealer_cards)

        # play the dealer
        while dealer_sum < 17 or (dealer_sum == 17 and usable_ace > 0 and self.dealer_hit_soft_17):
            self.hit(self.dealer_cards)
            (dealer_sum, usable_ace) = self.compute_hand_value(self.dealer_cards)
        return dealer_sum

    def finish_game(self):
        dealer_sum = self.play_dealer()
        player_sum = self.cur_state[0]
        reward, reason = 0, ""

        if BlackJackPlayingEnv.check_blackjack(self.player_cards):
            if BlackJackPlayingEnv.check_blackjack(self.dealer_cards):
                reward = 0
                reason = "dealer and player blackjack"
            else:
                reward = 2 * self.blackjack_payout
                if self.floor_finished_reward: reward = np.floor(reward)
                reason = "player blackjack"
        elif BlackJackPlayingEnv.check_blackjack(self.dealer_cards):
            reward = -2
            reason = "dealer blackjack"
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
        # check if the action is legal
        if not self.is_legal_action(action):
            return self.cur_state, self.illegal_action_reward, False, {"reason": "illegal action"}

        reward, reason, done, info = 0, "", False, {}

        # hit
        if action == ACTION_HIT:
            self.hit(self.player_cards)
            self.cur_state = self.compute_state()
            if self.cur_state[OBS_PLAYER_SUM_IDX] > 21:
                reward = -2
                reason = "player bust"
                done = True
            else:
                reward = 0
                done = False

        # stay
        elif action == ACTION_STAND:
            reward, reason, _ = self.finish_game()
            done = True

        # double
        elif action == ACTION_DOUBLE:
            self.hit(self.player_cards)
            self.cur_state = self.compute_state()
            if self.cur_state[OBS_PLAYER_SUM_IDX] > 21:
                reward = -4
                reason = "player bust"
            else:
                reward, reason, _ = self.finish_game()
                reward *= 2
            done = True

        # split
        elif action == ACTION_SPLIT:
            # Split the cards
            hand1 = [self.player_cards[0]]
            hand2 = [self.player_cards[1]]
            self.hit(hand1)
            self.hit(hand2)

            # Check if we got some blackjacks
            if BlackJackPlayingEnv.check_blackjack(hand1):
                if BlackJackPlayingEnv.check_blackjack(hand2): # Double blackjack
                    done = True
                    reward = 2 * self.get_blackjack_reward()
                    reason = "player double blackjack"
                else: # Blackjack for first card
                    reward = self.get_blackjack_reward()
                    reason = "player blackjack"
                    self.player_cards = hand2
                    self.cur_state = self.compute_state(hand2)
            elif BlackJackPlayingEnv.check_blackjack(hand2): # Blackjack for second card
                reward = self.get_blackjack_reward()
                reason = "player blackjack"
                self.player_cards = hand1
                self.cur_state = self.compute_state(hand1)
            else: # No blackjack
                reward = 0
                reason = "player split"
                self.player_cards = hand1
                self.cur_state = self.compute_state(hand1)
                state2 = self.compute_state(hand2)
                info["other_state"] = state2
                self.player_other_hands = [hand2] + self.player_other_hands

        # surrender
        elif action == ACTION_SURRENDER:
            reward = -1
            reason = "player surrender"
            done = True

        # Even if the game is done, we still need to play the other games if there are any (due to splits)
        if done and len(self.player_other_hands) > 0:
            done = False
            info["split_finished_state"] = self.cur_state
            self.player_cards = self.player_other_hands.pop(0)
            self.cur_state = self.compute_state(self.player_cards)
            info["additional"] = "playing other game due to split"
            info["split_done"] = True

        if len(reason) > 0: info["reason"] = reason
        return self.cur_state, reward, done, info

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
            return self.cur_state[OBS_CAN_DOUBLE_IDX]
        elif action == ACTION_SPLIT:
            return self.cur_state[OBS_CAN_SPLIT_IDX]
        elif action == ACTION_SURRENDER:
            return self.surrender_allowed

    def compute_hand_value(self, cards: List[int]):
        hand_sum = sum([card_value(card) for card in cards])
        usable_aces = cards.count(1)
        while hand_sum > 21 and usable_aces > 0:
            hand_sum -= 10
            usable_aces -= 1
        usable_ace = (usable_aces > 0)
        return (hand_sum, usable_ace)

    def check_if_can_split(self, cards: List[int]):
        if self.split_allowed and len(cards) == 2 and cards[0] == cards[1]:
            if self.has_already_split:
                if self.resplit_allowed: return True
            else:
                return True
        return False

    def check_if_can_double(self, cards: List[int]):
        if len(cards) == 2 and self.double_allowed:
            if self.has_already_split:
                if self.double_after_split_allowed: return True
            else:
                return True
        return False

    def compute_state(self, cards: List[int] = None):
        if cards is None: cards = self.player_cards
        (player_sum, usable_ace) = self.compute_hand_value(cards)
        dealer_card_value = card_value(self.dealer_cards[0])
        split  = self.check_if_can_split(cards)
        double = self.check_if_can_double(cards)
        return (player_sum, dealer_card_value, usable_ace, split, double, self.decks.get_true_count())

    def check_blackjack(cards):
        return len(cards) == 2 and card_value(cards[0]) + card_value(cards[1]) == 21

    def get_blackjack_reward(self):
        reward = 2 * self.blackjack_payout
        if self.floor_finished_reward: reward = np.floor(reward)
        return reward

    def reset_from_cards(self, player_cards: List[int], dealer_cards: List[int]):
        self.player_cards = player_cards
        self.dealer_cards = dealer_cards
        self.player_other_hands = [] # for splits
        self.has_already_split = False
        
        # We include done and reward in the observation to account for potential naturals (blackjack).
        info = {"done": False, "reward": 0}

        # Check if we got some blackjacks
        if BlackJackPlayingEnv.check_blackjack(self.player_cards):
            info["done"] = True
            if BlackJackPlayingEnv.check_blackjack(self.dealer_cards):
                info["reward"] = 0
                info["reason"] = "dealer and player blackjack"
            else:
                info["reward"] = self.get_blackjack_reward()
                info["reason"] = "player blackjack"
        elif BlackJackPlayingEnv.check_blackjack(self.dealer_cards):
            info["done"] = True
            info["reward"] = -2
            info["reason"] = "dealer blackjack"

        # compute the state
        self.cur_state = self.compute_state()
        return self.cur_state, info # reward and done are only used for naturals

    def reset(self):
        # Draw two cards for the player and the dealer
        card1_player = self.decks.draw()
        card1_dealer = self.decks.draw()
        card2_player = self.decks.draw()
        card2_dealer = self.decks.draw()
        return self.reset_from_cards([card1_player, card2_player], [card1_dealer, card2_dealer])


    def render(self, mode='human'):
        pass

    def close(self):
        pass