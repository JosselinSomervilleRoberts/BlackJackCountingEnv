import random
import numpy as np

def card_value(card):
    if card == 1:
        return 11
    elif card >= 10:
        return 10
    else:
        return card


class DeckOfCards:

    def __init__(self):
        self.cards = []

    def shuffle(self):
        random.shuffle(self.cards)

    def get_cards(self):
        return self.cards

    def generate_cards(self, nb_decks=1):
        self.cards = []
        for _ in range(nb_decks):
            for value in range(1, 14):
                for _ in range(4):
                    self.cards.append(value)

    def round_finished(self):
        pass


class InfiniteDeckOfCards(DeckOfCards):

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.generate_cards(nb_decks=1)

    def draw(self):
        return random.choice(self.cards)


class SimulatedCountingDeckOfCards(DeckOfCards):

    def __init__(self, true_count: int, n_binomial = 50):
        super().__init__()
        self.true_count = true_count
        self.n_binomial = n_binomial
        self.reset()

    def reset(self):
        self.prob_neutral = np.random.binomial(p=3/13., n=self.n_binomial) / self.n_binomial
        self.prob_high = 1/2. + self.true_count / 104. - self.prob_neutral / 2.

    def draw(self):
        x = random.random()
        if x <= self.prob_neutral:
            return random.choice([7, 8, 9])
        elif x <= self.prob_neutral + self.prob_high:
            return random.choice([1, 10, 11, 12, 13])
        else:
            return random.choice([2, 3, 4, 5, 6])

    def get_true_count(self):
        """Returns the true count."""
        return self.true_count


class DecksOfCards(DeckOfCards):

    def __init__(self, nb_decks: int, fraction_not_in_play: float = 0.2):
        super().__init__()
        self.nb_decks = nb_decks
        self.threshold = (1. - fraction_not_in_play) * self.nb_decks * 52
        self.reset()

    def reset(self):
        self.generate_cards(nb_decks=self.nb_decks)
        self.shuffle()
        self.nb_cards_out = 0
        self.high_low_count = 0
        self.cards_out = [0] * 13
        self.needs_shuffle = False

    def draw(self):
        if len(self.cards) == 0: raise Exception("No more cards in the deck.")
        card = self.cards.pop()
        self.nb_cards_out += 1
        self.cards_out[card - 1] += 1
        if self.nb_cards_out >= self.threshold: self.needs_shuffle = True
        if card >= 10 or card == 1: self.high_low_count -= 1
        elif card < 7: self.high_low_count += 1
        return card

    def get_running_count(self):
        """Returns the running count."""
        return self.high_low_count

    def get_true_count(self):
        """Returns the true count."""
        if self.high_low_count == 0: return 0 # handle division by zero
        return int(self.high_low_count / (self.nb_decks - self.nb_cards_out / 52.))

    def get_cards_out(self):
        return self.cards_out

    def round_finished(self):
        if self.needs_shuffle:
            self.reset()
            return True
        return False