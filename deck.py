import random


class DecksOfCards:

    def __init__(self, nb_decks: int, fraction_not_in_play: float = 0.2):
        self.nb_decks = nb_decks
        self.threshold = (1. - fraction_not_in_play) * self.nb_decks * 52
        self.reset()

    def reset(self):
        self.generate_cards()
        self.shuffle()
        self.nb_cards_out = 0
        self.high_low_count = 0
        self.cards_out = [0] * 10
        self.needs_shuffle = False

    def generate_cards(self):
        self.cards = []
        for deck in range(self.nb_decks):
            for value in range(1, 14):
                value = min(value, 10)
                if value == 1: value = 11
                for _ in range(4):
                    self.cards.append(value)

    def shuffle(self):
        random.shuffle(self.cards)
        # Fisher-Yates shuffle
        # n = len(self.cards)
        # for i in range(n-1, 0, -1):
        #     j = random.randint(0, i)
        #     self.cards[i], self.cards[j] = self.cards[j], self.cards[i]
        # return self.cards

    def get_cards(self):
        return self.cards

    def draw(self):
        if len(self.cards) == 0: raise Exception("No more cards in the deck.")
        card = self.cards.pop()
        self.nb_cards_out += 1
        self.cards_out[card - 2] += 1
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
        return self.high_low_count / (self.nb_decks - self.nb_cards_out / 52.)

    def get_cards_out(self):
        return self.cards_out

    def round_finished(self):
        if self.needs_shuffle:
            self.reset()
            return True
        return False