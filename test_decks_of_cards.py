from deck import DecksOfCards, card_value
import random
import math
import numpy as np

# Constants
NB_TESTS = 1000
HORIZON = 100


def check_frequency_of_cards(cards, n, variance_multiplier=10):
    """Checks that the frequency of each card is correct.
    The variance_multiplier ensures a correct test with a probability higher than 99% (see https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval)"""
    probability = 1/13.
    tolerance = probability * (1 - probability) / math.sqrt(n) * variance_multiplier
    for card in range(13):
        assert(abs(cards[card] /float(n) - probability) < tolerance)



####################################################################################################
########                                CARD VALUE TESTS                                    ########
####################################################################################################

def test_card_value():
    """Checks that the card_value function is correct."""
    assert card_value(1) == 11
    assert card_value(2) == 2
    assert card_value(3) == 3
    assert card_value(4) == 4
    assert card_value(5) == 5
    assert card_value(6) == 6
    assert card_value(7) == 7
    assert card_value(8) == 8
    assert card_value(9) == 9
    assert card_value(10) == 10
    assert card_value(11) == 10
    assert card_value(12) == 10
    assert card_value(13) == 10

####################################################################################################
########                               DRAW TESTS                                           ########
####################################################################################################

def test_draw_validity():
    """Checks that the draw method is valid (cards between 1 and 13)."""
    for _ in range(NB_TESTS):
        nb_decks = random.randint(1,8)
        deck = DecksOfCards(nb_decks=nb_decks, fraction_not_in_play=0.0)
        for _ in range(min(HORIZON, nb_decks * 52)):
            card = deck.draw()
            assert card >= 1 and card <= 13

def test_draw_frequency():
    """Checks that the draw method gives the correct frequency of cards."""
    n = NB_TESTS * HORIZON
    DECK_SIZE = int(0.4 * n) # We make the deck 20 times bigger than the amount of cards we draw so that the probability of each card is always around 1/13
    deck = DecksOfCards(nb_decks=DECK_SIZE, fraction_not_in_play=0.0)
    for _ in range(n):
        deck.draw()

    # Checks the frequency of each card
    cards_out = deck.get_cards_out()
    check_frequency_of_cards(cards_out, n)

def test_draw_out_of_cards():
    """Checks that the draw method raises an error when there are no more cards."""
    for _ in range(NB_TESTS):
        nb_decks = random.randint(1,8)
        deck = DecksOfCards(nb_decks=nb_decks, fraction_not_in_play=0.0)
        for _ in range(nb_decks * 52):
            deck.draw()
        try:
            deck.draw()
            assert False
        except:
            assert True



####################################################################################################
########                            RUNNING COUNT TESTS                                     ########
####################################################################################################
def test_running_count_correctness():
    """Checks that the running count is correct."""
    for _ in range(NB_TESTS):
        nb_decks = random.randint(1,8)
        deck = DecksOfCards(nb_decks=nb_decks, fraction_not_in_play=0.0)
        running_count = 0
        for _ in range(min(HORIZON, nb_decks * 52)):
            card = deck.draw()
            if card >= 10 or card == 1: running_count -= 1
            elif card < 7: running_count += 1
            assert deck.get_running_count() == running_count

def test_running_count_sums_to_zero():
    """Checks that the running count sums to zero when you draw all the cards."""
    for _ in range(NB_TESTS):
        nb_decks = random.randint(1,8)
        deck = DecksOfCards(nb_decks=nb_decks, fraction_not_in_play=0.0)
        for _ in range(nb_decks * 52):
            deck.draw()
        assert deck.get_running_count() == 0

def test_running_count_resets_on_reshuffle():
    """Checks that the running count resets to zero when you reshuffle."""
    for _ in range(NB_TESTS):
        nb_decks = random.randint(1,8)
        fraction_not_in_play = random.random()
        deck = DecksOfCards(nb_decks=nb_decks, fraction_not_in_play=fraction_not_in_play)
        for _ in range(nb_decks * 52):
            deck.draw()
            if deck.round_finished():
                assert deck.get_running_count() == 0
                break



####################################################################################################
########                               TRUE COUNT TESTS                                     ########
####################################################################################################

def test_true_count_correctness():
    """Checks that the true count is correct."""
    for _ in range(NB_TESTS):
        nb_decks = random.randint(1,8)
        deck = DecksOfCards(nb_decks=nb_decks, fraction_not_in_play=0.0)
        running_count = 0
        for _ in range(min(HORIZON, nb_decks * 52 - 1)): # Avoid last card, because of division by zero
            card = deck.draw()
            if card >= 10 or card == 1: running_count -= 1
            elif card < 7: running_count += 1
            assert deck.get_true_count() == running_count / (nb_decks - deck.nb_cards_out / 52.)

def test_true_count_sums_to_zero():
    """Checks that the true count sums to zero when you draw all the cards."""
    for _ in range(NB_TESTS):
        nb_decks = random.randint(1,8)
        deck = DecksOfCards(nb_decks=nb_decks, fraction_not_in_play=0.0)
        for _ in range(nb_decks * 52):
            deck.draw()
        assert deck.get_true_count() == 0

def test_true_count_resets_on_reshuffle():
    """Checks that the true count resets to zero when you reshuffle."""
    for _ in range(NB_TESTS):
        nb_decks = random.randint(1,8)
        fraction_not_in_play = random.random()
        deck = DecksOfCards(nb_decks=nb_decks, fraction_not_in_play=fraction_not_in_play)
        for _ in range(nb_decks * 52):
            deck.draw()
            if deck.round_finished():
                assert deck.get_true_count() == 0
                break



####################################################################################################
########                               CARDS OUT TESTS                                      ########
####################################################################################################

def test_cards_out_correctness():
    """Checks that the cards out are correct."""
    for _ in range(NB_TESTS):
        nb_decks = random.randint(1,8)
        deck = DecksOfCards(nb_decks=nb_decks, fraction_not_in_play=0.0)
        cards_out = [0] * 13
        for _ in range(min(HORIZON, nb_decks * 52)):
            card = deck.draw()
            cards_out[card - 1] += 1
            assert deck.get_cards_out() == cards_out

def test_cards_out_sums_to_nb_cards_out():
    """Checks that the cards out sums to the number of cards out."""
    for _ in range(NB_TESTS):
        nb_decks = random.randint(1,8)
        deck = DecksOfCards(nb_decks=nb_decks, fraction_not_in_play=0.0)
        for _ in range(min(HORIZON, nb_decks * 52)):
            deck.draw()
            assert sum(deck.get_cards_out()) == deck.nb_cards_out

def test_cards_out_resets_on_reshuffle():
    """Checks that the cards out resets to zero when you reshuffle."""
    for _ in range(NB_TESTS):
        nb_decks = random.randint(1,8)
        fraction_not_in_play = random.random()
        deck = DecksOfCards(nb_decks=nb_decks, fraction_not_in_play=fraction_not_in_play)
        for _ in range(nb_decks * 52):
            deck.draw()
            if deck.round_finished():
                assert sum(deck.get_cards_out()) == 0
                break



####################################################################################################
########                                 RESHUFFLE TESTS                                    ########
####################################################################################################

def test_reshuffle_is_properly_reshuffled():
    """Checks that the new deck is different than the previous one."""
    TOLERANCE_DECKS = 0.001 # fraction of times two decks can be the same after a reshuffle
    TOLERANCE_CARDS = 0.250 # fraction of times two cards can be the same after a reshuffle
    nb_identical_decks = 0
    nb_identical_cards = 0

    for _ in range(NB_TESTS):
        nb_decks = random.randint(1,8)
        fraction_not_in_play = random.random()
        deck = DecksOfCards(nb_decks=nb_decks, fraction_not_in_play=fraction_not_in_play)
        previous_deck = deck.get_cards().copy()
        while deck.round_finished() == False: # deck has not been reshuffled yet
            deck.draw()
        # checks if at least one card is different
        nb_identical_cards_in_deck = 0
        for i in range(nb_decks * 52):
            if deck.get_cards()[i] == previous_deck[i]:
                nb_identical_cards_in_deck += 1
        if nb_identical_cards_in_deck == nb_decks * 52: nb_identical_decks += 1
        nb_identical_cards += nb_identical_cards_in_deck / float(nb_decks * 52)

    # Checks that the decks are shuffled
    assert nb_identical_decks <= TOLERANCE_DECKS * NB_TESTS
    assert nb_identical_cards <= TOLERANCE_CARDS * NB_TESTS

def test_reshuffle_at_right_time():
    """Checks that the need to reshuffle is correct."""
    for _ in range(NB_TESTS):
        nb_decks = random.randint(1,8)
        fraction_not_in_play = random.random()
        deck = DecksOfCards(nb_decks=nb_decks, fraction_not_in_play=fraction_not_in_play)
        nb_cards_out = 0
        while deck.round_finished() == False: # deck has not been reshuffled yet
            deck.draw()
            nb_cards_out += 1
        # Check that the number of cards out is correct
        assert 0 <= nb_cards_out - (1. - fraction_not_in_play) * nb_decks * 52 < 1

def test_reshuffle_draw_frequency():
    """Checks that the draw frequency is correct after a reshuffle."""
    total_n = 0
    total_cards_out = np.zeros(13)

    for _ in range(NB_TESTS):
        nb_decks = random.randint(1,8)
        fraction_not_in_play = random.random()
        deck = DecksOfCards(nb_decks=nb_decks, fraction_not_in_play=fraction_not_in_play)
        while deck.round_finished() == False: # deck has not been reshuffled yet
            deck.draw()

        # Check that the draw frequency is correct
        n = min(HORIZON, int(nb_decks * 0.25 * 52)) # we draw at most 25% of the deck
        for _ in range(n):
            deck.draw()

        # Checks the frequency of each card
        cards_out = deck.get_cards_out()
        total_cards_out += np.array(cards_out)
        total_n += n
    
    # Checks on average the frequency of each card
    check_frequency_of_cards(total_cards_out.tolist(), total_n)