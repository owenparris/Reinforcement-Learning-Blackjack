import random
import pickle
import os
import math
import blackjack_fundamentals as blackjack
from collections import defaultdict as dd

SHOE_SIZE = 6

Q2 = dd(float)
Q4 = dd(float)
Q4_SQ = dd(float)
Q4_N = dd(int)

def init_shoe():
    deck = blackjack.DECK[:] * SHOE_SIZE
    random.shuffle(deck)
    deck_info = [0, 0, SHOE_SIZE]
    return deck, deck_info

def update_count(card, deck_info, deck_len):
    if 2 <= card <= 6:
        deck_info[0] += 1
    elif card == 10 or card == 1:
        deck_info[0] -= 1
        if card == 1:
            deck_info[1] += 1
    deck_info[2] = deck_len / 52.0

def discretize_true_count(tc):
    return max(-10, min(10, math.floor(tc)))

def get_bet_size_state(deck_info):
    decks_remaining = max(deck_info[2], 0.5)
    true_count = deck_info[0] / decks_remaining
    return (discretize_true_count(true_count),)

def allowed_actions(state):
    actions = ["hit", "stand"]
    if state[3]:
        actions.append("double")
    if state[4]:
        actions.append("split")
    return actions

def action_decision(state):
    return max(allowed_actions(state), key=lambda a: Q2[(state, a)])

def get_action_state(player, dealer):
    return (
        blackjack.value(player),
        dealer[0],
        int(blackjack.is_soft(player)),
        int(len(player) == 2),
        int(len(player) == 2 and player[0] == player[1])
    )

def play_hand(player, dealer, deck, deck_info):
    player_blackjack = (len(player) == 2 and blackjack.value(player) == 21)
    dealer_blackjack = (len(dealer) == 2 and blackjack.value(dealer) == 21)

    if player_blackjack:
        if dealer_blackjack:
            return 0
        # still need to play dealer for count tracking
        while blackjack.value(dealer) < 17:
            card = deck.pop()
            update_count(card, deck_info, len(deck))
            dealer.append(card)
        return 1.5

    if dealer_blackjack:
        return -1

    state = get_action_state(player, dealer)
    action = action_decision(state)
    did_double = False

    while True:
        if action == "hit":
            card = deck.pop()
            update_count(card, deck_info, len(deck))
            player.append(card)
            if blackjack.value(player) > 21:
                return -1
            state = get_action_state(player, dealer)
            action = action_decision(state)

        elif action == "stand":
            break

        elif action == "double":
            card = deck.pop()
            update_count(card, deck_info, len(deck))
            player.append(card)
            if blackjack.value(player) > 21:
                return -2
            did_double = True
            break

        elif action == "split":
            new_player = [player.pop()]
            c1 = deck.pop()
            update_count(c1, deck_info, len(deck))
            player.append(c1)
            c2 = deck.pop()
            update_count(c2, deck_info, len(deck))
            new_player.append(c2)
            return (
                play_hand(player, dealer, deck, deck_info) +
                play_hand(new_player, dealer, deck, deck_info)
            )

    while blackjack.value(dealer) < 17:
        card = deck.pop()
        update_count(card, deck_info, len(deck))
        dealer.append(card)

    p, d = blackjack.value(player), blackjack.value(dealer)
    if d > 21 or p > d:
        return 1 + did_double
    if p < d:
        return -(1 + did_double)
    return 0

def get_variance(state):
    mean = Q4[state]
    mean_sq = Q4_SQ[state]
    return max(mean_sq - mean**2, 0.5)

def kelly_bet(state, bankroll):
    ev = Q4[state]
    var = get_variance(state)

    if ev <= 0:
        return 1

    f = min(ev / var, 0.25)
    bet = int(bankroll * f)

    return min(max(bet, 1), 100)

def train_mc(episodes=100000):
    deck, deck_info = init_shoe()
    bankroll = 1000.0

    for _ in range(episodes):
        if len(deck) <= 78:
            deck, deck_info = init_shoe()

        bet_size_state = get_bet_size_state(deck_info)
        bet = kelly_bet(bet_size_state, bankroll)

        player, dealer = [], []

        for _ in range(2):
            c = deck.pop()
            update_count(c, deck_info, len(deck))
            player.append(c)

            c = deck.pop()
            update_count(c, deck_info, len(deck))
            dealer.append(c)

        net = play_hand(player, dealer, deck, deck_info)

        profit = bet * net           
        bankroll = max(bankroll + profit, 1e-6)

        Q4_N[bet_size_state] += 1
        alpha = 1.0 / Q4_N[bet_size_state]
        Q4[bet_size_state] += alpha * (net - Q4[bet_size_state])
        Q4_SQ[bet_size_state] += alpha * (net**2 - Q4_SQ[bet_size_state])      

def test(n=500):
    results = []
    bankroll = 1000.0
    for _ in range(n):
        deck, deck_info = init_shoe()
        hand_count = 0
        while len(deck) > 78:
            bet_size_state = get_bet_size_state(deck_info)
            bet = kelly_bet(bet_size_state, bankroll)

            player, dealer = [], []

            for _ in range(2):
                card = deck.pop()
                update_count(card, deck_info, len(deck))
                player.append(card)

                card = deck.pop()
                update_count(card, deck_info, len(deck))
                dealer.append(card)

            net = play_hand(player, dealer, deck, deck_info)
            bankroll += bet * net
            hand_count += 1
        results.append(bankroll)

    return results

def save_q(filename="blackjack_kelly.pkl"):
    with open(filename, "wb") as f:
        pickle.dump((dict(Q4), dict(Q4_SQ), dict(Q4_N)), f)

def load_q(filename="blackjack_kelly.pkl"):
    global Q4, Q4_SQ, Q4_N
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
            if len(data) == 2:
                q, qsq = data       # old format (no Q4_N)
                qn = {}
            else:
                q, qsq, qn = data   # new format
            Q4.update(q)
            Q4_SQ.update(qsq)
            Q4_N.update(qn)

def load_q_action_table(filename="blackjack_v2_q.pkl"):
    global Q2
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, tuple):
                q, _ = data
            else:
                q = data
            Q2.update(q)

if __name__ == "__main__":
    load_q()
    load_q_action_table()
    for i in range(1500):
        train_mc(50000)
        print(f"Iteration {i+1} done")
        save_q()
    for tc in range(-10, 11):
        state = (tc,)
        print(tc, round(Q4[state], 4))