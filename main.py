import blackjack_fundamentals as blackjack
import random
import pickle
import os
import math
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict as dd

SHOE_SIZE = 6
DEFAULT_BET = 10
N_SIMS = 10000
BANKROLL = 1000.0

Q2 = dd(float)
Q4 = dd(float)
Q4_SQ = dd(float)
Q4_N = dd(int)

SPLIT_ARRAY = np.load('strategy_arrays/split_array.npy')
DOUBLE_ARRAY_SOFT = np.load('strategy_arrays/double_array_soft.npy')
DOUBLE_ARRAY_HARD = np.load('strategy_arrays/double_array_hard.npy')
STAND_ARRAY_SOFT = np.load('strategy_arrays/stand_array_soft.npy')
STAND_ARRAY_HARD = np.load('strategy_arrays/stand_array_hard.npy')

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

def allowed_actions(state):
    actions = ["hit", "stand"]
    if state[3]:
        actions.append("double")
    if state[4]:
        actions.append("split")
    return actions

def get_bet_size_state(deck_info):
    decks_remaining = max(deck_info[2], 0.5)
    true_count = deck_info[0] / decks_remaining
    return (discretize_true_count(true_count),)

def discretize_true_count(tc):
    return max(-10, min(10, math.floor(tc)))

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

def basic_strategy(state):
    
    player = state[0]
    dealer_upcard = state[1]
    hand_is_soft = bool(state[2])
    can_double = bool(state[3])
    can_split = bool(state[4])

    if can_split:
        card = None
        if hand_is_soft:
            card = 1
        else:
            card = player // 2
        if SPLIT_ARRAY[card][dealer_upcard]:
            return "split"
    
    if hand_is_soft:
        if can_double and DOUBLE_ARRAY_SOFT[player][dealer_upcard]:
            return "double"
        elif STAND_ARRAY_SOFT[player][dealer_upcard]:
            return "stand"
        return "hit"
    
    if can_double and DOUBLE_ARRAY_HARD[player][dealer_upcard]:
        return "double"
    elif STAND_ARRAY_HARD[player][dealer_upcard]:
        return "stand"
    return "hit"

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

def play_hand(player, dealer, deck, deck_info, monte_carlo = True, split = False):
    player_blackjack = (len(player) == 2 and blackjack.value(player) == 21) and not split
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
    if monte_carlo:
        action = action_decision(state)
    else:
        action = basic_strategy(state)
    did_double = False

    while True:
        if action == "hit":
            card = deck.pop()
            update_count(card, deck_info, len(deck))
            player.append(card)
            if blackjack.value(player) > 21:
                return -1
            state = get_action_state(player, dealer)
            if monte_carlo:
                action = action_decision(state)
            else:
                action = basic_strategy(state)

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
                play_hand(player, dealer, deck, deck_info, monte_carlo, split = True) +
                play_hand(new_player, dealer, deck, deck_info, monte_carlo, split = True)
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


def kelly_and_mc_test(n=100):
    br = BANKROLL

    for _ in range(n):
        deck, deck_info = init_shoe()
        hand_count = 0
        while len(deck) > 78:
            bet_size_state = get_bet_size_state(deck_info)
            bet = kelly_bet(bet_size_state, br)

            player, dealer = [], []

            for _ in range(2):
                card = deck.pop()
                update_count(card, deck_info, len(deck))
                player.append(card)

                card = deck.pop()
                update_count(card, deck_info, len(deck))
                dealer.append(card)

            net = play_hand(player, dealer, deck, deck_info, monte_carlo = True)
            br += bet * net
            hand_count += 1
    return br

def basic_strategy_test(n=100):
    br = BANKROLL
    for _ in range(n):
        deck, deck_info = init_shoe()
        hand_count = 0
        while len(deck) > 78:
            bet = DEFAULT_BET

            player, dealer = [], []

            for _ in range(2):
                card = deck.pop()
                update_count(card, deck_info, len(deck))
                player.append(card)

                card = deck.pop()
                update_count(card, deck_info, len(deck))
                dealer.append(card)

            net = play_hand(player, dealer, deck, deck_info, monte_carlo = False)
            br += bet * net
            hand_count += 1
    return br

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

load_q()
load_q_action_table()

kelly_results  = [kelly_and_mc_test()  for _ in range(N_SIMS)]
basic_results  = [basic_strategy_test()  for _ in range(N_SIMS)]

kelly_arr = np.array(kelly_results)
basic_arr = np.array(basic_results)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f'Kelly Counter vs Basic Strategy — 100 Shoes, {N_SIMS:,} simulations', fontsize=14)

# --- Left: bankroll distribution histogram ---
ax = axes[0]
bins = np.linspace(
    min(kelly_arr.min(), basic_arr.min()),
    max(kelly_arr.max(), basic_arr.max()),
    60
)
ax.hist(basic_arr, bins=bins, alpha=0.6, color='#e67e22', label='Basic Strategy (flat bet)')
ax.hist(kelly_arr,  bins=bins, alpha=0.6, color='#2980b9', label='Kelly Counter')
ax.axvline(BANKROLL, color='black', linestyle='--', linewidth=0.8, label=f'Starting bankroll ${BANKROLL:.0f}')
ax.axvline(np.median(kelly_arr),  color='#2980b9', linestyle='-',  linewidth=1.5, label=f'Kelly median  ${np.median(kelly_arr):.0f}')
ax.axvline(np.median(basic_arr),  color='#e67e22', linestyle='-',  linewidth=1.5, label=f'Basic median  ${np.median(basic_arr):.0f}')
ax.set_xlabel('Final Bankroll ($)')
ax.set_ylabel('Frequency')
ax.set_title('Final Bankroll Distribution')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)

# --- Right: summary stats bar chart ---
ax = axes[1]
metrics = ['Median\nFinal BR', 'Mean\nFinal BR', 'Win Rate\n(% > starting)', 'Bust Rate\n(% < 0)']

kelly_stats = [
    np.median(kelly_arr),
    np.mean(kelly_arr),
    np.mean(kelly_arr > BANKROLL) * 100,
    np.mean(kelly_arr < 1) * 100,
]
basic_stats = [
    np.median(basic_arr),
    np.mean(basic_arr),
    np.mean(basic_arr > BANKROLL) * 100,
    np.mean(basic_arr < 0) * 100,
]

x = np.arange(len(metrics))
w = 0.35
bars1 = ax.bar(x - w/2, kelly_stats, w, color='#2980b9', alpha=0.8, label='Kelly Counter')
bars2 = ax.bar(x + w/2, basic_stats, w, color='#e67e22', alpha=0.8, label='Basic Strategy')

for bar in bars1 + bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h:.1f}',
            ha='center', va='bottom', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=9)
ax.set_title('Summary Statistics')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('kelly_vs_basic.png', dpi=150)
plt.show()
