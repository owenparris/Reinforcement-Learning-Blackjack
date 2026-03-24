import random
import pickle
import os
import blackjack_fundamentals as blackjack
from collections import defaultdict as dd

def can_double(state):
    return state[3]

def can_split(state):
    return state[4]

def allowed_actions(state):
    actions = ["hit", "stand"]
    if state[3]:
        actions.append("double")
    if state[4]:        
        actions.append("split")
    return actions

def get_state(player_hand, dealer_hand):
    return (
        blackjack.value(player_hand),
        dealer_hand[0],
        int(blackjack.is_soft(player_hand)),
        int(len(player_hand) == 2),
        int(len(player_hand) == 2 and player_hand[0] == player_hand[1])
    )

def epsilon_greedy(state, epsilon):
    if random.random() < epsilon:
        return random.choice(allowed_actions(state))

    return max(allowed_actions(state), key=lambda a: Q2[(state, a)])

def decision(state):
    max_q = float('-inf')
    best_action = None

    for action in allowed_actions(state):
        q = Q2[(state, action)]
        if q > max_q:
            max_q = q
            best_action = action

    if best_action is None:
        return "hit"
    
    return best_action

def generate_episode(epsilon):
    deck = blackjack.DECK[:]
    random.shuffle(deck)

    player = [deck.pop(-1), deck.pop(-1)]
    dealer = [deck.pop(-1), deck.pop(-1)]
    return generate_episode_hand(player, dealer, deck, epsilon)

def generate_episode_hand(player, dealer, deck, epsilon):
    episode = []

    state = get_state(player, dealer)

    action = epsilon_greedy(state, epsilon)
    episode.append((state, action))

    did_double = False

    while True:
        if action == "hit":
            player.append(deck.pop(-1))
            if blackjack.value(player) > 21:
                return [(episode, -1)]
            state = get_state(player, dealer)
            action = epsilon_greedy(state, epsilon)
            episode.append((state, action))
            continue

        if action == "stand":
            break

        if action == "double":
            player.append(deck.pop(-1))
            if blackjack.value(player) > 21:
                return [(episode, -2)]
            did_double = True
            break

        if action == "split":
            new_player = [player.pop(-1)]
            player.append(deck.pop(-1))
            new_player.append(deck.pop(-1))

            eps1 = generate_episode_hand(player, dealer, deck, epsilon)
            eps2 = generate_episode_hand(new_player, dealer, deck, epsilon)

            results = []

            for ep, r in (eps1):
                results.append(([(state, "split")] + ep, r))

            for ep, r in (eps2):
                results.append(([(state, "split")] + ep, r))

            return results
    
    while blackjack.value(dealer) < 17:
        dealer.append(deck.pop(-1))

    p, d = blackjack.value(player), blackjack.value(dealer)

    if d > 21 or p > d:
        return [(episode, +(1+did_double))]
    if p < d:
        return [(episode, -(1+did_double))]
    return [(episode, 0)]

Q2 = dd(float)
counts = dd(int)

def train_mc(episode_count=10000000, epsilon=0.1):
    for i in range(episode_count):
        episodes = generate_episode(epsilon)

        for episode, reward in episodes:   # <-- fix here
            visited = set()

            for state, action in episode:
                if (state, action) not in visited:
                    visited.add((state, action))

                    counts[(state, action)] += 1
                    Q2[(state, action)] += (
                        (reward - Q2[(state, action)]) 
                        / counts[(state, action)]
                    )

def save_q_table(filename="blackjack_v2_q.pkl"):
    with open(filename, "wb") as f:
        pickle.dump((dict(Q2), dict(counts)), f)  # save both

def load_q_table(filename="blackjack_v2_q.pkl"):
    global Q2, counts
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, tuple):
                q, c = data
                counts.update(c)
            else:
                q = data  # old single-dict format
            Q2.update(q)



if __name__ == "__main__":
    load_q_table()
    for i in range(1000):
        train_mc(100000, epsilon=0.01)
        print(f"Episodes trained: {(i+1)*100000}")
        save_q_table()
    save_q_table()

