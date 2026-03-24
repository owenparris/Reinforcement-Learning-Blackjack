DECK = list(range(1,10))*4 + [10]*16
ACTIONS = ["hit", "stand", "double", "split"]

def value(hand):
    total = sum(hand)
    aces = hand.count(1)
    while total <= 11 and aces:
        total += 10
        aces -= 1
    return total

def is_soft(hand):
    total = sum(hand)
    return 1 in hand and total + 10 <= 21
