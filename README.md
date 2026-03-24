# Blackjack Strategy Optimisation

This project uses large-scale Monte Carlo simulation to improve blackjack decision-making beyond basic strategy.

## Overview

The goal was to build a realistic simulation of blackjack and test whether better decision rules and bet sizing could increase expected value over time.

## Key Features

* Simulates 50M+ hands of blackjack
* Models realistic game conditions (multiple decks, dealer rules, etc.)
* Compares performance against standard basic strategy
* Implements Kelly Criterion for bet sizing

## Approach

* Simulated gameplay repeatedly to estimate expected value of different decisions
* Used Monte Carlo methods to evaluate long-run outcomes under variance
* Applied the Kelly Criterion to size bets based on edge and bankroll

## Results

* Achieves higher expected value than basic strategy
* Kelly sizing improves long-term bankroll growth compared to flat betting
* Performance remains stable across large simulation runs

## Tech

* Python
* NumPy

## Notes

This project is focused on decision-making under uncertainty and balancing risk vs return, rather than exploiting unrealistic assumptions.

