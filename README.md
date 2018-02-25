# deep-RL-time-series

Deep reinforcement learning for time series: playing idealized trading games
source code for the paper:

In this paper I explored deep reinforcement learing as a method to find the optimal strategies for trading. I compared three neural networks: Stacked Gated Recurrent Unit (GRU), stacked Convolutional Neural Network (CNN), and multi-layer perception (MLP). I designed two simple trading games aiming to test if the trained agent can 1) capture the underlying dynamics (to be used in mometum trading), and 2) utilize the hidden relation among the inputs (to be used in arbitrage trading). Experimented with It turns out that GRU performs best. However as these are just simplified worlds for the agent to play with, more further investigation is deserved.

