
## **Deep reinforcement learning for time series: playing idealized trading games**

This repo is the code for the paper [paper](https://arxiv.org/abs/1803.03916). I explored deep reinforcement learing as a method to find the optimal trading strategies via:
* capturing the underlying dynamics (to be used in momentum trading)
* utilizing the hidden relation among the inputs (to be used in arbitrage trading)

Several neural networks are compared: 
* Gated Recurrent Unit (GRU)
* Long Short-Term Memory (LSTM)
* Convolutional Neural Network (CNN)
* Multi-Layer Perception (MLP)

### Dependencies

You can get all dependencies via the [Anaconda](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) environment file:

    conda env create -f env.yml

### Play with it
Just call the main function

    python main.py

You can play with model parameters (specified in main.py), if you get good results or any trouble, please contact me at gxiang1228@gmail.com
