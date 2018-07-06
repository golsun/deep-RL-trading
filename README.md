# **Playing trading games with deep reinforcement learning**

This repo is the code for this [paper](https://arxiv.org/abs/1803.03916). Deep reinforcement learing is used to find optimal strategies in these two scenarios:
* Momentum trading: capture the underlying dynamics
* Arbitrage trading: utilize the hidden relation among the inputs

Several neural networks are compared: 
* Recurrent Neural Networks (GRU/LSTM)
* Convolutional Neural Network (CNN)
* Multi-Layer Perception (MLP)

More about deep reinforcement learning (deep Q-learning): 

* https://keon.io/deep-q-learning/ and its code on [GitHub](https://github.com/keon/deep-q-learning)
* Google’s DeepMind published its famous paper [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)

### Setup and Dependencies

Python version: python 3.6

#### Use Pip

    pip install -r requirements.txt

#### Anaconda

You can get all dependencies via the [Anaconda](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) environment file, [env.yml](https://github.com/golsun/deep-RL-time-series/blob/master/env.yml):

    conda env create -f env.yml

### Play with it

- before start at the 1st time, generate database: ```python sampler.py```
- call the main function: ```python main.py```


You can play with model parameters (specified in main.py), if you get good results or any trouble, please contact me at gxiang1228@gmail.com
