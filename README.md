
This repo is the source code for:

### **Deep reinforcement learning for time series: playing idealized trading games**

You can find the paper [here](https://arxiv.org/abs/1803.03916), and please reach me at gxiang1228@gmail.com for any questions/comments!

 In this paper I explored deep reinforcement learing as a method to find the optimal strategies for trading. I compared several neural networks: Stacked Gated Recurrent Unit (GRU), stacked Long Short-Term Memory (LSTM), stacked Convolutional Neural Network (CNN), and multi-layer perception (MLP). I designed two simple trading games aiming to test if the trained agent can: 
* capture the underlying dynamics (to be used in momentum trading)
* utilize the hidden relation among the inputs (to be used in arbitrage trading)

It turns out that GRU performs best. However as these are just simplified worlds for the agent to play with, more further investigation is deserved.

### Dependencies

After you get my repo, you need packages like keras, tensorflow, keras, h5py. But don't worry about these dependencies, I've created a [Anaconda environment](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) file, env.yml, for you to run my repo. Simply do this in your terminal:

    conda env create -f env.yml

### Play with it
Just call the main function

    python main.py

But you can choose model (MLP, CNN, or GRU) and parameters by playing with the main function, and you can play with sampler.py to generate a different artifical input datasets to train and test.
