# EWC-sample-PMNIST
Run [EWC](https://arxiv.org/abs/1612.00796) on Permuted MNIST in Domian-Incremental Learning scenario with TF 2.3. 

Followed [the A-Gem paper](https://arxiv.org/abs/1812.00420), the model is set to a simple fully-connected neuron network with two 256-wide hidden layers. Run EWC with different lambda value and compare these results with Sequential Fine Tuning (SFT), i.e. training tasks sequentailly without using EWC.

The curves of averge accuracy on 10 tasks are showed as follows.

![Avg Acc](https://github.com/QiaoZhongzheng/EWC-sample-PMNIST/blob/main/saved/images/avg_acc_curves.png)

Compare the accuracy curves of SFT with EWC,lambda=10:

![SFT](https://github.com/QiaoZhongzheng/EWC-sample-PMNIST/blob/main/saved/images/SFT.png)

![EWC_10](https://github.com/QiaoZhongzheng/EWC-sample-PMNIST/blob/main/saved/images/EWC_10.png)
