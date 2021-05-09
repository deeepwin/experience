---
layout: post
title:  "CPU and GPU Selection for Reinforcement Learning Workstation"
date:   2021-05-10 18:59:37 +0100
categories: workstation
---

# CPU and GPU Selection for Reinforcement Learning Workstation

## situation

You are fed up with all the deep learning cloud platforms like GCP and want to build your own deep learning rig, but you are not sure what CPU and GPU to buy? 

This post present my own experience and learnings on building you own workstation. But first, here some additional useful ressources:

[Which GPU(s) to Get for Deep Learning: My Experience and Advice for Using GPUs in Deep Learning](https://timdettmers.com/2020/09/07/which-gpu-for-deep-learning/)

[Why Don't You Build a RL Computer?](https://rivesunder.gitlab.io/rl/2020/04/04/build_a_rl_pc_1.html)

Nevertheless even after reading these posts, it was not clear to me what CPU and GPU to choose. I hope I can answer this question to you.

## key take aways from this post

If you have not enough time to read the full blog, this are the main take aways:

* Select a CPU with as many cores as possible, CPU power is important in RL. Invest enough money into the CPU.
* Install a lot of CPU memory. If you parallize training with separate processess, each one will load Tensorflow and require memory.
* Use a GPU with a lot of memory at least 11GB. Memory is the first limitation on a GPU.
* The utilization of the GPU is mostly defined by the size of your neural network.
* If your network is small, you do not need a power full GPU. Start with a GTC 1080 Ti. RTX 2080 Ti is nice, I guess RTX 3090 even better - 24GB!

## Long 
