# Provably faster bilevel Optimizers MRBO & VRBO.
Codes for paper Provably Faster Algorithms for Bilevel Optimization. Our code is built on [stocBiO](https://github.com/JunjieYang97/StocBiO). 



## Some experiment examples

In the following, we provide some experiments to demonstrate the better performance of the proposed algorithm VRBO and MRBO. 

We compare our algorithm to various hyperparameter benchmarks on MNIST dataset:

<img src="exps/bilevel_all_train_loss_nr01.png" width="350">

We compare our algorithms to various momentum-based algorithms on MNIST dataset:

<img src="exps/bilevel_mom_train_loss_nr01.png" width="350">

We evaluate the performance of our algorithm VRBO with respect to different inner-step numbers:

<img src="exps/bilevel_vrbo_train_loss_nr01.png" width="350">

This repo is still under construction and any comment is welcome! 
