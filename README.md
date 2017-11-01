
# Deep Reinforcement Learning with Baysian Optimization (Deep-RL-BO) 

Intelligent Systems (INTSYS) Assignment 2

# Tutorials/Practicals

To start the IPython notebooks simply run start_ipython.sh in the "Tutorials/IPython Notebooks" folder in your Mac OS or Linux shell, and then open one of the notebooks from your browser. Obviously this requires that IPython be installed in your Python distribution. 
There are some nice walk-throughs on how Bayesian Optimization works in the "Tutorials/Walkthroughs" folder. The one by B. Shariari et al. is excellent for mathematical grounding, and the hyperopt tutorial by j. Bergstra et al. is excellent for getting knee-deep into optimizing scikit-learn machine learning algorithms with TPE.
Finally, J. Snoek et al. have lovely tutorial on Bayesian optimization with an application to user modelling and sensor selection. For the practicals, a short description of each practical is as follows:

	
# Deep Reinforcement Learning: Cartpole and Atari

The code for the Cartpole and Atari game demos is in the Code directory. These programs optimize the hyperparameters for a simple Cartpole example and the Atari game Breakout. One can see that initial model guesses are poor and do not train adequately in the alloted time.
Later configurations vastly outperform previous ones.

# Presentation

Presentation slides are in the Presentation folder

# References 

Some quick references of TPE, SMAC, Fabolas, ReMBO, and MTBO are below:

	Shahriari, B., Swersky, K., Wang, Z., Adams, R. P., & de Freitas, N. (2016). 
	Taking the human out of the loop: A review of bayesian optimization. 
	Proceedings of the IEEE, 104(1), 148-175

    Hutter, F. and Hoos, H. H. and Leyton-Brown, K.
    Sequential Model-Based Optimization for General Algorithm Configuration
    In: Proceedings of the conference on Learning and Intelligent OptimizatioN (LION 5)

    Bergstra, J., Yamins, D., & Cox, D. D. (2013).
    Hyperopt: A python library for optimizing the hyperparameters of machine learning algorithms.
    In Proceedings of the 12th Python in Science Conference (pp. 13-20).

    Klein, A., Falkner, S., Bartels, S., Hennig, P., & Hutter, F. (2016).
    Fast bayesian optimization of machine learning hyperparameters on large datasets.
    arXiv preprint arXiv:1605.07079.

    Swersky, K., Snoek, J., & Adams, R. P. (2013).
    Multi-task bayesian optimization.
    In Advances in neural information processing systems (pp. 2004-2012).
	
	Wang, Z., Zoghi, M., Hutter, F., Matheson, D., & De Freitas, N. (2013, August). 
	Bayesian Optimization in High Dimensions via Random Embeddings.
	In IJCAI (pp. 1778-1784)

References are in the References folder and contain links to the source material for the practicals in the "Tutorials/IPython Notebooks" folder.