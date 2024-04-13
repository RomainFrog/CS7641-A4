Code base is publicly available here: https://github.com/RomainFrog/CS7641-A4


To run the code, first create an environment with the following libraries:
- numpy
- pandas
- matplotlib
- seaborn
- tqdm
- gymnasium
- pathos
- bettermdptools (modified version provided in the repo)


The code is provided with the following architecture:

- utils.py  (auxiliary functions)
- frozen_lake_2.py (implementation of modified frozen lake with custom rewards and random starts)
- mountainCartWrapper2.py (Mountaincart wrapper for discretized state space made in collaboration with Quentin Fitte-Rey, Romain Froger and Gauthier Roy)
- frozen_lake_analysis.ipynb (notebook with results for VI and PI on frozen lake)
- frozen_lake_q_learning.ipynb (notebook with results for Q-learning on frozen lake)
- mdp_mountain_cart.ipynb (notebook with all results on Mountain Cart)

