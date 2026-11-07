# Q-learning on Taxi-v3

Tabular Q-learning agent for Gymnasium's `Taxi-v3` environment, written for a reinforcement learning assignment (HW6).

The taxi has to pick up a passenger and drop them at the right spot on a 5×5 grid. That gives 500 discrete states and 6 actions, which is small enough to hold the whole Q-table in memory.

## What the notebook does

1. Builds the environment and inspects the state and action spaces.
2. Trains for 25,000 episodes with an epsilon-greedy policy (α = 0.7, γ = 0.95). Epsilon decays exponentially from 1.0 toward 0.05, so early episodes explore and later ones exploit.
3. Updates the table with the standard rule:

   `Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') − Q(s,a)]`

4. Evaluates the learned greedy policy (no exploration) over 100 episodes and reports average reward and steps per episode.

## Run

```bash
pip install gymnasium numpy
jupyter notebook HW6.ipynb
```

Run the cells top to bottom. The hyperparameters (learning rate, gamma, decay rate, episode count) are all in the "Training parameters" cell if you want to experiment.
