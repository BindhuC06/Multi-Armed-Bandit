import numpy as np
import matplotlib.pyplot as plt
import os
from main import MultiArmedBanditEnv, EpsilonGreedyAgent

# Config -- copy from main.py
NUM_ARMS = 5
STEPS = 1000
EPSILON = 0.1
np.random.seed(42)

os.makedirs("results", exist_ok=True)

# Run simulation
env = MultiArmedBanditEnv(num_arms=NUM_ARMS)
agent = EpsilonGreedyAgent(num_arms=NUM_ARMS, epsilon=EPSILON)

rewards_history = []
optimal_selections = []
total = 0

for step in range(STEPS):
    arm = agent.choose_arm()
    reward = env.pull_arm(arm)
    agent.update_estimates(arm, reward)
    total += reward
    rewards_history.append(total / (step + 1))
    optimal_selections.append(1 if arm == env.best_arm else 0)

# Plot 1 — The Learning curve
plt.figure(figsize=(10, 5))
plt.plot(rewards_history, label='Agent average reward')
plt.axhline(y=env.true_means[env.best_arm], color='r', linestyle='--', label='Optimal arm mean')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('Epsilon-Greedy: Learning Curve')
plt.legend()
plt.tight_layout()
plt.savefig("results/learning_curve.png", dpi=150)
plt.close()
print("Saved: results/learning_curve.png")

# Plot 2 — Optimal arm selection rate over time
cumulative_optimal = np.cumsum(optimal_selections) / (np.arange(STEPS) + 1)
plt.figure(figsize=(10, 5))
plt.plot(cumulative_optimal, color='green', label='% Optimal arm chosen')
plt.axhline(y=1 - EPSILON, color='r', linestyle='--', label=f'Theoretical max ({(1-EPSILON)*100:.0f}%)')
plt.xlabel('Steps')
plt.ylabel('Optimal Arm Selection Rate')
plt.title('Epsilon-Greedy: Optimal Arm Selection Over Time')
plt.legend()
plt.tight_layout()
plt.savefig("results/optimal_selection_rate.png", dpi=150)
plt.close()
print("Saved: results/optimal_selection_rate.png")

# Plot 3 — Final estimated vs true values per arm
x = np.arange(NUM_ARMS)
width = 0.35
plt.figure(figsize=(8, 5))
plt.bar(x - width/2, env.true_means, width, label='True Mean', color='steelblue')
plt.bar(x + width/2, agent.q_values, width, label='Estimated Mean', color='orange')
plt.xlabel('Arm')
plt.ylabel('Reward Mean')
plt.title('True vs Estimated Arm Values')
plt.xticks(x, [f'Arm {i}' for i in range(NUM_ARMS)])
plt.legend()
plt.tight_layout()
plt.savefig("results/true_vs_estimated.png", dpi=150)
plt.close()
print("Saved: results/true_vs_estimated.png")