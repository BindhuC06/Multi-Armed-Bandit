import numpy as np
import random
class MultiArmedBanditEnv:
    # The Bandit Environment
    def __init__(self, num_arms=5):
        self.num_arms = num_arms
        self.true_means = np.random.uniform(-2.0, 2.0, size=self.num_arms)
        self.true_stds = np.random.uniform(0.5, 1.5, size=self.num_arms)
        self.best_arm = np.argmax(self.true_means)

    def pull_arm(self, arm_index):
        if arm_index < 0 or arm_index >= self.num_arms:
            raise ValueError(f"Invalid arm index: {arm_index}")
            
        mean = self.true_means[arm_index]
        std = self.true_stds[arm_index]
        reward = np.random.normal(loc=mean, scale=std)
        return reward


class EpsilonGreedyAgent:
    # Balancing Exploration and exploitation
    def __init__(self, num_arms, epsilon=0.1):
        self.num_arms = num_arms
        self.epsilon = epsilon  # Probability of exploration
        
        self.arm_counts = np.zeros(num_arms)  # Number of times each arm was choosen
        self.q_values = np.zeros(num_arms)  # Running average reward for each arm

    def choose_arm(self):
        # Explore
        if np.random.random()<self.epsilon:
            return np.random.choice(self.num_arms)
        # Exploit
        else:
            return np.argmax(self.q_values)

    def update_estimates(self, chosen_arm, reward):
        self.arm_counts[chosen_arm]+=1
        n=self.arm_counts[chosen_arm]
        old_value = self.q_values[chosen_arm]
        self.q_values[chosen_arm] = old_value + (1 / n) * (reward - old_value)
if __name__ == "__main__":
    np.random.seed(42)
    NUM_ARMS = 5
    STEPS = 1000
    EPSILON = 0.3
    env = MultiArmedBanditEnv(num_arms=NUM_ARMS)
    agent = EpsilonGreedyAgent(num_arms=NUM_ARMS, epsilon=EPSILON)

    for i in range(NUM_ARMS):
        print(f"Arm {i}: Normal Distribution (Mean = {env.true_means[i]:.2f}, Std Dev = {env.true_stds[i]:.2f})")
    print(f"The mathematically optimal arm to choose is: Arm #{env.best_arm}\n")

    total_reward = 0
    for step in range(STEPS):
        # 1. Agent chooses which arm to pull
        chosen_arm = agent.choose_arm()
        
        # 2. Environment generates a random reward based on that arm's distribution
        reward = env.pull_arm(chosen_arm)
        
        # 3. Agent logs the reward to optimize future decisions
        agent.update_estimates(chosen_arm, reward)
        
        total_reward += reward

    print("--- Simulation Results ---")
    print(f"Total Steps: {STEPS}")
    print(f"Total Accumulated Reward: {total_reward:.2f}")
    print(f"Average Reward Per Step: {(total_reward / STEPS):.2f}\n")
    
    print("--- Agent's Final Learned Values ---")
    for i in range(NUM_ARMS):
        print(f"Arm {i}: Pulled {int(agent.arm_counts[i])} times | Estimated Mean = {agent.q_values[i]:.2f}")
