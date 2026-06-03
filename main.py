import numpy as np
import random
import matplotlib.pyplot as plt

class EpsilonGreedyAgent:
    def __init__(self,num_act, epsilon=0.1):
        self.num_act=num_act
        self.epsilon=epsilon

        self.action_count=np.zeros(num_act)
        self.qvals=np.zeros(num_act)

    def action(self):
        if np.random.random()<self.epsilon:
            return random.randrange(self.num_act)
        else:
            return np.argmax(self.qvals)
        
    def update(self,action,reward):
        self.action_count[action]+=1
        n = self.action_count[action]
        # Incremental update rule: Q(a) = Q(a) + 1/n * (Target - Q(a))
        self.qvals[action] += (reward - self.qvals[action]) / n


true_rewards = [2,1,4,6]
agent = EpsilonGreedyAgent(num_act=4, epsilon=0.1)
cumulative_rewards = []
total = 0
for i in range(300):
    action = agent.action()
    reward = true_rewards[action] + np.random.normal(0, 0.5)  # add noise
    agent.update(action, reward)
    total += reward
    cumulative_rewards.append(total / (i + 1))  # running average

plt.plot(cumulative_rewards)
plt.axhline(y=5, color='b', linestyle='--', label='Optimal (action 1)')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('Epsilon-Greedy Bandit Learning Curve')
plt.legend()
plt.show()

print("Final Q-values:", np.round(agent.qvals, 2))