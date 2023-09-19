import gym
import time
import numpy as np

# For plotting 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.style.use('book.mplstyle')

# Create environment
env = gym.make("FrozenLake-v1", is_slippery=True)
env.reset()
env.render()

pi_table = [[0,0,1,0],[0,0,1,0],[0,1,0,0],[0,0,0,0],
            [0,0,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,0],
            [0,0,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,0],
            [0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]]
pi_table = np.array(pi_table)
print(pi_table)

# Run 10 episodes
num_of_episodes = 10
max_steps = 20
gamma = 0.9
epi_cum_reward = np.zeros([num_of_episodes,1])
for epi in range(num_of_episodes):
    state = env.reset()
    action = np.argmax(pi_table[state,:])
    for step in range(max_steps):
        new_state, reward, done, info = env.step(action)
        gamma_pow = np.power(gamma,step)
        epi_cum_reward[epi] = gamma_pow*reward+epi_cum_reward[epi]
        action = np.argmax(pi_table[new_state,:])
        if done:
            break
    env.close()
plt.plot(np.linspace(1,num_of_episodes,num=num_of_episodes),epi_cum_reward)
plt.grid()
plt.xlabel('Episodi nro.')
plt.ylabel('Tuotto (kumulatiivinen palkinto)')
plt.axis([0,num_of_episodes+1,0,1])
#plt.xticks(np.arange(0, 11, step=1))
plt.axis([0,num_of_episodes+1,0,0.1])
plt.show()
