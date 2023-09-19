import gym
import time
import numpy as np

# For plotting 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.style.use('book.mplstyle')
import gaud_sty

# Create environment
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
env.reset()
print(env.render())

pi_table = [[0,0,1,0],[0,0,1,0],[0,1,0,0],[0,0,0,0],
            [0,0,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,0],
            [0,0,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,0],
            [0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]]
pi_table = np.array(pi_table)
print(pi_table)

# Run 10 episodes
num_of_episodes = 10
max_steps = 10
gamma = 0.9
epi_cum_reward = np.zeros([num_of_episodes,1])
for epi in range(num_of_episodes):
    state = env.reset()
    action = np.argmax(pi_table[state[0],:])
    for step in range(max_steps):
        #new_state, reward, done, info = env.step(action)
        new_state, reward, done, truncated, info  = env.step(action)
        gamma_pow = np.power(gamma,step)
        epi_cum_reward[epi] = gamma_pow*reward+epi_cum_reward[epi]
        action = np.argmax(pi_table[new_state,:])
        if done:
            break
    env.close()
if gaud_sty.color==True:
    plt.plot(np.linspace(1,10,num=10),epi_cum_reward,'r-',linewidth=4)
else:
    plt.plot(np.linspace(1,10,num=10),epi_cum_reward,'k-',linewidth=4)
plt.grid()
plt.xlabel('Episodi nro')
plt.ylabel('Tuotto (kumulatiivinen palkinto)')
plt.xticks(np.arange(0, 11, step=1))
plt.axis([0,num_of_episodes+1,0,1])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch09_openai_1_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch09_openai_1.png')
plt.show()

#
# PLOT 2
#

# Create environment
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode='ansi')
env.reset()
print(env.render())

pi_table = [[0,0,1,0],[0,0,1,0],[0,1,0,0],[0,0,0,0],
            [0,0,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,0],
            [0,0,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,0],
            [0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]]
pi_table = np.array(pi_table)
print(pi_table)

# Run 10 episodes
num_of_episodes = 10
max_steps = 10
gamma = 0.9
epi_cum_reward = np.zeros([num_of_episodes,1])
for epi in range(num_of_episodes):
    state = env.reset()
    action = np.argmax(pi_table[state[0],:])
    for step in range(max_steps):
        new_state, reward, done, truncated, info = env.step(action)
        gamma_pow = np.power(gamma,step)
        epi_cum_reward[epi] = gamma_pow*reward+epi_cum_reward[epi]
        action = np.argmax(pi_table[new_state,:])
        if done:
            break
    env.close()
if gaud_sty.color==True:
    plt.plot(np.linspace(1,10,num=10),epi_cum_reward,'r-',linewidth=4)
else:
    plt.plot(np.linspace(1,10,num=10),epi_cum_reward,'k-',linewidth=4)
plt.grid()
plt.xlabel('Episodi nro')
plt.ylabel('Tuotto (kumulatiivinen palkinto)')
plt.xticks(np.arange(0, 11, step=1))
plt.axis([0,num_of_episodes+1,0,1])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch09_openai_2_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch09_openai_2.png')
plt.show()
