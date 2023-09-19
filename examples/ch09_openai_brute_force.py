import gym
import time
import numpy as np

# For plotting 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.style.use('book.mplstyle')
import gaud_sty

#
# PLOT 1: Non-slipper
#
rnd_seed = 1

# Create environment
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode='ansi')
env.reset(seed=rnd_seed)
#env.seed(rnd_seed)
np.random.seed(rnd_seed)
print(env.render())

pi_table = [[0,0,1,0],[0,0,1,0],[0,1,0,0],[0,0,0,0],
            [0,0,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,0],
            [0,0,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,0],
            [0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]]
pi_table = np.array(pi_table)

# Run 10 episodes
num_of_episodes = 1000
num_of_repeat = 10
max_steps = 20
gamma = 0.9
epi_cum_reward = np.zeros([num_of_episodes,num_of_repeat])
cum_reward_best = -10*np.ones([num_of_episodes,1])
best_reward = -100
pi_best = pi_table
for epi in range(num_of_episodes):
    pi_rand = np.random.rand(pi_table.shape[0],pi_table.shape[1])
    for rep in range(num_of_repeat):
        state = env.reset()
        action = np.argmax(pi_rand[state[0],:])

        for step in range(max_steps):
            new_state, reward, done, truncated, info = env.step(action)
            gamma_pow = np.power(gamma,step)
            epi_cum_reward[epi,rep] = gamma_pow*reward+epi_cum_reward[epi,rep]
            action = np.argmax(pi_rand[new_state,:])
            if done:
                break
            env.close()

    avg_reward = np.mean(epi_cum_reward[epi,:])
    if avg_reward > best_reward:
        pi_best = pi_rand
        best_reward =  avg_reward
    cum_reward_best[epi] = best_reward

if gaud_sty.color:
    plt.plot(np.linspace(1,num_of_episodes,num=num_of_episodes),cum_reward_best, 'r-', linewidth=2)
else:
    plt.plot(np.linspace(1,num_of_episodes,num=num_of_episodes),cum_reward_best, 'k-', linewidth=2)
plt.grid()
plt.xlabel('Episodi nro.')
plt.ylabel('Tuotto (kumulatiivinen palkinto)')
#plt.xticks(np.arange(0, 11, step=1))
#plt.axis([0,num_of_episodes+1,0,0.1])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch09_openai_brute_force_1_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch09_openai_brute_force_1.png')
plt.show()

#
# PLOT 2: Slippery
#


# Create environment
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode='ansi')
env.reset(seed=rnd_seed)
print(env.render())

pi_table = [[0,0,1,0],[0,0,1,0],[0,1,0,0],[0,0,0,0],
            [0,0,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,0],
            [0,0,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,0],
            [0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]]
pi_table = np.array(pi_table)

# Run 10 episodes
num_of_episodes = 1000
num_of_repeat = 10
max_steps = 20
gamma = 0.9
epi_cum_reward = np.zeros([num_of_episodes,num_of_repeat])
cum_reward_best = -10*np.ones([num_of_episodes,1])
best_reward = -100
pi_best = pi_table
for epi in range(num_of_episodes):
    pi_rand = np.random.rand(pi_table.shape[0],pi_table.shape[1])
    for rep in range(num_of_repeat):
        state = env.reset()
        action = np.argmax(pi_rand[state[0],:])

        for step in range(max_steps):
            new_state, reward, done, truncated, info = env.step(action)
            gamma_pow = np.power(gamma,step)
            epi_cum_reward[epi,rep] = gamma_pow*reward+epi_cum_reward[epi,rep]
            action = np.argmax(pi_rand[new_state,:])
            if done:
                break
            env.close()

    avg_reward = np.mean(epi_cum_reward[epi,:])
    if avg_reward > best_reward:
        pi_best = pi_rand
        best_reward =  avg_reward
    cum_reward_best[epi] = best_reward

if gaud_sty.color:
    plt.plot(np.linspace(1,num_of_episodes,num=num_of_episodes),cum_reward_best, 'r-', linewidth=2)
else:
    plt.plot(np.linspace(1,num_of_episodes,num=num_of_episodes),cum_reward_best, 'k-', linewidth=2)
plt.grid()
plt.xlabel('Episodi nro.')
plt.ylabel('Tuotto (kumulatiivinen palkinto)')
#plt.xticks(np.arange(0, 11, step=1))
#plt.axis([0,num_of_episodes+1,0,0.1])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch09_openai_brute_force_2_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch09_openai_brute_force_2.png')
plt.show()
