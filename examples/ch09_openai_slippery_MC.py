import gym
import time
import numpy as np

# For plotting 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.style.use('book.mplstyle')
import gaud_sty

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
num_of_repeat = 1000
max_steps = 20
gamma = 0.9
epi_cum_reward = np.zeros([num_of_episodes,num_of_repeat])
for epi in range(num_of_episodes):
    for rep in range(num_of_repeat):
        state = env.reset()
        action = np.argmax(pi_table[state[0],:])

        for step in range(max_steps):
            new_state, reward, done, truncated, info = env.step(action)
            gamma_pow = np.power(gamma,step)
            epi_cum_reward[epi,rep] = gamma_pow*reward+epi_cum_reward[epi,rep]
            action = np.argmax(pi_table[new_state,:])
            if done:
                break
            env.close()

if gaud_sty.color==True:
    plt.plot(np.linspace(1,num_of_episodes,num=num_of_episodes),np.mean(epi_cum_reward, axis=1), 'r-', linewidth=2)
else:
    plt.plot(np.linspace(1,num_of_episodes,num=num_of_episodes),np.mean(epi_cum_reward, axis=1), 'k-', linewidth=2)
plt.grid()
plt.xlabel('Episodi nro')
plt.ylabel('Tuotto (kumulatiivinen palkinto)')
plt.xticks(np.arange(0, 11, step=1))
plt.axis([0,num_of_episodes+1,0,0.1])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch09_openai_slippery_MC_1_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch09_openai_slippery_MC_1.png')
plt.show()
