import gym
import time
import numpy as np

# For plotting 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.style.use('book.mplstyle')
import gaud_sty

#
# PLOT 1: non-slippery
#

# Create environment
slippery = False
env = gym.make("FrozenLake-v1", is_slippery=slippery, render_mode='ansi')
env.reset()
print(env.render())

pi_table = [[0,0,1,0],[0,0,1,0],[0,1,0,0],[0,0,0,0],
            [0,0,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,0],
            [0,0,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,0],
            [0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]]
pi_table = np.array(pi_table)

# Run 10 episodes
num_of_episodes = 10
if slippery:
    num_of_repeat = 1000
else:
    num_of_repeat = 10

max_steps = 20
gamma = 0.9
alpha = 0.5
epi_cum_reward = np.zeros([num_of_episodes,num_of_repeat])
cum_reward_best = -10*np.ones([num_of_episodes,1])
best_reward = -100
pi_best = pi_table

#pi_rand = np.random.rand(pi_table.shape[0],pi_table.shape[1])
pi_rand = np.zeros(pi_table.shape) #pi_table.shape[0],pi_table.shape[1])
for epi in range(num_of_episodes):
    for rep in range(num_of_repeat):

        # Evaluation loop
        state = env.reset()
        action = np.argmax(pi_rand[state[0],:])
        for step in range(max_steps):
            new_state, reward, done, truncated, info = env.step(action)
            gamma_pow = np.power(gamma,step)
            epi_cum_reward[epi,rep] = gamma_pow*reward+epi_cum_reward[epi,rep]
            action = np.argmax(pi_table[new_state,:])
            if done:
                break
            env.close()

        # Learning loop
        state = env.reset()
        state = state[0]
        action = np.random.randint(0,3) #np.argmax(pi_rand[state,:])
        for step in range(max_steps):
            new_state, reward, done, truncated, info = env.step(action)
            pi_rand[state,action] = pi_rand[state,action] + alpha*(reward+gamma*np.max(pi_rand[new_state,:])-pi_rand[state,action])
            action = np.random.randint(0,3) #np.argmax(pi_rand[state,:])

            if done:
                break
            env.close()


if gaud_sty.color:
    plt.plot(np.linspace(1,num_of_episodes,num=num_of_episodes),np.mean(epi_cum_reward, axis=1), 'r-', linewidth=2)
else:
    plt.plot(np.linspace(1,num_of_episodes,num=num_of_episodes),np.mean(epi_cum_reward, axis=1), 'k-', linewidth=2)
plt.grid()
plt.xlabel('Episodi nro')
plt.ylabel('Tuotto (kumulatiivinen palkinto)')
#plt.xticks(np.arange(0, 11, step=1))
if slippery:
    plt.axis([0,num_of_episodes+1,0,0.1])
else:
    plt.axis([0,num_of_episodes+1,0,1.0])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch09_openai_q_learning_1_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch09_openai_q_learning_1.png')
plt.show()

#
# PLOT 2: slippery
#

# Create environment
slippery = True
env = gym.make("FrozenLake-v1", is_slippery=slippery, render_mode='ansi')
env.reset()
print(env.render())

pi_table = [[0,0,1,0],[0,0,1,0],[0,1,0,0],[0,0,0,0],
            [0,0,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,0],
            [0,0,0,0],[0,0,0,0],[0,1,0,0],[0,0,0,0],
            [0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]]
pi_table = np.array(pi_table)

# Run 10 episodes
num_of_episodes = 10
if slippery:
    num_of_repeat = 1000
else:
    num_of_repeat = 10

max_steps = 20
gamma = 0.9
alpha = 0.5
epi_cum_reward = np.zeros([num_of_episodes,num_of_repeat])
cum_reward_best = -10*np.ones([num_of_episodes,1])
best_reward = -100
pi_best = pi_table

#pi_rand = np.random.rand(pi_table.shape[0],pi_table.shape[1])
pi_rand = np.zeros(pi_table.shape) #pi_table.shape[0],pi_table.shape[1])
for epi in range(num_of_episodes):
    for rep in range(num_of_repeat):

        # Evaluation loop
        state = env.reset()
        state = state[0]
        action = np.argmax(pi_rand[state,:])
        for step in range(max_steps):
            new_state, reward, done, truncated, info = env.step(action)
            gamma_pow = np.power(gamma,step)
            epi_cum_reward[epi,rep] = gamma_pow*reward+epi_cum_reward[epi,rep]
            action = np.argmax(pi_table[new_state,:])
            if done:
                break
            env.close()

        # Learning loop
        state = env.reset()
        state = state[0]
        action = np.random.randint(0,3) #np.argmax(pi_rand[state,:])
        for step in range(max_steps):
            new_state, reward, done, truncated, info = env.step(action)
            pi_rand[state,action] = pi_rand[state,action] + alpha*(reward+gamma*np.max(pi_rand[new_state,:])-pi_rand[state,action])
            action = np.random.randint(0,3) #np.argmax(pi_rand[state,:])

            if done:
                break
            env.close()


if gaud_sty.color:
    plt.plot(np.linspace(1,num_of_episodes,num=num_of_episodes),np.mean(epi_cum_reward, axis=1), 'r-', linewidth=2)
else:
    plt.plot(np.linspace(1,num_of_episodes,num=num_of_episodes),np.mean(epi_cum_reward, axis=1), 'k-', linewidth=2)
plt.grid()
plt.xlabel('Episodi nro')
plt.ylabel('Tuotto (kumulatiivinen palkinto)')
#plt.xticks(np.arange(0, 11, step=1))
if slippery:
    plt.axis([0,num_of_episodes+1,0,0.1])
else:
    plt.axis([0,num_of_episodes+1,0,1.0])
if gaud_sty.savefig:
    if gaud_sty.color:
        plt.savefig(gaud_sty.save_dir+'ch09_openai_q_learning_2_col.png')
    else:
        plt.savefig(gaud_sty.save_dir+'ch09_openai_q_learning_2.png')
plt.show()

