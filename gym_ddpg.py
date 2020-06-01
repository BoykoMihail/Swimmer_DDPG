import filter_env
from ddpg import *
import gc
from collections import deque
from dm_control import suite
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from IPython import display
from dm_control_to_gym import Env_DM_Control
gc.enable()

ENV_NAME = 'Swimmer-v3'
EPISODES = 1100
TEST = 10

def obs2state(observation):
    """Converts observation dictionary to state tensor"""
    l1 = [val.tolist() for val in list(observation.values())]
    l2 = []
    for sublist in l1:
        try:
            l2.extend(sublist)
        except:
            l2.append(sublist)
    return l2
    
if __name__ == '__main__':
#    max_frame = 1001
#
#    width = 480
#    height = 480
#    video = np.zeros((10001, height, 2 * width, 3), dtype=np.uint8)
       
#    env = gym.make(ENV_NAME)
    env = suite.load(domain_name="swimmer", task_name="swimmer6")
    
    agent = DDPG(env)
    
    loss = []
    Y = []
    resultLast100Episodes = []
    X = []
    for episode in range(EPISODES):
        currentReward = 0
        time_step = env.reset()
        state = obs2state(time_step.observation)
#        state = env.reset()
        # Train
        vid = 0
        for step in range(1000):
            total_reward = 0
##            env.render()
#            action = agent.action(state)
#            next_state,reward,done,_ = env.step(action)
#            agent.perceive(state,action,reward,next_state,done)
#            state = next_state
#            currentReward += reward
#            if done:
#                break
            
#            next_state,reward,done,_ = env.step(action)
#            state = obs2state(time_step.observation)
            action = agent.action(state)
            time_step = env.step(action)
            
            reward = time_step.reward
            nextState = obs2state(time_step.observation)
            state = nextState
            terminal = time_step.last()

            currentReward += reward
            agent.perceive(state,action,reward,nextState,terminal)
            if terminal:
                break
        loss.append(currentReward)
#        if episode > 1000:
        is_solved = np.mean(loss[-10:])
        Y.append(is_solved)
        resultLast100Episodes.append(is_solved)
        X.append(episode)
        
        num_bins = 25
        fig, axs = plt.subplots(2)
         
        n, bins, patches = axs[0].hist(resultLast100Episodes, num_bins, facecolor='blue', alpha=0.5)
         
        axs[1].plot(X, Y)
         
        plt.savefig("resultHist3.png")
        
        print('episode: ', episode, 'SCORE: ',currentReward,'MEAN:', is_solved)
        
    num_bins = 25
    fig, axs = plt.subplots(2)
    
    n, bins, patches = axs[0].hist(resultLast100Episodes, num_bins, facecolor='blue', alpha=0.5)
    
    axs[1].plot(X, Y)
     
   # function to show the plot
    plt.show()
    
    plt.savefig("resultHist3.png")
