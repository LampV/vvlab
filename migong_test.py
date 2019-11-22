from maze_env import Maze
from ddpg import Agent, ActorNetwork, CriticNetwork
from ou import OUProcess
from exp_replay import ExpReplay, Step
import tensorflow as tf
import numpy as np
import sys
from RL_brain import DeepQNetwork
sys.stdout.flush()
env = Maze(2, 2)
def create_nn(state_size: int, action_size: int):
    ACTOR_LEARNING_RATE, CRITIC_LEARNING_RATE = 0.001, 0.01   # AC的学习率
    TAU, N_H1, N_H2 = 0.001, 10, 10   # 构建网络的参数
    MEM_SIZE, START_MEM, BATCH_SIZE = 500, 200, 32   # 约定经验回放池最大200，20个开始输出，每次默认获取10个
    actor = ActorNetwork(state_size=state_size, action_size=action_size,
                            lr=ACTOR_LEARNING_RATE, n_h1=N_H1, n_h2=N_H2, tau=TAU)
    critic = CriticNetwork(state_size=state_size, action_size=action_size,
                            lr=CRITIC_LEARNING_RATE, n_h1=N_H1, n_h2=N_H2, tau=TAU)
    noise = OUProcess(action_size)
    exprep = ExpReplay(mem_size=MEM_SIZE, start_mem=START_MEM, state_size=[state_size], kth=-1, batch_size=BATCH_SIZE)
    sess = tf.Session()
    agent = Agent(actor=actor, critic=critic, exprep=exprep, noise=noise, action_bound=4)
    sess.run(tf.initialize_all_variables())
    return agent, sess

def train(agent, sess, max_episode=100):
    EVALUATE_EVERY = 5
    step = 0
    for e in range(max_episode):
        env.reset()
        cur_state = env.get_obs()
        done, reward = False, 0
        while not done:
            if e > 20 and e % EVALUATE_EVERY != 0:
                # action = agent.get_action(cur_state, sess)[0]
                action = agent.get_action_noise(cur_state, sess, rate=np.power((max_episode - e)/max_episode, 3))[0]
            else:
                # action = agent.get_action(cur_state, sess)[0]
                action = agent.get_action_noise(cur_state, sess, rate=np.power((max_episode - e)/max_episode, 3))[0]
            # env.render()
            int_action = min(int(action), 3)
            int_action = max(int(int_action), 0)
            # print('action: ', action)
            next_state, reward, done, _ = env.step(int_action)
            print('state, action, reward: ', cur_state, action, reward)
            agent.add_step(Step(cur_step=cur_state, action=action, next_step=next_state, reward=reward, done=done))  # 添加到经验回放池
            cur_state = next_state
            step += 1
            if step %5 == 0:
                agent.learn_batch(sess)
        if e % 10 == 0:
            print('Print choices: ')
            for i in range(4):
                print(agent.get_action([int(j ==i) for j in range(4)], sess)[0], end=', ')
                if i % 2 == 1:
                    print('')
            print('\n', '=' * 20)
        print(e, ': ', reward)

if __name__ == '__main__':
    agent, sess = create_nn(4, 1)
    train(agent, sess)
    
