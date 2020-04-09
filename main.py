import os
import logging
import sys
import socket
import numpy as np
import random

import rospy
import torch
import torch.nn as nn
from mpi4py import MPI

from collections import deque

from src.stageworld import StageWorld

# from model.ppo import ppo_update_stage1, generate_train_data
# from model.ppo import generate_action
# from model.ppo import transform_buffer
from src.replay_buffer import ReplayMemory
from src.sac import SAC

from tensorboardX import SummaryWriter
import datetime
import yaml

def run(param, writer):
    updates = 0
    master_send =False
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #building model
    #(self, robot_index, env_num, goal_size=0.5):
    env = StageWorld(rank)
    reward = None
    #[min][max]
    action_bound = [[0, -1], [1, 1]]
    agent = SAC(param['LASER_HIST'], 2, param, action_bound)

    if rank == 0 :
        policy_path = 'policy'

        if not os.path.exists(policy_path):
            os.makedirs(policy_path)
        
        path_actor = policy_path + filename_actor
        path_critic = policy_path + filename_critic

        if os.path.exists(path_actor) and os.path.exists(path_critic):
            logger.info('#############################')
            logger.info('########LOADING MODEL########')
            logger.info('#############################')
            
            agent.load_model(path_actor, path_critic)
        else:
            logger.info('#############################')
            logger.info('########START TRAINING#######')
            logger.info('#############################')
        

    buffer = ReplayMemory(param['BUFFER_SIZE'])
    global_update = 0
    global_step = 0

    if env.robot_index == 0:
        env.reset_world()
        #print('reset ok!')
    for i_episode in range(param['MAX_EPISODES']):
        env.reset_pose()
        print('reset pose ok!')

        env.generate_goal_point()
        print('generate goal point ok!')

        done = False
        ep_reward = 0
        step = 1

        obs = env.get_laser_observation()
        obs_stack = deque([obs, obs, obs])
        goal = np.asarray(env.get_local_goal())
        speed = np.asarray(env.get_self_speed())
        state = [obs_stack, goal, speed]
        print('set ok!')

        while not done and not rospy.is_shutdown():
            # select action
            if len(buffer) > param['BATCH_SIZE']:
                action = agent.select_action(state)
                print('POLICY ', action)
               
            else:
                #random  check type
                a =[]
                v= random.uniform(action_bound[0][0], action_bound[1][0])
                w = random.uniform(action_bound[0][1], action_bound[1][1])
                a.append(v)
                a.append(w)
                action = np.asarray(a)
                print('RANDOM ', action)


            # update paramters
            if len(buffer) > param['BATCH_SIZE']:
                #else robot sampling the data from buffer
                state_batch, action_batch, reward_batch, next_state_batch, done = buffer.sample(batch_size=param['BATCH_SIZE'])
                robot_sample_list = (state_batch, action_batch, reward_batch, next_state_batch, done)
                #Master gather data from else robot
                robot_sample_list = comm.gather(robot_sample_list, root=0)

                if robot_index == 0:
                    for sample in robot_sample_list:
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(sample, updates)

                        # writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                        # writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                        # writer.add_scalar('loss/policy', policy_loss, updates)
                        # writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                        # writer.add_scalar('entropy_temprature/alpha', alpha, updates)

                        updates += 1
                        if updates % target_updata_freq == 0:
                            master_send_robot = []
                            state_dict = agent.get_policy_state_dict()
                            for i in range(size):
                                master_send_robot.append(state_dict)
                        else:
                            master_send_robot = None
                else:
                    master_send_robot = None
                
                master_state_dict = comm.scatter(master_send_robot, root=0)
                agent.load_policy_state_dict(master_state_dict)    
                
            # execute actions
            env.control_vel(action)
            # rate.sleep()
            rospy.sleep(0.001)

            r, done, result = env.get_reward_and_terminate(step)
            ep_reward += r
            global_step += 1

            # get next state
            s_next = env.get_laser_observation()
            left = obs_stack.popleft()
            obs_stack.append(s_next)
            goal_next = np.asarray(env.get_local_goal())
            speed_next = np.asarray(env.get_self_speed())
            state_next = [obs_stack, goal_next, speed_next]

            # push buffer
            buffer.push(state, action, r, state_next, done)
            
            step += 1
            state = state_next 
            #writer.add_scalar('reward/train', episode_reward, i_episode)
        
        if env.index == 0:
            if update != 0 and update % 100 == 0:
                #torch.save(policy.state_dict(), policy_path + '/Stage1_{}'.format(update))
                agent.save_model(env_name = param['name'], suffix="01", actor_path=path_actor, critic_path=path_critic)
                logger.info('########################## model saved when update {} times#########'
                            '################'.format(global_update))
        distance = np.sqrt((env.goal_point[0] - env.init_pose[0])**2 + (env.goal_point[1]-env.init_pose[1])**2)

        logger.info('Env %02d, Goal (%05.1f, %05.1f), Episode %05d, setp %03d, Reward %-5.1f, Distance %05.1f, %s' % \
                    (env.index, env.goal_point[0], env.goal_point[1], id + 1, step, ep_reward, distance, result))
        logger_cal.info(ep_reward)

        #test
        # if i_episode % 10 == 0 :
        #     avg_reward = 0.
        #     episodes = 10
        #     for _  in range(episodes):
        #         state = env.reset()
        #         episode_reward = 0
        #         done = False
        #         while not done:
        #             action = agent.select_action(state, evaluate=True)

        #             next_state, reward, done, _ = env.step(action)
        #             episode_reward += reward


        #             state = next_state
        #         avg_reward += episode_reward
        #     avg_reward /= episodes


        #     writer.add_scalar('avg_reward/test', avg_reward, i_episode)


if __name__ == '__main__':
    f = open(r'src/config.yaml')
    param = yaml.load(f)

    #config log
    hostname = socket.gethostname()
    if not os.path.exists('./log/' + hostname):
        os.makedirs('./log/' + hostname)
    output_file = './log/' + hostname + '/output.log'
    cal_file = './log/' + hostname + '/cal.log'
    filename_actor = '/stage01actor.pth'
    filename_critic = '/stage01critic.pth'

    # config log
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(output_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    logger_cal = logging.getLogger('loggercal')
    logger_cal.setLevel(logging.INFO)
    cal_f_handler = logging.FileHandler(cal_file, mode='a')
    file_handler.setLevel(logging.INFO)
    logger_cal.addHandler(cal_f_handler)

    writer = SummaryWriter(logdir='runs/{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 'SAC',
                                                             param['policyname']))

    #training model
    try:
        run(param, writer)
    except KeyboardInterrupt:
        pass