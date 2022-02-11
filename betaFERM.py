import gym
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import *
from gym import spaces
import random
import pandas as pd
import time


class betaFERMEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    #metadata = {'render.modes': ['human']}
    # ><

    def __init__(self):
        super(betaFERMEnv, self, ).__init__()
        var_dict_ = {"eq_dim": 4, "init": [0.15, 5, 0.1, 0.4]
                    }

        agent_dict_ = {"action_dim": 1, "obs_dim": 5, "obs_high": [1000, 1000, 1000, 1000, 1000],
                      "obs_low": [-1000, -1000, -1000, -1000, -1000],
                      "act_high": [1], "act_low": [0],
                      "st": 0.1, "st_dt": 2, "steps": 30,
                      "dt_sp": 5
                      }
        random.seed(21)
        self.var_dict = var_dict_
        acts_dim = agent_dict_.get("action_dim")
        self.obs_dim = agent_dict_.get("obs_dim")
        acts_high = agent_dict_.get("act_high")
        acts_low = agent_dict_.get("act_low")
        obs_high = agent_dict_.get("obs_high")
        obs_low = agent_dict_.get("obs_low")
        self.total_step = agent_dict_.get("steps")
        self.sampling_time = agent_dict_.get("st")
        self.sampling_diff = agent_dict_.get("st_dt")
        self.eq_dim = self.var_dict.get("eq_dim")
        self.observation = np.zeros(self.obs_dim)
        self.OBJ = Fermentator(var_dict_)
        self.ep_buffer = EpisodeBuffer(self.OBJ)
        self.action_space = spaces.Box(low=-np.ones(acts_dim), high=np.ones(acts_dim), shape=(acts_dim,),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.ones(self.obs_dim), high=np.ones(self.obs_dim), shape=(self.obs_dim,),
                                            dtype=np.float32)
        diff_a = np.array([a_i - b_i for a_i, b_i in zip(acts_high, acts_low)])
        diff_obs = np.array([a_i - b_i for a_i, b_i in zip(obs_high, obs_low)])
        self.acts_mean = diff_a / 2
        self.observation_mean = diff_obs/2
        self.action_shift = acts_low + self.acts_mean
        self.observation_shift = obs_low + self.observation_mean
        self.RENDER = False
        self.input_dummy = np.zeros(self.eq_dim)

    def step(self, action_norm):
        action = action_norm * self.acts_mean + self.action_shift
        self.previous = self.OBJ.get_obs()
        self.OBJ.set_action(action)
        self.input_dummy = [self.OBJ.C, self.OBJ.S, self.OBJ.I, self.OBJ.V]
        result = odeint(self.OBJ.step, self.input_dummy, np.linspace(0, self.sampling_time, self.sampling_diff))

        self.OBJ.C = result[self.sampling_diff - 1][0]
        self.OBJ.S = result[self.sampling_diff - 1][1]
        self.OBJ.I = result[self.sampling_diff - 1][2]
        self.OBJ.V = result[self.sampling_diff - 1][3]

        self.OBJ.d1 = (self.OBJ.S - self.previous[1]) / self.sampling_time
        self.OBJ.d2 = (self.OBJ.d1 - self.OBJ.previous_d1) / self.sampling_time
        #Update state
        #self.OBJ.product = self.OBJ.P*self.OBJ.V
        self.observation = self.OBJ.get_obs()
        ##
        reward, done = self.reward_function()
        self.OBJ.last_rew += reward
        self.ep_buffer.update_history()
        self.observation = self.normalize_obs(self.observation)
        self.OBJ.clock += 1
        return np.array(self.observation).astype(np.float32), reward, done, {"ok": True}

    def normalize_obs(self, obs):
        obs_n = (obs-self.observation_shift)/self.observation_mean
        return obs_n

    def denormalize_obs(self, obs_n):
        obs = (obs_n * self.observation_mean) + self.observation_shift
        return obs

    # < >
    def reward_function(self):
        done_ = False
        reward_ = 0
        if self.OBJ.clock>0:
            reward_ = -abs(1- self.OBJ.I)**0.7 - 0.2*self.OBJ.clock**-1
        if self.OBJ.V > 1.2 or self.OBJ.clock>self.total_step:
            done_ = True
        return np.float64(reward_), done_

    def render(self, mode='human'):

        if self.ep_buffer.index > 2:
            ax, bx = 3, 3
            plt.subplot(ax, bx, 1)
            plt.plot(self.ep_buffer.C_history)
            plt.title("C")

            plt.subplot(ax, bx, 2)
            plt.plot(self.ep_buffer.action_history)
            plt.title("Qin")

            plt.subplot(ax, bx, 3)
            plt.plot(self.ep_buffer.S_history)
            plt.title("S")

            plt.subplot(ax, bx, 4)
            plt.plot(self.ep_buffer.V_history)
            plt.title("V")


            plt.subplot(ax, bx, 5)
            plt.plot(self.ep_buffer.rew_history)
            plt.title("rew")

            plt.subplot(ax, bx, 6)
            plt.plot(self.ep_buffer.I_history)
            plt.title("I")

            plt.subplot(ax, bx, 7)
            plt.plot(self.ep_buffer.prod_history)
            plt.title("prod")

            plt.draw()
            plt.pause(0.0000000000001)
            plt.clf()

    def close(self):
        pass

    def reset(self):
        self.render()
        self.OBJ.reset()
        self.observation = self.OBJ.get_obs()
        self.ep_buffer.reset_history()
        self.observation = self.normalize_obs(self.observation)
        return self.observation # reward, done, info can't be included


class EpisodeBuffer:
    def __init__(self, obj_):
        self.C_history = list()
        self.S_history = list()
        self.I_history = list()
        self.V_history = list()
        self.rew_history = list()
        self.prod_history = list()
        self.action_history = list()
        self.obj = obj_
        self.index = 0

    def update_history(self):
        self.C_history.append(np.float64(self.obj.C))
        self.S_history.append(np.float64(self.obj.S))
        self.I_history.append(np.float64(self.obj.I))
        self.V_history.append(np.float64(self.obj.V))
        self.prod_history.append(np.float64(self.obj.product))
        self.rew_history.append(np.float64(self.obj.last_rew))
        self.action_history.append(self.obj.u)
        self.index += 1

    def reset_history(self):
        self.C_history.clear()
        self.S_history.clear()
        self.rew_history.clear()
        self.I_history.clear()
        self.prod_history.clear()
        self.V_history.clear()
        self.action_history.clear()
        self.index = 0

    def render(self, mode='human'):
        ax, bx = 3, 2
        plt.subplot(ax, bx, 1)
        plt.plot(self.C_history)
        plt.title("C")

        plt.subplot(ax, bx, 2)
        plt.plot(self.action_history)
        plt.title("Qin")

        plt.subplot(ax, bx, 3)
        plt.plot(self.S_history)
        plt.title("S")

        plt.subplot(ax, bx, 4)
        plt.plot(self.I_history)
        plt.title("I")

        plt.subplot(ax, bx, 5)
        plt.plot(self.V_history)
        plt.title("V")

        plt.subplot(ax, bx, 6)
        plt.plot(self.rew_history)
        plt.title("rew")

class Fermentator:
    def __init__(self,var_dict_):
        self.var_dict = var_dict_
        self.u = np.array([0], dtype=np.float64)  # action non-normalized
        self.eq_dim = self.var_dict.get("eq_dim")
        self.init = np.array(self.var_dict.get("init"), dtype=np.float64)  # mol/L
        self.clock = 0
        self.C = self.init[0]
        self.S = self.init[1]
        self.I = self.init[2]
        self.V = self.init[3]
        self.previous_d1 = np.zeros(1)
        self.d1 = np.zeros(1)
        self.d2 = np.zeros(1)
        self.max = np.zeros(1)
        self.product = np.zeros(1)
        self.last_rew = np.zeros(1)
        self.C_dummy = np.zeros(1)
        self.S_dummy = np.zeros(1)
        self.I_dummy = np.zeros(1)
        self.V_dummy = np.zeros(1)
        self.out_dummy = np.zeros(self.eq_dim)

    def reset(self):
        self.u = np.array([0], dtype=np.float64)  # action non-normalized
        self.eq_dim = self.var_dict.get("eq_dim")
        self.init = np.array(self.var_dict.get("init"), dtype=np.float64)  # mol/L

        self.C = self.init[0]
        self.S = self.init[1]
        self.I = self.init[2]
        self.V = self.init[3]

        self.product = np.zeros(1)
        self.last_rew = np.zeros(1)
        self.clock = 0
        self.d1 = np.zeros(1)
        self.d2 = np.zeros(1)

    def step(self, var, t):
        # ODE X S P V CO
        # K:  K1_0 K2_1 KS_2 KO_3 KD_4 KH_5 KI_6 KP_7
        # gamma_s/o : gammax gammap
        self.C_dummy = var[0]
        self.S_dummy = var[1]
        self.I_dummy = var[2]
        self.V_dummy = var[3]

        rr = (0.55*self.S_dummy)/(0.05 + self.S_dummy)
        pi = (6.25*self.S_dummy)/(0.1 + self.S_dummy + 2*self.S_dummy**2)
        rt1 = (1.25*self.S_dummy)/(0.95+self.S_dummy)
        if rt1 > rr:
            rt = rt1
        else:
            rt = rr
        rf = rt - rr
        ycr, ycf, kd = 0.6, 0.15, 1.85
        self.out_dummy[3] = self.u  # dv
        self.out_dummy[0] = (rr*ycr + rf*ycf)*self.C_dummy - self.out_dummy[3]*self.C_dummy/self.V_dummy
        self.out_dummy[1] = self.u*self.S_dummy/self.V_dummy - rt*self.C_dummy - self.out_dummy[3]*self.S_dummy/self.V_dummy
        self.out_dummy[2] = (pi - kd*self.I_dummy) - self.I_dummy*(rr*ycr + rf*ycf)
        self.product = -kd*self.I_dummy - self.I_dummy*(rr*ycr + rf*ycf)
        return self.out_dummy

    def set_action(self, action_n):
        self.u = action_n

    def get_obs(self,):
        return [self.C, self.S, self.I, self.V, self.product]



if __name__ == '__main__':

    # ODE X P S V CO
    # K:  K1 K2 KS KO KD KH KI KP
    # gamma_s/o : gammax gammap
    # m ms mo

    var_dict = {"eq_dim": 4, "init": [0.15, 5, 0.1, 0.4]
                }

    agent_dict = {"action_dim": 1, "obs_dim": 4, "obs_high": [ 1000, 1000, 1000, 1000],
                  "obs_low": [-1000, -1000, -1000, -1000],
                  "act_high": [1], "act_low": [0],
                  "st": 0.1, "st_dt": 2, "steps": 100,
                  "dt_sp": 5
                  }

    #np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    env = CustomEnv(var_dict, agent_dict)
    len = agent_dict.get("steps")
    action_vec = np.zeros(len)
    state_vec = np.zeros(len)
    time_vec = np.linspace(0, 5, len)
    for i in range(agent_dict.get("steps")):
        action = env.action_space.sample()
        action = -1
        next_state, reward, done, info = env.step(action)
    env.ep_buffer.render()
    plt.pause(5000)
    plt.show()