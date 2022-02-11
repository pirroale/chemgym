import gym
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import *
from gym import spaces
import random
import pandas as pd
import time


class alfaFERMEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    #metadata = {'render.modes': ['human']}
    # ><

    def __init__(self):
        super(alfaFERMEnv, self, ).__init__()
        #Header
        random.seed(21)
        var_dict_ = {"eq_dim": 5, "init": [0.05, 0, 10, 60, 0.037], "growth": [0.1224, -60],
                    "death": [1.9 * 10 ** -3, -340],
                    "K": [1 * 10 ** -10, 1.3 * 10 ** -4, 0.1828, 0.0352, 0.0368, 4 * 10 ** -4, 0.1, 2 * 10 ** -4],
                    "xmax": [0.87],
                    "gamma_s": [0.25, 0.68],
                    "m": [0.0624, 4 * 10 ** -3], "kla": [1.27],
                    "gamma_o": [43.5, 253.3], "mu": [0.05],
                    "ph": [-7], "T": [303]
                    }

        agent_dict_ = {"action_dim": 1, "obs_dim": 5, "obs_high": 1000 * np.ones(5),
                      "obs_low": -1000 * np.ones(5),
                      "act_high": [1], "act_low": [0],
                      "st": 12, "st_dt": 2, "steps": 10,
                      "dt_sp": 5
                      }
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
        self.action_space = spaces.Box(low=-np.ones(acts_dim), high=np.ones(acts_dim), shape=(acts_dim,), dtype=np.float32)
        #self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=-np.ones(self.obs_dim), high=np.ones(self.obs_dim), shape=(self.obs_dim,), dtype=np.float32)
        diff_a = np.array([a_i - b_i for a_i, b_i in zip(acts_high, acts_low)])
        diff_obs = np.array([a_i - b_i for a_i, b_i in zip(obs_high, obs_low)])
        self.acts_mean = diff_a / 2
        self.observation_mean = diff_obs/2
        self.action_shift = acts_low + self.acts_mean
        self.observation_shift = obs_low + self.observation_mean
        self.RENDER = False
        self.input_dummy = np.zeros(self.eq_dim)

    def step(self, action_norm):
        self.OBJ.prev_obs = self.OBJ.get_obs()
        action = action_norm * self.acts_mean + self.action_shift
        self.OBJ.set_action(action)
        self.input_dummy = [self.OBJ.X, self.OBJ.P, self.OBJ.S, self.OBJ.V, self.OBJ.CO]
        result = odeint(self.OBJ.step, self.input_dummy, np.linspace(0, self.sampling_time, self.sampling_diff))

        self.OBJ.X = result[self.sampling_diff - 1][0]
        self.OBJ.P = result[self.sampling_diff - 1][1]
        self.OBJ.S = result[self.sampling_diff - 1][2]
        self.OBJ.V = result[self.sampling_diff - 1][3]
        self.OBJ.CO = result[self.sampling_diff - 1][4]


        #Update state
        self.observation = self.OBJ.get_obs()
        self.ep_buffer.update_history()
        ##
        reward, done = self.reward_function()
        self.OBJ.last_rew = reward
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
        if self.OBJ.clock > 3:
            last2_net_prod = self.ep_buffer.P_history[-3] * self.ep_buffer.V_history[-3]
            last_net_prod = self.ep_buffer.P_history[-2]*self.ep_buffer.V_history[-2]
            act_net_prod = self.ep_buffer.P_history[-1]*self.ep_buffer.V_history[-1]
            reward_ = 100*((act_net_prod - last_net_prod) - (last_net_prod - last2_net_prod))
        if (self.OBJ.V > 99) or (self.OBJ.clock > self.total_step):
            done_ = True
        return np.float64(reward_), done_

    def render(self, mode='human'):

        if self.ep_buffer.index > 2:
            ax, bx = 3, 2
            plt.subplot(ax, bx, 1)
            plt.plot(self.ep_buffer.X_history)
            plt.title("X")

            plt.subplot(ax, bx, 2)
            plt.plot(self.ep_buffer.action_history)
            plt.title("Qin")

            plt.subplot(ax, bx, 3)
            plt.plot(self.ep_buffer.S_history)
            plt.title("S")

            plt.subplot(ax, bx, 4)
            plt.plot(self.ep_buffer.P_history)
            plt.title("P")

            plt.subplot(ax, bx, 5)
            plt.plot(self.ep_buffer.V_history)
            plt.title("V")

            plt.subplot(ax, bx, 6)
            plt.plot(self.ep_buffer.CO_history)
            plt.title("CO")

            # plt.subplot(ax, bx, 7)
            # plt.plot(self.ep_buffer.prod1_history)
            # plt.plot(self.ep_buffer.prod2_history)
            # plt.plot(self.ep_buffer.prod3_history)
            # plt.plot(self.ep_buffer.prod4_history)
            # plt.title("prod")
            #
            # plt.subplot(ax, bx, 8)
            # plt.plot(self.ep_buffer.cons1_history)
            # plt.plot(self.ep_buffer.cons2_history)
            # plt.plot(self.ep_buffer.cons3_history)
            # plt.plot(self.ep_buffer.cons4_history)
            # plt.title("cons")
            #
            # plt.subplot(ax, bx, 9)
            # plt.plot(self.ep_buffer.act_t1_history)
            # plt.plot(self.ep_buffer.act_t2_history)
            # plt.plot(self.ep_buffer.act_t3_history)
            # plt.plot(self.ep_buffer.act_t4_history)
            # plt.title("act_t")

            plt.subplot(ax, bx, 10)
            plt.plot(self.ep_buffer.rew_history)
            plt.title("rew")

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
        self.X_history = list()
        self.S_history = list()
        self.P_history = list()
        self.V_history = list()
        self.CO_history = list()
        self.rew_history = list()
        self.prod1_history = list()
        self.prod2_history = list()
        self.prod3_history = list()
        self.prod4_history = list()
        self.cons1_history = list()
        self.cons2_history = list()
        self.cons3_history = list()
        self.cons4_history = list()
        self.act_t1_history = list()
        self.act_t2_history = list()
        self.act_t3_history = list()
        self.act_t4_history = list()
        self.action_history = list()
        self.obj = obj_
        self.index = 0

    def update_history(self):
        self.X_history.append(np.float64(self.obj.X))
        self.S_history.append(np.float64(self.obj.S))
        self.P_history.append(np.float64(self.obj.P))
        self.V_history.append(np.float64(self.obj.V))
        self.prod1_history.append(self.obj.prod[0])
        self.prod2_history.append(self.obj.prod[1])
        self.prod3_history.append(self.obj.prod[2])
        self.prod4_history.append(self.obj.prod[4])
        self.cons1_history.append(self.obj.cons[0])
        self.cons2_history.append(self.obj.cons[1])
        self.cons3_history.append(self.obj.cons[2])
        self.cons4_history.append(self.obj.cons[4])
        self.act_t1_history.append(self.obj.act_t[0])
        self.act_t2_history.append(self.obj.act_t[1])
        self.act_t3_history.append(self.obj.act_t[2])
        self.act_t4_history.append(self.obj.act_t[4])
        self.CO_history.append(np.float64(self.obj.CO))
        self.rew_history.append(np.float64(self.obj.last_rew))
        self.action_history.append(self.obj.u)
        self.index += 1

    def reset_history(self):
        self.X_history.clear()
        self.S_history.clear()
        self.rew_history.clear()
        self.P_history.clear()
        self.CO_history.clear()
        self.prod1_history.clear()
        self.prod2_history.clear()
        self.prod3_history.clear()
        self.prod4_history.clear()
        self.cons1_history.clear()
        self.cons2_history.clear()
        self.cons3_history.clear()
        self.cons4_history.clear()
        self.act_t1_history.clear()
        self.act_t2_history.clear()
        self.act_t3_history.clear()
        self.act_t4_history.clear()
        self.V_history.clear()
        self.action_history.clear()
        self.index = 0

    def render(self, mode='human'):
        ax, bx = 5, 2
        plt.subplot(ax, bx, 1)
        plt.plot(self.X_history)
        plt.title("X")

        plt.subplot(ax, bx, 2)
        plt.plot(self.action_history)
        plt.title("Qin")

        plt.subplot(ax, bx, 3)
        plt.plot(self.S_history)
        plt.title("S")

        plt.subplot(ax, bx, 4)
        plt.plot(self.P_history)
        plt.title("P")

        plt.subplot(ax, bx, 5)
        plt.plot(self.V_history)
        plt.title("V")

        plt.subplot(ax, bx, 6)
        plt.plot(self.CO_history)
        plt.title("CO")

        plt.subplot(ax, bx, 7)
        plt.plot(self.prod1_history)
        plt.plot(self.prod2_history)
        plt.plot(self.prod3_history)
        plt.plot(self.prod4_history)
        plt.title("prod")

        plt.subplot(ax, bx,8)
        plt.plot(self.cons1_history)
        plt.plot(self.cons2_history)
        plt.plot(self.cons3_history)
        plt.plot(self.cons4_history)
        plt.title("cons")

        plt.subplot(ax, bx, 9)
        plt.plot(self.act_t1_history)
        plt.plot(self.act_t2_history)
        plt.plot(self.act_t3_history)
        plt.plot(self.act_t4_history)
        plt.title("act_t")

        plt.subplot(ax, bx, 10)
        plt.plot(self.rew_history)
        plt.title("rew")


class Fermentator:
    def __init__(self,var_dict_):
        self.var_dict = var_dict_
        self.u = np.array([0], dtype=np.float64)  # action non-normalized
        self.eq_dim = self.var_dict.get("eq_dim")
        self.init = np.array(self.var_dict.get("init"), dtype=np.float64)  # mol/L
        self.growth = np.array(self.var_dict.get("growth"), dtype=np.float64)  # L
        self.death = np.array(self.var_dict.get("death"), dtype=np.float64)  # Kelvin
        self.K = np.array(self.var_dict.get("K"), dtype=np.float64)  # Kelvin
        self.xmax = np.array(self.var_dict.get("xmax"))
        self.gamma_s = np.array(self.var_dict.get("gamma_s"), dtype=np.float64)  # kj/kg/k
        self.gamma_o = np.array(self.var_dict.get("gamma_o"), dtype=np.float64)  # kj/kg/k
        self.m = np.array(self.var_dict.get("m"), dtype=np.float64)
        self.mu = np.array(self.var_dict.get("mu"), dtype=np.float64)
        self.T = np.array(self.var_dict.get("T"))
        self.ph = np.array(self.var_dict.get("ph"))
        self.kla = np.array(self.var_dict.get("kla"))
        self.clock = 0
        self.X = self.init[0]
        self.P = self.init[1]
        self.S = self.init[2]
        self.V = self.init[3]
        self.CO = self.init[4]
        self.previous = np.zeros(1)

        self.prod = np.zeros(5)
        self.cons = np.zeros(5)
        self.act_t = np.zeros(5)

        self.X_dummy = np.zeros(1)
        self.S_dummy = np.zeros(1)
        self.P_dummy = np.zeros(1)
        self.V_dummy = np.zeros(1)
        self.CO_dummy = np.zeros(1)
        self.out_dummy = np.zeros(self.eq_dim)

    def reset(self):
        self.u = np.array([0], dtype=np.float64)  # action non-normalized
        self.eq_dim = self.var_dict.get("eq_dim")
        self.init = np.array(self.var_dict.get("init"), dtype=np.float64)  # mol/L
        self.growth = np.array(self.var_dict.get("growth"), dtype=np.float64)  # L
        self.death = np.array(self.var_dict.get("death"), dtype=np.float64)  # Kelvin
        self.K = np.array(self.var_dict.get("K"), dtype=np.float64)  # Kelvin
        self.xmax = np.array(self.var_dict.get("xmax"))
        self.gamma_s = np.array(self.var_dict.get("gamma_s"), dtype=np.float64)  # kj/kg/k
        self.gamma_o = np.array(self.var_dict.get("gamma_o"), dtype=np.float64)  # kj/kg/k
        self.m = np.array(self.var_dict.get("m"), dtype=np.float64)
        self.mu = np.array(self.var_dict.get("mu"), dtype=np.float64)
        self.T = np.array(self.var_dict.get("T"))
        self.ph = np.array(self.var_dict.get("ph"))
        self.kla = np.array(self.var_dict.get("kla"))
        self.curr_product = np.zeros(1)
        self.prev_obs = np.zeros(self.eq_dim)
        self.X = self.init[0]
        self.P = self.init[1]
        self.S = self.init[2]
        self.V = self.init[3]
        self.CO = self.init[4]

        self.last_rew = np.zeros(1)
        self.clock = 0

    def step(self, var, t):
        # ODE X S P V CO
        # K:  K1_0 K2_1 KS_2 KO_3 KD_4 KH_5 KI_6 KP_7
        # gamma_s/o : gammax gammap
        self.X_dummy = var[0]
        self.P_dummy = var[1]
        self.S_dummy = var[2]
        self.V_dummy = var[3]
        self.CO_dummy = var[4]

        con1 = np.array([1]) + (self.K[0]/(np.power(np.array([10], dtype=np.float64), self.ph))) \
               + (np.power(np.array([10], dtype=np.float64), self.ph))/self.K[1]
        con2 = self.K[2]*self.X_dummy + self.S_dummy
        con3 = self.K[3]*self.X_dummy + self.CO_dummy
        beta = (self.mu*self.S_dummy)/(self.K[7] + self.S_dummy + np.power(self.S_dummy, np.array([2]))/self.K[6])
        kla = self.kla/np.power(self.V_dummy, np.array([0.4]))
        g_rate = (self.growth[0]*np.exp(self.growth[1]/self.T)*self.S_dummy*self.CO_dummy*(np.ones(1)-(self.X_dummy/self.xmax)))/(con1 * con2 * con3)

        d_rate = (self.death[0]*np.exp(self.death[1]/self.T)) * (np.ones([1]) - self.CO_dummy/(self.K[4] + self.CO_dummy))

        # Xd = [prod_term(+), cons(-)
        #X P S V CO
        self.prod[0] = g_rate * self.X_dummy
        self.prod[1] = beta*self.X_dummy
        self.prod[2] = 0
        self.prod[3] = 0
        self.prod[4] = kla * (0.037 - self.CO_dummy)

        self.cons[0] = d_rate*self.X_dummy
        self.cons[1] = self.K[5]*self.P_dummy
        self.cons[2] = self.out_dummy[0]/self.gamma_s[0] + self.out_dummy[1]/self.gamma_s[1]
        self.cons[3] = 0
        self.cons[4] = self.m[1]*self.X_dummy + self.out_dummy[0]/self.gamma_o[0] - self.out_dummy[1]/self.gamma_o[1]

        self.act_t[0] = self.X_dummy*self.u/self.V_dummy
        self.act_t[1] = self.P_dummy*self.u/self.V_dummy
        self.act_t[2] = self.X_dummy*self.u/self.V_dummy
        self.act_t[3] = self.u
        self.act_t[4] = self.CO_dummy*self.u/self.V_dummy

        self.out_dummy[0] = self.prod[0] - self.cons[0] - self.act_t[0] ##dx
        self.out_dummy[1] = self.prod[1] - self.cons[1] - self.act_t[1] ##dp
        self.out_dummy[2] = self.prod[2] - self.cons[2] - self.act_t[2] ##ds
        self.out_dummy[3] = self.prod[3] - self.cons[3] + self.act_t[3]
        self.out_dummy[4] = self.prod[4] - self.cons[4] - self.act_t[4]
        return self.out_dummy

    def set_action(self, action_n):
        self.u = action_n

    def get_obs(self,):
        return [self.P, self.V, self.X, self.u, self.CO]

if __name__ == '__main__':

    # ODE X P S V CO
    # K:  K1 K2 KS KO KD KH KI KP
    # gamma_s/o : gammax gammap
    # m ms mo

    var_dict = {"eq_dim": 5, "init": [0.05, 0, 10, 60, 0.037], "growth": [0.1224, -60],
                "death": [1.9 * 10 ** -3, -340],
                "K": [1 * 10 ** -10, 1.3 * 10 ** -4, 0.1828, 0.0352, 0.0368, 4 * 10 ** -4, 0.1, 2 * 10 ** -4],
                "xmax": [0.87],
                "gamma_s": [0.25, 0.68],
                "m": [0.0624, 4 * 10 ** -3], "kla": [1.27],
                "gamma_o": [43.5, 253.3], "mu": [0.05],
                "ph": [-7], "T": [303]
                }

    agent_dict = {"action_dim": 1, "obs_dim": 17, "obs_high": 1000 * np.ones(17),
                  "obs_low": -1000 * np.ones(17),
                  "act_high": [1], "act_low": [0],
                  "st": 1, "st_dt": 2, "steps": 100,
                  "dt_sp": 5
                  }

    #np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    env = CustomEnv(var_dict, agent_dict)
    len = agent_dict.get("steps")
    action_vec = np.zeros(len)
    state_vec = np.zeros(len)
    time_vec = np.linspace(0, 5, len)
    for i in range(agent_dict.get("steps")):
        #"action = env.action_space.sample()
        action = -1
        next_state, reward, done, info = env.step(action)
    env.ep_buffer.render()
    plt.pause(5000)
    plt.show()