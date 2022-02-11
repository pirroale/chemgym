import gym
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import *
from gym import spaces
import random
import pandas as pd

class CstrIsoT:
    def __init__(self, action_dim_, var_dict_):
        self.u = np.zeros(action_dim_) #action non-normalized
        self.action_dim = action_dim_
        self.var_dict = var_dict_
        self.c_dim = self.var_dict.get("C_dim")
        self.C = np.array(self.var_dict.get("C0"), dtype=np.float32)  # mol/L
        self.V = np.array(self.var_dict.get("V0"), dtype=np.float32)  # L
        self.cp = np.array([3.01], dtype=np.float32)  # kj/kg/k
        self.rho = np.array([0.934], dtype=np.float32)
        self.dhr = np.array([4.200, -11.000, -41.850])  # kj/KMOL
        self.T_inlet = np.array(self.var_dict.get("Tin"), dtype=np.float32)
        self.C_inlet = np.array(self.var_dict.get("Cin"), dtype=np.float32)
        self.v = np.array(self.var_dict.get("v"))
        self.k = np.array(self.var_dict.get("k0"), dtype=np.float32)
        self.target = 0
        self.last_rew = 0

        #Dummy variable

        self.C_dummy = np.zeros(self.c_dim)
        self.out_dummy = np.zeros(self.c_dim+2)
        #Counters
        self.clock = 0

    def reset(self):
        self.c_dim = self.var_dict.get("C_dim")
        self.C = np.array(self.var_dict.get("C0"), dtype=np.float32)  # mol/L
        self.V = np.array(self.var_dict.get("V0"), dtype=np.float32)  # L
        self.T_inlet = np.array(self.var_dict.get("Tin"), dtype=np.float32)
        self.C_inlet = np.array(self.var_dict.get("Cin"), dtype=np.float32)
        self.clock = 0
        self.last_rew = 0

    def step(self, var, t):
        C = np.zeros(self.c_dim)
        C[0:self.c_dim] = var[0:self.c_dim]
        r = np.zeros(self.c_dim)
        reaction_rate = np.array([self.k[0]*C[0], self.k[1]*C[1], self.k[2]*C[0]**2])  #1/min * kmol/m3
        v1, v2, v3 = self.v[0], self.v[1], self.v[2]
        r[0] = v1[0] * reaction_rate[0] + v3[0] * reaction_rate[2]
        r[1] = v1[1] * reaction_rate[0] + v2[1] * reaction_rate[1]
        dCdt = self.u[0] * (self.C_inlet - C) + r
        return [dCdt[0], dCdt[1]]

    def set_action(self, action_n):
        self.u = action_n

    def get_obs(self,):
        return [self.C[0], self.C[1]]

class isoCSTREnv(gym.Env):
    """Custom Environment that follows gym interface"""
    #metadata = {'render.modes': ['human']}
    # ><
    def __init__(self):
        super(isoCSTREnv, self, ).__init__()
        var_dict_ = {"C_dim": 2, "C0": [0.8, 0.5], "V0": [10], "T0": [407], "Tk0": [403], "Tin": [407], "Cin": [5.1, 0],
                    "v": [[-1, 1, 0, 0], [0, -1, 1, 0], [-1, 0, 0, 1]], "k0": [12.65, 12.65, 2]}
        agent_dict_ = {"action_dim": 1, "obs_dim": 2, "act_high": [100], "act_low": [5],
                      "obs_high": [10, 10],
                      "obs_low": [0, 0],
                      "st": 0.05, "st_dt": 10, "steps": 60,
                      "dt_sp": 5}
        #Header
        random.seed(22)
        acts_dim = agent_dict_.get("action_dim")
        obs_dim = agent_dict_.get("obs_dim")
        acts_high = agent_dict_.get("act_high")
        acts_low = agent_dict_.get("act_low")
        obs_high = agent_dict_.get("obs_high")
        obs_low = agent_dict_.get("obs_low")
        self.total_step = agent_dict_.get("steps")
        self.sampling_time = agent_dict_.get("st")
        self.sampling_diff = agent_dict_.get("st_dt")
        self.sp_vector = np.linspace(20, self.total_step, agent_dict_.get("dt_sp"))
        self.sp_stop = False
        self.RENDER = True
        self.SAVE = False
        self.sp_limit = [5, 10, 1]
        self.observation = np.zeros(obs_dim)
        self.cstr = CstrIsoT(acts_dim, var_dict_)
        self.cstr.reset()
        self.control_target()
        self.ep_buffer = EpisodeBuffer(self.cstr)
        self.ep_buffer.update_history()
        self.action_space = spaces.Box(low=-np.ones(acts_dim), high=np.ones(acts_dim), shape=(acts_dim,),
                                       dtype=np.float32)
        #self.observation_space = spaces.Box(low=np.array(obs_low), high=np.array(obs_high), shape=(obs_dim,),
        #                                  dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.ones(obs_dim), high=np.ones(obs_dim), shape=(obs_dim,),
                                            dtype=np.float32)

        diff_a = np.array([a_i - b_i for a_i, b_i in zip(acts_high, acts_low)])
        diff_obs = np.array([a_i - b_i for a_i, b_i in zip(obs_high, obs_low)])

        self.acts_mean = diff_a / 2
        self.observation_mean = diff_obs/2
        self.action_shift = acts_low + self.acts_mean
        self.observation_shift = obs_low + self.observation_mean
        self.previous_state = np.copy(self.observation)
        self.input_dummy = np.zeros(self.cstr.c_dim)
        self.previous_action = np.zeros(2)

    def step(self, action_norm):
        self.ep_buffer.update_history()
        self.control_target()
        self.random_disturb()
        if self.cstr.clock>1:
            self.previous_state = np.copy(self.observation)
            self.previous_action = np.copy(self.cstr.u)
        #self.random_disturb()
        action = action_norm * self.acts_mean + self.action_shift
        self.cstr.set_action(action)

        #Solve ODE system
        self.input_dummy[0:self.cstr.c_dim] = self.cstr.C[0:self.cstr.c_dim]

        result = odeint(self.cstr.step, self.input_dummy, np.linspace(0, self.sampling_time, self.sampling_diff))

        self.cstr.C[0:self.cstr.c_dim] = result[self.sampling_diff-1][0:self.cstr.c_dim]
        #Update state
        self.observation = self.cstr.get_obs()
        reward, done = self.reward_function()
        self.cstr.last_rew += reward
        self.cstr.clock += 1
        self.observation = self.normalize_obs(self.observation)
        return np.array(self.observation).astype(np.float32), reward, done, {"ok": True}

    def random_disturb(self):
        r = random.randrange(0, 100, 1)
        if r < 5:
            self.cstr.C_inlet[0] = random.uniform(0.5, 1.00)

    def linear_disturb(self):
        if self.cstr.clock < 13:
            self.cstr.C_inlet[0] = random.randrange(8, 12, 1)
        elif 13 < self.cstr.clock < 27:
            self.cstr.C_inlet[0] = random.randrange(8, 12, 1)
        elif 27 < self.cstr.clock < 50:
            self.cstr.C_inlet[0] = random.randrange(8, 12, 1)
        elif self.cstr.clock > 50:
            self.cstr.C_inlet[0] = random.randrange(8, 12, 1)

    def control_target(self):
        if self.cstr.clock < self.sp_vector[0] and not self.sp_stop:
            self.cstr.target = random.randrange(self.sp_limit[0], self.sp_limit[1], self.sp_limit[2]) / 10
            self.sp_stop = True
        elif self.sp_vector[0] < self.cstr.clock < self.sp_vector[1] and self.sp_stop:
            self.cstr.target = random.randrange(self.sp_limit[0], self.sp_limit[1], self.sp_limit[2]) / 10
            self.sp_stop = False
        elif self.sp_vector[1] < self.cstr.clock < self.sp_vector[2] and not self.sp_stop:
            self.cstr.target = random.randrange(self.sp_limit[0], self.sp_limit[1], self.sp_limit[2]) / 10
            self.sp_stop = True
        elif self.sp_vector[2] < self.cstr.clock < self.sp_vector[3] and self.sp_stop:
            self.cstr.target = random.randrange(self.sp_limit[0], self.sp_limit[1], self.sp_limit[2]) / 10
            self.sp_stop = False
        elif self.sp_vector[3] < self.cstr.clock < self.sp_vector[4] and not self.sp_stop:
            self.cstr.target = random.randrange(self.sp_limit[0], self.sp_limit[1], self.sp_limit[2]) / 10
            self.sp_stop = True
        #     self.sp_stop = True
        # elif 50 < self.cstr.clock < 60 and self.sp_stop:
        #     self.cstr.target = random.randrange(self.sp_limit[0], self.sp_limit[1], self.sp_limit[2]) / 10
        #     self.sp_stop = False
        # elif 60 < self.cstr.clock < 70 and not self.sp_stop:
        #     self.cstr.target = random.randrange(self.sp_limit[0], self.sp_limit[1], self.sp_limit[2]) / 10
        #     self.sp_stop = True
        # elif 70 < self.cstr.clock < 80 and self.sp_stop:
        #     self.cstr.target = random.randrange(self.sp_limit[0], self.sp_limit[1], self.sp_limit[2]) / 10
        #     self.sp_stop = False
        # elif 80 < self.cstr.clock < 100 and not self.sp_stop:
        #     self.cstr.target = random.randrange(self.sp_limit[0], self.sp_limit[1], self.sp_limit[2]) / 10
        #     self.sp_stop = True

    def control_target_1(self):
        self.cstr.target = 0.6

    def normalize_obs(self, obs):
        obs_n = (obs-self.observation_shift)/self.observation_mean
        return obs_n

    def denormalize_obs(self, obs_n):
        obs = (obs_n * self.observation_mean) + self.observation_shift
        return obs

    def reward_function(self):
        reward_ = 0
        done_ = False

        diff =abs(self.cstr.u - self.previous_action)
        if diff[0] > 5:
            r1 = -10
        else:
            r1 = 0
        if abs(self.cstr.C[1]-self.cstr.target) > 0.01:
            r2 = -abs(self.cstr.C[1] - self.cstr.target)
        else:
            r2 = 1
        reward_ = r1 + r2
        if self.cstr.clock > self.total_step:
            done_ = True
        return reward_, done_

    def render(self, mode='human'):
        #time_vec = np.linspace(0, 5, len(self.ep_buffer.C_history))
        plt.subplot(3, 1, 1)
        plt.plot(self.ep_buffer.CB_history)
        plt.plot(self.ep_buffer.target_history)
        # plt.plot(self.ep_buffer.C0_history)
        plt.title("Concentration")

        plt.subplot(3,1, 2)
        plt.plot(self.ep_buffer.action1_history)
        plt.title("F")

        #
        plt.subplot(3, 1, 3)
        plt.plot(self.ep_buffer.reward_history)
        plt.title("Cumulated reward")

        plt.draw()
        plt.pause(0.0000000000001)
        plt.clf()

    def close(self):
        pass

    def save_episode(self):
        df = pd.DataFrame([np.array(self.ep_buffer.CA_history), np.array(self.ep_buffer.CB_history),
                           np.array(self.ep_buffer.action1_history), np.array(self.ep_buffer.action2_history),
                           np.array(self.ep_buffer.T_history), np.array(self.ep_buffer.Tk_history)])
        with pd.ExcelWriter("buffer.xlsx", engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df.to_excel(writer, 'AI', index=False)

    def reset(self):
        if self.RENDER:
            self.render()
        self.cstr.reset()
        self.observation = self.cstr.get_obs()
        self.observation = self.normalize_obs(self.observation)
        if self.SAVE:
            self.save_episode()
        self.ep_buffer.reset_history()
        self.sp_stop = False
        return np.array(self.observation) # reward, done, info can't be included


class EpisodeBuffer:
    def __init__(self, obj_):
        self.target_history = list()
        self.CA_history = list()
        self.CB_history = list()
        self.C0_history = list()
        self.action1_history = list()
        self.reward_history = list()
        self.obj = obj_

    def update_history(self):
        self.target_history.append(self.obj.target)
        self.CB_history.append(self.obj.C[1])
        self.CA_history.append(self.obj.C[0])
        self.C0_history.append(self.obj.C_inlet[0])
        self.action1_history.append(self.obj.u[0])
        self.reward_history.append(self.obj.last_rew)

    def reset_history(self):
        self.target_history.clear()
        self.CB_history.clear()
        self.CA_history.clear()
        self.C0_history.clear()
        self.action1_history.clear()
        self.reward_history.clear()

