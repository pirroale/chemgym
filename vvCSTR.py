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
        self.T = np.array(self.var_dict.get("T0"), dtype=np.float32)[0]  # Kelvin
        self.Tk = np.array(self.var_dict.get("Tk0"), dtype=np.float32)[0]  # Kelvin
        self.cp = np.array([3.01], dtype=np.float32)  # kj/kg/k
        self.rho = np.array([0.934], dtype=np.float32)
        self.T_inlet = np.array(self.var_dict.get("Tin"), dtype=np.float32)
        self.C_inlet = np.array(self.var_dict.get("Cin"), dtype=np.float32)
        self.Ea = np.array(self.var_dict.get("Ea"), dtype=np.float32)
        self.v = np.array(self.var_dict.get("v"))
        self.k0 = np.array(self.var_dict.get("k0"), dtype=np.float32)
        self.target = 0
        self.k_dummy = np.array([0, 0, 0], dtype=np.float32)
        self.last_rew = 0

        #Dummy variable

        self.C_dummy = np.zeros(self.c_dim)
        self.out_dummy = np.zeros(self.c_dim+2)
        #Counters
        self.clock = 0
        self.ki = 400
        self.kp = 0.17
        self.ko = 60

    def reset(self):
        self.c_dim = self.var_dict.get("C_dim")
        self.C = np.array(self.var_dict.get("C0"), dtype=np.float32)  # mol/L
        self.V = np.array(self.var_dict.get("V0"), dtype=np.float32)  # L
        self.T = np.array(self.var_dict.get("T0"), dtype=np.float32)[0]  # Kelvin
        self.Tk = np.array(self.var_dict.get("Tk0"), dtype=np.float32)[0]  # Kelvin
        self.T_inlet = np.array(self.var_dict.get("Tin"), dtype=np.float32)
        self.C_inlet = np.array(self.var_dict.get("Cin"), dtype=np.float32)
        self.clock = 0
        self.last_rew = 0

    def step(self, var, t):
        C = np.zeros(self.c_dim)
        C[0:self.c_dim] = var[0:self.c_dim]
        T = np.array([var[self.c_dim]])
        Tk = np.array([var[self.c_dim+1]])
        self.k = self.k0*np.exp(self.Ea/T)
        r = np.zeros(self.c_dim)
        reaction_rate = np.array([self.k[0]*C[0], self.k[1]*C[1], self.k[2]*C[0]**2])  #1/min * kmol/m3
        v1, v2, v3 = self.v[0], self.v[1], self.v[2]
        r[0] = v1[0] * reaction_rate[0] + v3[0] * reaction_rate[2]
        r[1] = v1[1] * reaction_rate[0] + v2[1] * reaction_rate[1]
        Tdiff = T-Tk
        dCdt = self.u[0] * (self.C_inlet - C) + r
        t1 = self.u[0] * (self.T_inlet - T)
        t2 = - sum(self.dhr*reaction_rate)/(self.rho*self.cp)
        t3 = + np.array([4032*0.215])*-Tdiff/np.array([10*self.rho*self.cp])
        dTdt = t1 + t2 + t3
        dTkdt = (self.u[1] + 0.215*4032*Tdiff)/10

        return [dCdt[0], dCdt[1], dTdt, dTkdt]

    def step_pi(self, var, t):
        C = np.zeros(self.c_dim)
        C[0:self.c_dim] = var[0:self.c_dim]
        T = np.array([var[self.c_dim]])
        Tk = np.array([var[self.c_dim + 1]])
        epsilon = (self.target - self.C_dummy[1])
        error = var[self.c_dim + 2]
        k = self.k0 * np.exp(self.Ea / T)
        r = np.zeros(self.c_dim)
        reaction_rate = np.array([k[0] * C[0], k[1] * C[1], k[2] * C[0] ** 2])  # 1/min * kmol/m3
        v1, v2, v3 = self.v[0], self.v[1], self.v[2]
        r[0] = v1[0] * reaction_rate[0] + v2[0] * reaction_rate[1] + v3[0] * reaction_rate[2]
        r[1] = v1[1] * reaction_rate[0] + v2[1] * reaction_rate[1] + v3[1] * reaction_rate[2]
        u = self.ko + self.kp * epsilon + (self.ki/self.kp)*error
        self.set_action(u)

        Tdiff = T - Tk
        dCdt = self.u[0] * (self.C_inlet - C) + r
        t1 = self.u[0] * (self.T_inlet - T)
        t2 = - sum(self.dhr * reaction_rate) / (self.rho * self.cp)
        t3 = + np.array([4032 * 0.215]) * -Tdiff / np.array([10 * self.rho * self.cp])
        dTdt = t1 + t2 + t3
        dTkdt = (self.u[1] + 0.215 * 4032 * Tdiff) / 10
        dedt = epsilon

        out = [dCdt[0], dCdt[1], dTdt, dTkdt, dedt]
        return out

    def set_action(self, action_n):
        self.u = action_n

    def get_obs(self,):
        return [self.C[0], self.C[1], self.T, self.Tk]

class vvCSTREnv(gym.Env):
    """Custom Environment that follows gym interface"""
    #metadata = {'render.modes': ['human']}
    # ><
    def __init__(self):
        super(vvCSTREnv, self, ).__init__()
        var_dict_ = {"C_dim": 2, "C0": [0.8, 0.5], "V0": [10], "T0": [407], "Tk0": [403], "Tin": [407], "Cin": [5.1, 0],
                    "v": [[-1, 1, 0, 0], [0, -1, 1, 0], [-1, 0, 0, 1]], "k0": [1.287e12, 1.287e12, 9.043e9],
                    "Ea": [-9758, -9758, -8560]}
        agent_dict_ = {"action_dim": 2, "obs_dim": 4, "act_high": [20, 0], "act_low": [5, -2000],
                      "obs_high": [10, 10, 700, 700],
                      "obs_low": [0, 0, 273, 273],
                      "st": 0.005, "st_dt": 100, "steps": 60,
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
        self.RENDER = False
        self.SAVE = False
        self.sp_limit = [9, 13, 1]
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
        self.sp_stop = False
        self.previous_state = np.copy(self.observation)
        self.input_dummy = np.zeros(self.cstr.c_dim+2)
        self.previous_action = np.zeros(2)

    def step(self, action_norm):
        self.ep_buffer.update_history()
        self.control_target()
        if self.cstr.clock>1:
            self.previous_state = np.copy(self.observation)
            self.previous_action = np.copy(self.cstr.u)
        #self.random_disturb()
        action = action_norm * self.acts_mean + self.action_shift
        self.cstr.set_action(action)

        #Solve ODE system
        self.input_dummy[0:self.cstr.c_dim] = self.cstr.C[0:self.cstr.c_dim]
        self.input_dummy[self.cstr.c_dim] = self.cstr.T
        self.input_dummy[self.cstr.c_dim + 1] = self.cstr.Tk
        #input_ = np.concatenate((self.cstr.C, self.cstr.T, self.cstr.V), axis=0)

        result = odeint(self.cstr.step, self.input_dummy, np.linspace(0, self.sampling_time, self.sampling_diff))

        self.cstr.C[0:self.cstr.c_dim] = result[self.sampling_diff-1][0:self.cstr.c_dim]
        self.cstr.T = result[self.sampling_diff-1][self.cstr.c_dim]
        self.cstr.Tk = result[self.sampling_diff-1][self.cstr.c_dim+1]
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
            self.cstr.C_inlet[0] = random.uniform(9.00, 11.00)

    def linear_disturb(self):
        if self.cstr.clock < 13:
            self.cstr.C_inlet[0] = random.randrange(8, 12, 1)
        elif 13 < self.cstr.clock < 27:
            self.cstr.C_inlet[0] = random.randrange(8, 12, 1)
        elif 27 < self.cstr.clock < 50:
            self.cstr.C_inlet[0] = random.randrange(8, 12, 1)
        elif self.cstr.clock > 50:
            self.cstr.C_inlet[0] = random.randrange(8, 12, 1)

    def control_target_1(self):
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

    def control_target(self):
        self.cstr.target = 0.6

    def control_target_1(self):
        if self.cstr.clock < 10:
            self.cstr.target = 1.2
        elif 10 < self.cstr.clock < 20:
            self.cstr.target = random.randrange(7, 12, 1) / 10
        elif 20 < self.cstr.clock < 30:
            self.cstr.target = random.randrange(7, 12, 1) / 10
        elif 30 < self.cstr.clock < 40:
            self.cstr.target = random.randrange(7, 12, 1)/10
        elif 40 < self.cstr.clock < 50:
            self.cstr.target = random.randrange(7, 12, 1)/10
        elif 50 < self.cstr.clock < 60:
            self.cstr.target = random.randrange(7, 12, 1)/10
        elif 60 < self.cstr.clock < 70:
            self.cstr.target = random.randrange(7, 12, 1)/10
        elif 70 < self.cstr.clock < 80:
            self.cstr.target = random.randrange(7, 12, 1)/10
        elif 80 < self.cstr.clock < 100:
            self.cstr.target = random.randrange(7, 12, 1) / 10

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
        if diff[0] > 5 and diff[1] > 50:
            r1 = -50
        else:
            r1 = 0
        if abs(self.cstr.C[1]-self.cstr.target) > 0.01:
            r2 = -20*abs(self.cstr.C[1] - self.cstr.target)
        else:
            r2 = 1
        reward_ = r1 + r2
        if self.cstr.clock > self.total_step:
            done_ = True
        return reward_, done_

    def render(self, mode='human'):
        #time_vec = np.linspace(0, 5, len(self.ep_buffer.C_history))
        plt.subplot(2, 3, 1)
        plt.plot(self.ep_buffer.CA_history)
        plt.plot(self.ep_buffer.CB_history)
        plt.plot(self.ep_buffer.target_history)
        # plt.plot(self.ep_buffer.C0_history)
        plt.title("Concentration")

        plt.subplot(2, 3, 2)
        plt.plot(self.ep_buffer.action1_history)
        plt.title("F")

        plt.subplot(2, 3, 3)
        plt.plot(self.ep_buffer.action2_history)
        plt.title("Q")

        plt.subplot(2, 3, 4)
        plt.plot(self.ep_buffer.T_history)
        plt.plot(self.ep_buffer.Tk_history)
        plt.title("Temperature")
        #
        plt.subplot(2, 3, 5)
        plt.plot(self.ep_buffer.reward_history)
        plt.title("Cumulated reward")

        plt.draw()
        plt.pause(0.0000000000001)
        plt.clf()

    def render_X(self):
        # time_vec = np.linspace(0, 5, len(self.ep_buffer.C_history))
        plt.subplot(2, 2, 1)
        plt.plot(self.ep_buffer.CA_history)
        plt.plot(self.ep_buffer.CB_history)
        plt.plot(self.ep_buffer.target_history)
        # plt.plot(self.ep_buffer.C0_history)
        plt.title("Concentration")

        plt.subplot(2, 2, 2)
        plt.plot(self.ep_buffer.action1_history)
        plt.title("F")

        plt.subplot(2, 2, 3)
        plt.plot(self.ep_buffer.action2_history)
        plt.title("Q")

        plt.subplot(2, 2, 4)
        plt.plot(self.ep_buffer.T_history)
        plt.plot(self.ep_buffer.Tk_history)
        plt.title("Temperature")

        plt.draw()
        #plt.pause(2000)

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

    def control_pi(self, target, disturb):
        input_dummy_1 = np.zeros(self.cstr.c_dim+3)
        error = 0
        input_dummy_1[-1] = error
        for j in range(60):
            self.cstr.target, self.cstr.C_inlet[0] = target[j], disturb[j]
            self.ep_buffer.update_history()
            input_dummy_1[0:self.cstr.c_dim] = self.cstr.C[0:self.cstr.c_dim]
            input_dummy_1[self.cstr.c_dim] = self.cstr.T
            input_dummy_1[self.cstr.c_dim + 1] = self.cstr.Tk
            input_dummy_1[self.cstr.c_dim + 2] = error
            result = odeint(self.cstr.step_pi, input_dummy_1, np.linspace(0, self.sampling_time, self.sampling_diff))
            for i in range(self.cstr.c_dim):
                self.cstr.C[i] = result[self.sampling_diff - 1][i]
            self.cstr.T = result[self.sampling_diff - 1][self.cstr.c_dim]
            self.cstr.Tk = result[self.sampling_diff - 1][self.cstr.c_dim + 1]
            error = result[self.sampling_diff - 1][self.cstr.c_dim + 2]


class EpisodeBuffer:
    def __init__(self, obj_):
        self.T_history = list()
        self.Tk_history = list()
        self.target_history = list()
        self.CA_history = list()
        self.CB_history = list()
        self.C0_history = list()
        self.action1_history = list()
        self.action2_history = list()
        self.reward_history = list()
        self.obj = obj_

    def update_history(self):
        self.T_history.append(self.obj.T)
        self.Tk_history.append(self.obj.Tk)
        self.target_history.append(self.obj.target)
        self.CB_history.append(self.obj.C[1])
        self.CA_history.append(self.obj.C[0])
        self.C0_history.append(self.obj.C_inlet[0])
        self.action1_history.append(self.obj.u[0])
        self.action2_history.append(self.obj.u[1])
        self.reward_history.append(self.obj.last_rew)

    def reset_history(self):
        self.T_history.clear()
        self.Tk_history.clear()
        self.target_history.clear()
        self.CB_history.clear()
        self.CA_history.clear()
        self.C0_history.clear()
        self.action1_history.clear()
        self.action2_history.clear()
        self.reward_history.clear()


