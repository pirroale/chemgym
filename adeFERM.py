import gym
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import *
from gym import spaces
import random
import json
import math


class adeFERMEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    #metadata = {'render.modes': ['human']}
    # ><

    def __init__(self):
        super(adeFERMEnv, self, ).__init__()
        agent_dict_ = {"action_dim": 1, "obs_dim": 41, "act_high":[0.01000], "act_low": [0.0],
              "obs_high": 200*np.ones(41), "obs_low": -200*np.ones(41),
              "st_dt": 1000, "steps": 10,
              "dt": 3.5}
        acts_dim = agent_dict_.get("action_dim")
        obs_dim = agent_dict_.get("obs_dim")
        acts_high = agent_dict_.get("act_high")
        acts_low = agent_dict_.get("act_low")
        obs_high = agent_dict_.get("obs_high")
        obs_low = agent_dict_.get("obs_low")
        self.total_step = agent_dict_.get("steps")
        self.sampling_diff = agent_dict_.get("st_dt")
        self.time_step = agent_dict_.get("dt")
        self.action_space = spaces.Box(low=-np.ones(acts_dim), high=np.ones(acts_dim), shape=(acts_dim,),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.ones(obs_dim), high=np.ones(obs_dim), shape=(obs_dim,),
                                            dtype=np.float32)
        diff_a = np.array([a_i - b_i for a_i, b_i in zip(acts_high, acts_low)])
        diff_obs = np.array([a_i - b_i for a_i, b_i in zip(obs_high, obs_low)])

        self.acts_mean = diff_a / 2
        self.observation_mean = diff_obs / 2
        self.action_shift = acts_low + self.acts_mean
        self.observation_shift = obs_low + self.observation_mean
        self.obj = OBJ()
        self.initial_state = self.obj.init
        self.current_state = np.copy(self.initial_state)
        self.previous_state = np.copy(self.initial_state)
        self.clock = 0
        self.last_rew = 0
        self.current_time = 0
        self.buffer = EpisodeBuffer()

    def reset(self):
        self.render()
        self.current_state = np.copy(self.initial_state)
        observation = self.normalize_obs(self.get_observation())
        self.buffer.reset()
        self.obj.action = 0
        self.obj.pH = list()
        self.clock = 0
        self.current_time = 0
        return np.array(observation).astype(np.float32)

    def step(self, action_norm):
        action = action_norm * self.acts_mean + self.action_shift
        self.obj.action = action
        self.previous_state = np.copy(self.current_state)
        #result = solve_ivp(self.obj.ade_SOB,(0, self.time_step), self.current_state, method='BDF')
        result = odeint(self.obj.ade_SRB, self.current_state, np.linspace(self.current_time, self.current_time+self.time_step, self.sampling_diff, dtype=np.float64))
        self.current_state = result[-1, 0:len(self.initial_state)]
        reward, done = self.reward_function()
        self.last_rew = reward
        self.clock += 1
        self.current_time += self.time_step
        observation = self.normalize_obs(self.get_observation())
        self.buffer.update([self.current_state[6], self.current_state[19], self.current_state[27], self.current_state[28]], self.obj, self.last_rew)
        return np.array(observation).astype(np.float32), reward, done, {"ok": True}

    def render(self, mode='human'):
        ax, bx = 3, 2
        if len(self.buffer.action) > 0:
            plt.subplot(ax, bx, 1)
            plt.plot(self.buffer.buffer_x1)
            plt.title("Acido Acetivo x[6]")

            plt.subplot(ax, bx, 2)
            plt.plot(self.buffer.buffer_x2)
            plt.title("Liquid Methane x[19]")

            plt.subplot(ax, bx, 3)
            plt.plot(self.buffer.buffer_x3)
            plt.title("Gas methane x[27]")

            plt.subplot(ax, bx, 4)
            plt.plot(self.buffer.action)
            plt.ylim([0, 0.02])
            plt.title("Dil")

            plt.subplot(ax, bx, 5)
            plt.plot(self.buffer.buffer_rew)
            plt.title("not cumulated reward")

            plt.draw()
            plt.pause(0.0000001)
            plt.clf()

    def close(self):
        pass

    def normalize_obs(self, obs):
        obs_n = (obs-self.observation_shift)/self.observation_mean
        return obs_n

    def denormalize_obs(self, obs_n):
        obs = (obs_n * self.observation_mean) + self.observation_shift
        return obs

    def get_observation(self):
        return np.copy(self.current_state)

    def reward_function(self):
        done = False
        r = np.zeros(4)
        r[0] = self.current_state[27]/10**0.2
        r[1] = -abs(self.current_state[28])**0.2
        r[2] = (self.current_state[6])**0.2
        reward = np.sum(r)
        #reward = -self.current_state[27]*self.obj.V_gas/100
        if self.clock > self.total_step:
            done = True
        return reward, done


class EpisodeBuffer:
    def __init__(self):
        self.buffer_x1 = list()
        self.buffer_x2 = list()
        self.buffer_x3 = list()
        self.buffer_x4 = list()
        self.buffer_rew = list()
        self.action = list()

    def update(self, data, obj, rew):
        self.buffer_x1.append(data[0])
        self.buffer_x2.append(data[1])
        self.buffer_x3.append(data[2])
        self.buffer_x4.append(data[3])
        self.buffer_rew.append(rew)
        self.action.append(obj.action)

    def reset(self):
        self.buffer_x1.clear()
        self.buffer_x2.clear()
        self.buffer_x3.clear()
        self.buffer_x4.clear()
        self.buffer_rew.clear()
        self.action.clear()
        
class OBJ():
    def __init__(self):

        self.action = np.zeros(1, dtype=np.float64)
        self.ph =list()
        # General Parameters
        self.f_sI_xc = 0.1
        self.f_xI_xc = 0.2
        self.f_ch_xc = 0.2
        self.f_pr_xc = 0.2
        self.f_li_xc = 0.3
        self.N_xc = 0.002685714
        self.N_I = 0.004285714
        self.N_aa = 0.007
        self.C_xc = 0.02786
        self.C_sI = 0.03
        self.C_ch = 0.0313
        self.C_pr = 0.03
        self.C_li = 0.022
        self.C_xI = 0.03
        self.C_su = 0.0313
        self.C_aa = 0.03
        self.f_fa_li = 0.95
        self.C_fa = 0.0217
        self.f_h2_su = 0.19
        self.f_bu_su = 0.13
        self.f_pro_su = 0.27
        self.f_ac_su = 0.41
        self.N_bac = 0.005714286
        self.C_bu = 0.025
        self.C_pro = 0.0268
        self.C_ac = 0.0313
        self.C_bac = 0.0313
        self.Y_su = 0.1
        self.f_h2_aa = 0.06
        self.f_va_aa = 0.23
        self.f_bu_aa = 0.26
        self.f_pro_aa = 0.05
        self.f_ac_aa = 0.4
        self.C_va = 0.024
        self.Y_aa = 0.08
        self.Y_fa = 0.06
        self.Y_c4 = 0.06
        self.Y_pro = 0.04
        self.C_ch4 = 0.0156
        self.Y_ac = 0.05
        self.Y_h2 = 0.06
        self.f_ac_li = 0.7
        self.f_h2_li = 0.3
        self.f_pro_va = 0.54
        self.f_ac_va = 0.31
        self.f_h2_va = 0.15
        self.f_ac_bu = 0.8
        self.f_h2_bu = 0.2
        self.f_ac_pro = 0.57
        self.f_h2_pro = 0.43

        self.k_dis = 0.5
        self.k_hyd_ch = 1.25
        self.k_hyd_pr = 0.525
        self.k_hyd_li = 0.8
        self.K_S_IN = 0.0001
        self.km_su = 20
        self.ks_su = 0.3
        self.pH_UL_aa = 5.5
        self.pH_LL_aa = 4
        self.km_aa = 45
        self.ks_aa = 0.3
        self.km_fa = 6
        self.ks_fa = 0.8
        self.K_Ih2_fa = 0.000005
        self.km_c4 = 20
        self.ks_c4 = 0.2
        self.K_Ih2_c4 = 0.00001
        self.km_pro = 15
        self.ks_pro = 0.2
        self.K_Ih2_pro = 0.000035
        self.km_ac = 8
        self.ks_ac = 0.20
        self.K_I_nh3 = 0.003
        self.pH_UL_ac = 7
        self.pH_LL_ac = 6
        self.km_h2 = 55
        self.ks_h2 = 0.0000099
        self.pH_UL_h2 = 6
        self.pH_LL_h2 = 5
        self.k_dec_Xsu = 0.05
        self.k_dec_Xaa = 0.02
        self.k_dec_Xfa = 0.06
        self.k_dec_Xc4 = 0.02
        self.k_dec_Xpro = 0.03
        self.k_dec_Xac = 0.05
        self.k_dec_Xh2 = 0.09

        self.K_a_va = 1.38 * 10 ** (-5)
        self.K_a_bu = 1.51 * 10 ** (-5)
        self.K_a_pro = 1.32 * 10 ** (-5)
        self.K_a_ac = 1.74 * 10 ** (-5)
        self.k_A_Bva = 1e10
        self.k_A_Bbu = 1e10
        self.k_A_Bpro = 1e10
        self.k_A_Bac = 1e10
        self.k_A_B_co2 = 1e10
        self.k_A_BIN = 1e10
        self.p_atm = 1.01325
        self.k_p = 50000

        # Sulphur based parameters for SO4(2-) reduction
        self.X_acSRB_in = 0.01
        self.X_actSRB_in = 0.01
        self.X_h2tSRB_in = 0.01
        self.p_gas_h2s_in = 0
        self.S_h2s_in = 0
        self.S_co2_in = 0
        self.Y_ac = 0.035
        self.Y_act = 0.041
        self.Y_h2t = 0.077
        self.Y_fb = 0.043
        self.Y_ab = 0.018
        self.Y_actMB = 0.026
        self.Y_h2tMB = 0.018
        self.m6 = 32
        self.m2 = 112
        self.m3 = 64
        self.m4 = 16
        self.m1 = 342
        self.m5 = 96
        self.K_a1_h2s = 10 ** (-7)
        self.K_a2_h2s = 10 ** (-14)
        self.K_a1_co2 = 4.71 * 10 ** (-7)
        self.K_a2_co2 = 5.13 * 10 ** (-11)
        self.f_so42_ac = 9 / 14
        self.f_so42_act = 3 / 2
        self.f_so42_h2t = 3 / 2
        self.f_co2_ac = 1 / self.m2
        self.f_co2_act = 2 / self.m3
        self.f_co2_h2t = 1 / (2 * self.m4)
        self.f_co2_fb = 4 / self.m1
        self._co2_ab = 1 / self.m2
        self.f_co2_acMB = 1 / self.m3
        self.f_co2_h2tMB = 1 / (4 * self.m4)
        self.f_h2s_ac = 3 / 14
        self.f_h2s_act = 1 / 2
        self.f_h2s_h2t = 1 / 2
        self.f_ac_ac = 4 / 7
        self.k = 1.222
        self.km_acSRB = 0.81 / (self.Y_ac * self.k)
        self.km_actSRB = 0.51 / (self.Y_act * self.k)
        self.km_h2tSRB = 5 / (self.Y_h2t * self.k)
        self.km_fb = 8 / (self.Y_fb * self.k)
        self.km_ab = 0.16 / (self.Y_ab * self.k)
        self.km_actMB = 0.24 / (self.Y_actMB * self.k)
        self.km_h2tMB = 1 / (self.Y_h2tMB * self.k)
        self.KI_so42_ac = 0.285
        self.KI_so42_act = 0.285
        self.KI_so42_h2t = 0.55
        self.KI_FB = 0.55
        self.KI_AB = 0.215
        self.KI_act = 0.285
        self.KI_h2t = 0.215
        self.Ks_so42_ac = 0.295
        self.Ks_so42_act = 0.024
        self.Ks_so42_h2t = 0.00005
        self.K1 = 0.0074
        self.K2 = 0.0192
        self.K3 = 0.0009
        self.k_dec_acSRB = 0.018
        self.k_dec_actSRB = 0.025
        self.k_dec_h2tSRB = 0.03
        self.Vs_h2s = 0.777
        self.Vs_h2 = 1.554
        self.Vs_ch4 = 0.388
        self.Vs_co2 = 24.862
        self.H_h2s = 0.343 / 10000

        # Oxydation reation parameters and coefficients for H2S degradation
        self.km_o2 = 107.4  # d-1
        self.ko2 = 0.0013  # g/m3
        self.alpha = 1  # --
        self.K_I_o2 = 1.9e-7  # kg/m3

        self.W_ch_in = 47.3  # (1) kgCOD/m3
        self.W_pr_in = 21.9  # (2) kgCOD/m3
        self.W_li_in = 30.8  # (3) kgCOD/m3
        self.W_I_in = 25.4  # (4) kgCOD/m3
        self.W_xc_in = 115.4  # (1) + (2) + (3) + (4) kgCOD/m3

        self.S_I_in = 16.5  # kgCOD/m3
        self.S_IN_in = 0.012  # kgCOD/m3

        self.S_su_in = 0.9  # kgCOD/m3
        self.S_aa_in = 0.0011  # kgCOD/m3
        self.S_fa_in = 0.001  # kgCOD/m3
        self.S_va_in = 4.4  # kgCOD/m3
        self.S_bu_in = 3.6  # kgCOD/m3
        self.S_pro_in = 2.3  # kgCOD/m3
        self.S_ac_in = 4.1  # kgCOD/m3

        self.S_gas_o2_in = 1e-10  # kg/m3
        self.S_o2_in = 1e-10  # kg/m3
        self.S_so42_in = 0  # kgS/m3

        self.X_su_in = 0.01  # kgCOD/m3
        self.X_aa_in = 0.01  # kgCOD/m3
        self.X_fa_in = 0.01  # kgCOD/m3
        self.X_c4_in = 0.01  # kgCOD/m3
        self.X_pro_in = 0.01  # kgCOD/m3
        self.X_ac_in = 0.01  # kgCOD/m3
        self.X_h2_in = 0.01  # kgCOD/m3

        self.S_h2_in = 0  # kgCOD/m3
        self.S_ch4_in = 0  # kgCOD/m3
        self.S_va_dis_in = 0  # kgCOD/m3
        self.S_bu_dis_in = 0  # kgCOD/m3
        self.S_pro_dis_in = 0  # kgCOD/m3
        self.S_ac_dis_in = 0  # kgCOD/m3
        self.S_hco3_in = 0  # kgCOD/m3
        self.S_IC_in = 0  # kgCOD/m3
        self.S_gas_h2_in = 0  # kgCOD/m3
        self.p_gas_h2_in = 0  # kgCOD/m3
        self.S_gas_co2_in = 0  # kgCOD/m3
        self.p_gas_co2_in = 0  # kgCOD/m3
        self.S_gas_ch4_in = 0  # kgCOD/m3
        self.p_gas_ch4_in = 0  # kgCOD/m3
        self.S_Hplus_in = 0  # kgCOD/m3
        self.S_nh3_in = 0  # kgCOD/m3
        self.S_nh4_in = 0  # kgCOD/m3
        self.S_cat_in = 0  # kgCOD/m3
        self.S_an_in = 0  # kgCOD/m3

        # Pre-evaluation parameters
        self.R_gas = 0.083145
        self.T_base = 298.15
        self.T_op = 308.15
        self.K_w = math.exp(55900 / (self.R_gas * 100) * (1 / self.T_base - 1 / self.T_op)) * 10 ** (-14)
        self.K_a_co2 = 10 ** (-6.35) * math.exp(7646 / (self.R_gas * 100) * (1 / self.T_base - 1 / self.T_op))
        self.K_a_IN = 10 ** (-9.25) * math.exp(51965 / (self.R_gas * 100) * (1 / self.T_base - 1 / self.T_op))
        self.p_gas_h2o = 0.0313 * np.exp(5290 * (1 / self.T_base - 1 / self.T_op))
        self.K_H_co2 = 0.035 * math.exp(-19140 / (self.R_gas * 100) * (1 / self.T_base - 1 / self.T_op))
        self.K_H_ch4 = 0.0014 * math.exp(-14240 / (self.R_gas * 100) * (1 / self.T_base - 1 / self.T_op))
        self.K_H_h2 = 0.00078 * math.exp(-4180 / (self.R_gas * 100) * (1 / self.T_base - 1 / self.T_op))
        self.k_L_a = 0.56 * self.T_op + 27.9

        self.V_liq = 3400  # m3
        self.V_gas = 300 / 3400 * self.V_liq  # m3

        self.init_SOB = np.array([self.W_xc_in, self.W_ch_in, self.W_pr_in, self.W_li_in, self.S_su_in, self.X_su_in,
                           self.S_aa_in, self.X_aa_in, self.S_fa_in, self.X_fa_in, self.S_va_in, self.S_bu_in,
                           self.X_c4_in, self.S_pro_in, self.X_pro_in, self.S_ac_in, self.X_ac_in, self.S_h2_in,
                           self.X_h2_in, self.S_ch4_in, self.S_va_dis_in, self.S_bu_dis_in, self.S_pro_dis_in,
                           self.S_ac_dis_in, self.S_hco3_in, self.S_IC_in, self.S_gas_h2_in, self.S_gas_ch4_in,
                           self.S_gas_co2_in, self.S_IN_in, self.S_nh3_in, self.S_I_in, self.W_I_in, self.S_cat_in,
                           self.S_an_in, self.S_so42_in, self.X_acSRB_in, self.X_actSRB_in, self.X_h2tSRB_in,
                           self.p_gas_h2s_in, self.S_h2s_in, self.S_o2_in, self.S_gas_o2_in], dtype=np.float64)

        self.init_0 = np.array([self.W_xc_in, self.W_ch_in, self.W_pr_in, self.W_li_in, self.S_su_in, self.X_su_in,
                       self.S_aa_in, self.X_aa_in, self.S_fa_in, self.X_fa_in, self.S_va_in, self.S_bu_in,
                       self.X_c4_in, self.S_pro_in, self.X_pro_in, self.S_ac_in, self.X_ac_in, self.S_h2_in,
                       self.X_h2_in, self.S_ch4_in, self.S_va_dis_in, self.S_bu_dis_in, self.S_pro_dis_in,
                       self.S_ac_dis_in, self.S_hco3_in, self.S_IC_in, self.S_gas_h2_in, self.S_gas_ch4_in,
                       self.S_gas_co2_in, self.S_IN_in, self.S_nh3_in, self.S_I_in, self.W_I_in, self.S_cat_in,
                       self.S_an_in], dtype=np.float64)

        self.init = np.array([self.W_xc_in, self.W_ch_in, self.W_pr_in, self.W_li_in, self.S_su_in, self.X_su_in,
                           self.S_aa_in, self.X_aa_in, self.S_fa_in, self.X_fa_in, self.S_va_in, self.S_bu_in,
                           self.X_c4_in, self.S_pro_in, self.X_pro_in, self.S_ac_in, self.X_ac_in, self.S_h2_in,
                           self.X_h2_in, self.S_ch4_in, self.S_va_dis_in, self.S_bu_dis_in, self.S_pro_dis_in,
                           self.S_ac_dis_in, self.S_hco3_in, self.S_IC_in, self.S_gas_h2_in, self.S_gas_ch4_in,
                           self.S_gas_co2_in, self.S_IN_in, self.S_nh3_in, self.S_I_in, self.W_I_in, self.S_cat_in,
                           self.S_an_in, self.S_so42_in, self.X_acSRB_in, self.X_actSRB_in, self.X_h2tSRB_in,
                           self.p_gas_h2s_in, self.S_h2s_in], dtype=np.float64)

    def ade_SOB(self, x, t):
        Dil = self.action[0]
        W_xc = x[0]
        W_ch = x[1]
        W_pr = x[2]
        W_li = x[3]
        S_su = x[4]
        X_su = x[5]
        S_aa = x[6]
        X_aa = x[7]
        S_fa = x[8]
        X_fa = x[9]
        S_va = x[10]
        S_bu = x[11]
        X_c4 = x[12]
        S_pro = x[13]
        X_pro = x[14]
        S_ac = x[15]
        X_ac = x[16]
        S_h2 = x[17]
        X_h2 = x[18]
        S_ch4 = x[19]
        S_va_dis = x[20]
        S_bu_dis = x[21]
        S_pro_dis = x[22]
        S_ac_dis = x[23]
        S_hco3 = x[24]
        S_IC = x[25]
        S_gas_h2 = x[26]
        S_gas_ch4 = x[27]
        S_gas_co2 = x[28]
        S_IN = x[29]
        S_nh3 = x[30]
        S_I = x[31]
        W_I = x[32]
        S_cat = x[33]
        S_an = x[34]
        S_so42 = x[35]
        X_acSRB = x[36]
        X_actSRB = x[37]
        X_h2tSRB = x[38]
        p_gas_h2s = x[39]
        S_h2s = x[40]
        S_o2 = x[41]
        S_gas_o2 = x[42]

        S_co2 = S_IC - S_hco3
        S_nh4 = S_IN - S_nh3
        theta = S_cat + S_nh4 - S_hco3 - S_ac_dis / 64 - \
                S_pro_dis / 112 - S_bu_dis / 160 - S_va_dis / 208 - S_an

        S_Hplus = (-theta / 2 + 0.5 * np.sqrt(theta ** 2 + 4 * self.K_w))

        if S_Hplus > 0.0001:
            S_Hplus = 0.00001
        if S_Hplus == 0:
            S_Hplus = 1e-14
        p_H1 = -math.log10(S_Hplus)
        self.ph.append(p_H1)

        KpH_aa = 10 ** ((-self.pH_LL_aa + self.pH_UL_aa) / 2)
        KpH_ac = 10 ** ((-self.pH_LL_ac + self.pH_UL_ac) / 2)
        KpH_h2 = 10 ** ((-self.pH_LL_h2 + self.pH_UL_h2) / 2)

        I_pH_aa = KpH_aa ** 2 / (S_Hplus ** 2 + KpH_aa ** 2)
        I_pH_ac = KpH_ac ** 3 / (S_Hplus ** 3 + KpH_ac ** 3)
        I_pH_h2 = KpH_h2 ** 3 / (S_Hplus ** 3 + KpH_h2 ** 3)

        I_h2_fa = 1 / (1 + S_h2 / self.K_Ih2_fa)
        I_h2_c4 = 1 / (1 + S_h2 / self.K_Ih2_c4)
        I_h2_pro = 1 / (1 + S_h2 / self.K_Ih2_pro)
        I_nh3 = 1 / (1 + S_nh3/self.K_I_nh3)
        if S_o2 == 0:
            I_o2_ch4 = 1
        else:
            I_o2_ch4 = 1 / (1 + S_o2 / self.K_I_o2)

        I_IN_lim = 1 / (1 + self.K_S_IN / S_IN)
        I5 = I_pH_aa * I_IN_lim
        I6 = I_pH_aa * I_IN_lim
        I7 = I_pH_aa * I_IN_lim * I_h2_fa
        I8 = I_pH_aa * I_IN_lim * I_h2_c4
        I9 = I_pH_aa * I_IN_lim * I_h2_c4
        I10 = I_pH_aa * I_IN_lim * I_h2_pro
        I11 = I_pH_ac * I_IN_lim * I_nh3
        I12 = I_pH_h2 * I_IN_lim

        Sdis_h2s = S_h2s / (self.m6 * (1 + self.K_a1_h2s / S_Hplus + self.K_a1_h2s * self.K_a2_h2s / S_Hplus ** 2))

        rho1 = self.k_dis * W_xc
        rho2 = self.k_hyd_ch * W_ch
        rho3 = self.k_hyd_pr * W_pr
        rho4 = self.k_hyd_li * W_li
        rho5 = self.km_su * S_su * (1 - Sdis_h2s / self.KI_FB) / (self.ks_su + S_su) * X_su * I5
        rho6 = self.km_aa * S_aa * (1 - Sdis_h2s / self.KI_FB) / (self.ks_aa + S_aa) * X_aa * I6
        rho7 = self.km_fa * S_fa * (1 - Sdis_h2s / self.KI_FB) / (self.ks_fa + S_fa) * X_fa * I7
        rho8 = self.km_c4 * S_va * (1 - Sdis_h2s / self.KI_FB) / (self.ks_c4 + S_va) * \
               S_va / (S_bu + S_va + 10 ** (-6)) * X_c4 * I8
        rho9 = self.km_c4 * S_bu * (1 - Sdis_h2s / self.KI_FB) / (self.ks_c4 + S_bu) * \
               S_bu / (S_bu + S_va + 10 ** (-6)) * X_c4 * I9
        rho10 = self.km_pro * S_pro * (1 - Sdis_h2s / self.KI_AB) / (self.ks_pro + S_pro) * X_pro * I10
        rho11 = self.km_ac * S_ac * (1 - Sdis_h2s / self.KI_act) / (self.ks_ac + S_ac) * X_ac * I11
        rho12 = self.km_h2 * S_h2 * (1 - Sdis_h2s / self.KI_h2t) / (self.ks_h2 + S_h2) * X_h2 * I12
        rho13 = self.k_dec_Xsu * X_su
        rho14 = self.k_dec_Xaa * X_aa
        rho15 = self.k_dec_Xfa * X_fa
        rho16 = self.k_dec_Xc4 * X_c4
        rho17 = self.k_dec_Xpro * X_pro
        rho18 = self.k_dec_Xac * X_ac
        rho19 = self.k_dec_Xh2 * X_h2

        p_gas_h2 = S_gas_h2 * self.R_gas * self.T_op / 16
        p_gas_ch4 = S_gas_ch4 * self.R_gas * self.T_op / 64
        p_gas_co2 = S_gas_co2 * self.R_gas * self.T_op
        p_gas = p_gas_h2 + p_gas_ch4 + p_gas_co2 + self.p_gas_h2o + p_gas_h2s
        q_gas = self.k_p * (p_gas - self.p_atm) * p_gas / self.p_atm
        p_gas_o2 = 0.05 * 0.21 * p_gas  # riduzione a percentuale di ossigeno
        # perc_o2 = p_gas_o2/p_gas

        s_1 = -self.C_xc + self.f_sI_xc * self.C_sI + self.f_ch_xc * self.C_ch + self.f_pr_xc * self.C_pr + self.f_li_xc * self.C_li + self.f_xI_xc * self.C_xI
        s_2 = -self.C_ch + self.C_su
        s_3 = -self.C_pr + self.C_aa
        s_4 = -self.C_li + (1 - self.f_fa_li) * self.C_su + self.f_fa_li * self.C_fa
        s_5 = -self.C_su + (1 - self.Y_su) * (self.f_bu_su * self.C_bu + self.f_pro_su * self.C_pro + self.f_ac_su * self.C_ac) + self.Y_su * self.C_bac
        s_6 = -self.C_aa + (1 - self.Y_aa) * (self.f_va_aa * self.C_va + self.f_bu_aa * self.C_bu +
                                    self.f_pro_aa * self.C_pro + self.f_ac_aa * self.C_ac) + self.Y_aa * self.C_bac
        s_7 = -self.C_fa + (1 - self.Y_fa) * 0.7 * self.C_ac + self.Y_fa * self.C_bac
        s_8 = -self.C_va + (1 - self.Y_c4) * 0.54 * self.C_pro + (1 - self.Y_c4) * 0.31 * self.C_ac + self.Y_c4 * self.C_bac
        s_9 = -self.C_bu + (1 - self.Y_c4) * 0.8 * self.C_ac + self.Y_c4 * self.C_bac
        s_10 = -self.C_pro + (1 - self.Y_pro) * 0.57 * self.C_ac + self.Y_pro * self.C_bac
        s_11 = -self.C_ac + (1 - self.Y_ac) * self.C_ch4 + self.Y_ac * self.C_bac
        s_12 = (1 - self.Y_h2) * self.C_ch4 + self.Y_h2 * self.C_bac
        s_13 = -self.C_bac + self.C_xc
        rhoA4 = self.k_A_Bva * (S_va_dis * (self.K_a_va + S_Hplus) - self.K_a_va * S_va)
        rhoA5 = self.k_A_Bbu * (S_bu_dis * (self.K_a_bu + S_Hplus) - self.K_a_bu * S_bu)
        rhoA6 = self.k_A_Bpro * (S_pro_dis * (self.K_a_pro + S_Hplus) - self.K_a_pro * S_pro)
        rhoA7 = self.k_A_Bac * (S_ac_dis * (self.K_a_ac + S_Hplus) - self.K_a_ac * S_ac)
        rhoA10 = self.k_A_B_co2 * (S_hco3 * (self.K_a_co2 + S_Hplus) - self.K_a_co2 * S_IC)
        rhoA11 = self.k_A_BIN * (S_nh3 * (self.K_a_IN + S_Hplus) - self.K_a_IN * S_IN)

        rhoT8 = self.k_L_a * (S_h2 - 16 * self.K_H_h2 * p_gas_h2)
        rhoT9 = self.k_L_a * (S_ch4 - 64 * self.K_H_ch4 * p_gas_ch4)
        rhoT10 = self.k_L_a * (S_co2 - self.K_H_co2 * p_gas_co2)

        rhoTo2 = self.k_L_a * (S_o2 - 0.000434 * p_gas_o2)

        Sdis_co2 = S_co2 / (1 + self.K_a1_co2 / S_Hplus + self.K_a1_co2 * self.K_a2_co2 / S_Hplus ** 2)
        G = self.V_liq * self.k_L_a * ((S_h2 - p_gas_h2 / self.K_H_h2) * self.Vs_h2 + (Sdis_h2s - p_gas_h2s /self.H_h2s) *
                             self.Vs_h2s + (S_ch4 - p_gas_ch4 / self.K_H_ch4) * self.Vs_ch4 + (
                                         Sdis_co2 - p_gas_co2 / self.K_H_co2) * self.Vs_co2)

        rho20 = self.km_acSRB * X_acSRB * S_so42 * S_pro * \
                (1 - Sdis_h2s / self.KI_so42_ac) / ((self.Ks_so42_ac + S_pro) * (self.K1 + S_so42))
        rho21 = self.km_actSRB * X_actSRB * S_so42 * S_ac * \
                (1 - Sdis_h2s / self.KI_so42_act) / ((self.Ks_so42_act + S_ac) * (self.K2 + S_so42))
        rho22 = self.km_h2tSRB * X_h2tSRB * S_so42 * S_h2 * \
                (1 - Sdis_h2s / self.KI_so42_h2t) / ((self.Ks_so42_h2t + S_h2) * (self.K3 + S_so42))


        if S_h2s < 0:
            S_h2s = 10 ** (-10)

        y = np.empty(43)

        y[0] = Dil * (self.W_xc_in - W_xc) - rho1 + rho13 + rho14 + \
               rho15 + rho16 + rho17 + rho18 + rho19

        y[1] = Dil * (self.W_ch_in - W_ch) + self.f_ch_xc * rho1 - rho2

        y[2] = Dil * (self.W_pr_in - W_pr) + self.f_pr_xc * rho1 - rho3

        y[3] = Dil * (self.W_li_in - W_li) + self.f_li_xc * rho1 - rho4

        y[4] = Dil * (-S_su) + rho2 + (1 - self.f_fa_li) * rho4 - rho5

        y[5] = Dil * (-X_su) + self.Y_su * rho5 - rho13

        y[6] = Dil * (-S_aa) + rho3 - rho6

        y[7] = Dil * (-X_aa) + self.Y_aa * rho6 - rho14

        y[8] = Dil * (-S_fa) + self.f_fa_li * rho4 - rho7

        y[9] = Dil * (-X_fa) + self.Y_fa * rho7 - rho15

        y[10] = Dil * (-S_va) + (1 - self.Y_aa) * self.f_va_aa * rho6 - rho8

        y[11] = Dil * (-S_bu) + (1 - self.Y_su) * self.f_bu_su * rho5 + (1 - self.Y_aa) * self.f_bu_aa * rho6 - rho9

        y[12] = Dil * (-X_c4) + self.Y_c4 * rho8 + self.Y_c4 * rho9 - rho16

        y[13] = Dil * (-S_pro) + (1 - self.Y_su) * self.f_pro_su * rho5 + \
                (1 - self.Y_aa) * self.f_pro_aa * rho6 + (1 - self.Y_c4) * self.f_pro_va * rho8 - rho10

        y[14] = Dil * (-X_pro) + self.Y_pro * rho10 - rho17

        y[15] = Dil * (-S_ac) + (1 - self.Y_su) * self.f_ac_su * rho5 + (1 - self.Y_aa) * self.f_ac_aa * rho6 + (1 - self.Y_fa) * self.f_ac_li * \
                rho7 + (1 - self.Y_c4) * self.f_ac_va * rho8 + (1 - self.Y_c4) * self.f_ac_bu * \
                rho9 + (1 - self.Y_pro) * self.f_ac_pro * rho10 - rho11 + (1 - self.k * self.Y_ac) * self.f_ac_ac * rho20 - rho21

        y[16] = Dil * (-X_ac) + self.Y_ac * rho11 - rho18

        y[17] = Dil * (-S_h2) + (1 - self.Y_su) * self.f_h2_su * rho5 + (1 - self.Y_aa) * self.f_h2_aa * rho6 + (1 - self.Y_fa) * self.f_h2_li * \
                rho7 + (1 - self.Y_c4) * self.f_h2_va * rho8 + (1 - self.Y_c4) * self.f_h2_bu * \
                rho9 + (1 - self.Y_pro) * self.f_h2_pro * rho10 - rho12 - rhoT8 - rho22

        y[18] = Dil * (-X_h2) + self.Y_h2 * rho12 - rho19

        y[19] = Dil * (-S_ch4) + (1 - self.Y_ac) * rho11 * I_o2_ch4 + (1 - self.Y_h2) * rho12 * I_o2_ch4 - rhoT9

        y[20] = -rhoA4

        y[21] = -rhoA5

        y[22] = -rhoA6

        y[23] = -rhoA7

        y[24] = -rhoA10

        y[25] = Dil * (self.S_IC_in - S_IC) - (
                    s_1 * rho1 + s_2 * rho2 + s_3 * rho3 + s_4 * rho4 + s_5 * rho5 + s_6 * rho6 + s_7 * rho7 + s_8 *
                    rho8 + s_9 * rho9 + s_10 * rho10 + s_11 * rho11 + s_12 * rho12 + s_13 * (
                                rho13 + rho14 + rho15 + rho16 + rho17 + rho18 + rho19)) - rhoT10

        y[26] = -S_gas_h2 * q_gas / self.V_gas + rhoT8 * self.V_liq / self.V_gas

        y[27] = -S_gas_ch4 * q_gas / self.V_gas + rhoT9 * self.V_liq / self.V_gas

        y[28] = -S_gas_co2 * q_gas / self.V_gas + rhoT10 * self.V_liq / self.V_gas

        y[29] = Dil * (self.S_IN_in - S_IN) - self.Y_su * self.N_bac * rho5 + (
                self.N_aa - self.Y_aa * self.N_bac) * rho6 - self.Y_fa * self.N_bac * rho7 - self.Y_c4 * self.N_bac * rho8 - self.Y_c4 * self.N_bac * rho9 - self.Y_pro * self.N_bac * rho10 - \
                self.Y_ac * self.N_bac * rho11 - self.Y_h2 * self.N_bac * rho12 + \
                (self.N_bac - self.N_xc) * (rho13 + rho14 + rho15 + rho16 + rho17 + rho18 + rho19) + \
                (self.N_xc - self.f_xI_xc * self.N_I - self.f_sI_xc * self.N_I - self.f_pr_xc * self.N_aa) * rho1

        y[30] = -rhoA11

        y[31] = Dil * (self.S_I_in - S_I) + self.f_sI_xc * rho1

        y[32] = Dil * (self.W_I_in - W_I) + self.f_xI_xc * rho1

        y[33] = Dil * (self.S_cat_in - S_cat)

        y[34] = Dil * (self.S_an_in - S_an)

        y[35] = Dil * (self.S_so42_in - S_so42) - self.f_so42_ac * (1 - self.k * self.Y_ac) * rho20 - \
                self.f_so42_act * (1 - self.k * self.Y_act) * rho21 - self.f_so42_h2t * (1 - self.k * self.Y_h2t) * rho22

        y[36] = Dil * (-X_acSRB) + rho20 - self.k_dec_acSRB * X_acSRB

        y[37] = Dil * (-X_actSRB) + rho21 - self.k_dec_actSRB * X_actSRB

        y[38] = Dil * (-X_h2tSRB) + rho22 - self.k_dec_h2tSRB * X_h2tSRB

        y[39] = (self.k_L_a * (Sdis_h2s - p_gas_h2s / self.H_h2s) *
                 self.Vs_h2s * self.V_liq * p_gas - G * p_gas_h2s) / self.V_gas

        y[40] = Dil * (self.S_h2s_in - S_h2s) + self.f_h2s_ac * (1 - self.k * self.Y_ac) * rho20 + self.f_h2s_act * \
                (1 - self.k * self.Y_act) * rho21 + self.f_h2s_h2t * (1 - self.k * self.Y_h2t) * rho22 - self.k_L_a * (Sdis_h2s - p_gas_h2s / self.H_h2s) \
                - self.km_o2 * S_o2 / (self.ko2 + S_o2) * S_h2s ** self.alpha

        y[41] = Dil * (self.S_o2_in - S_o2) - rhoTo2 - self.km_o2 * S_o2 / (self.ko2 + S_o2) * S_h2s ** self.alpha

        y[42] = -S_gas_o2 * q_gas / self.V_gas + rhoTo2 * self.V_liq / self.V_gas - self.km_o2 * S_o2 / (self.ko2 + S_o2) * S_h2s ** self.alpha

        return y

    def ade(self, x, t):
        Dil = self.action[0]
        q_in =Dil*self.V_liq
        W_xc = x[0]
        W_ch = x[1]
        W_pr = x[2]
        W_li = x[3]
        S_su = x[4]
        X_su = x[5]
        S_aa = x[6]
        X_aa = x[7]
        S_fa = x[8]
        X_fa = x[9]
        S_va = x[10]
        S_bu = x[11]
        X_c4 = x[12]
        S_pro = x[13]
        X_pro = x[14]
        S_ac = x[15]
        X_ac = x[16]
        S_h2 = x[17]
        X_h2 = x[18]
        S_ch4 = x[19]
        S_va_dis = x[20]
        S_bu_dis = x[21]
        S_pro_dis = x[22]
        S_ac_dis = x[23]
        S_hco3 = x[24]
        S_IC = x[25]
        S_gas_h2 = x[26]
        S_gas_ch4 = x[27]
        S_gas_co2 = x[28]
        S_IN = x[29]
        S_nh3 = x[30]
        S_I = x[31]
        W_I = x[32]
        S_cat = x[33]
        S_an = x[34]
        S_co2 = S_IC - S_hco3
        S_nh4 = S_IN - S_nh3
        theta = S_cat + S_nh4 - S_hco3 - S_ac_dis / 64 - S_pro_dis / 112 - S_bu_dis / 160 - S_va_dis / 208 - S_an

        S_Hplus = (-theta / 2 + 0.5 * np.sqrt(theta ** 2 + 4 * self.K_w))

        if S_Hplus > 0.0001:
            S_Hplus = 0.00001

        p_H1 = -math.log(S_Hplus, 10)
        pHstd = 7

        if p_H1 < 5:
            p_H = pHstd
        else:
            p_H = p_H1

        KpH_aa = 10 ** ((-self.pH_LL_aa + self.pH_UL_aa) / 2)
        KpH_ac = 10 ** ((-self.pH_LL_ac + self.pH_UL_ac) / 2)
        KpH_h2 = 10 ** ((-self.pH_LL_h2 + self.pH_UL_h2) / 2)

        I_pH_aa = KpH_aa ** 2 / (S_Hplus ** 2 + KpH_aa ** 2)
        I_pH_ac = KpH_ac ** 3 / (S_Hplus ** 3 + KpH_ac ** 3)
        I_pH_h2 = KpH_h2 ** 3 / (S_Hplus ** 3 + KpH_h2 ** 3)

        I_TS = 1  # /(1+S_TS/K_TS)
        I_IN_lim = 1 / (1 + self.K_S_IN / S_IN)
        I_h2_fa = 1 / (1 + S_h2 / self.K_Ih2_fa)
        I_h2_c4 = 1 / (1 + S_h2 / self.K_Ih2_c4)
        I_h2_pro = 1 / (1 + S_h2 / self.K_Ih2_pro)
        I_nh3 = 1 / (1 + S_nh3 / self.K_I_nh3)
        I5 = I_pH_aa * I_IN_lim
        I6 = I_pH_aa * I_IN_lim
        I7 = I_pH_aa * I_IN_lim * I_h2_fa
        I8 = I_pH_aa * I_IN_lim * I_h2_c4
        I9 = I_pH_aa * I_IN_lim * I_h2_c4
        I10 = I_pH_aa * I_IN_lim * I_h2_pro
        I11 = I_pH_ac * I_IN_lim * I_nh3
        I12 = I_pH_h2 * I_IN_lim

        rho1 = self.k_dis * W_xc
        rho2 = self.k_hyd_ch * W_ch * I_TS
        rho3 = self.k_hyd_pr * W_pr * I_TS
        rho4 = self.k_hyd_li * W_li * I_TS
        rho5 = self.km_su * S_su / (self.ks_su + S_su) * X_su * I5
        rho6 = self.km_aa * S_aa / (self.ks_aa + S_aa) * X_aa * I6
        rho7 = self.km_fa * S_fa / (self.ks_fa + S_fa) * X_fa * I7
        rho8 = self.km_c4 * S_va / (self.ks_c4 + S_va) * S_va / (S_bu + S_va + 10 ** (-6)) * X_c4 * I8
        rho9 = self.km_c4 * S_bu / (self.ks_c4 + S_bu) * S_bu / (S_bu + S_va + 10 ** (-6)) * X_c4 * I9
        rho10 = self.km_pro * S_pro / (self.ks_pro + S_pro) * X_pro * I10
        rho11 = self.km_ac * S_ac / (self.ks_ac + S_ac) * X_ac * I11
        rho12 = self.km_h2 * S_h2 / (self.ks_h2 + S_h2) * X_h2 * I12
        rho13 = self.k_dec_Xsu * X_su
        rho14 = self.k_dec_Xaa * X_aa
        rho15 = self.k_dec_Xfa * X_fa
        rho16 = self.k_dec_Xc4 * X_c4
        rho17 = self.k_dec_Xpro * X_pro
        rho18 = self.k_dec_Xac * X_ac
        rho19 = self.k_dec_Xh2 * X_h2

        p_gas_h2 = S_gas_h2 * self.R_gas * self.T_op / 16
        p_gas_ch4 = S_gas_ch4 * self.R_gas * self.T_op / 64
        p_gas_co2 = S_gas_co2 * self.R_gas * self.T_op
        p_gas = p_gas_h2 + p_gas_ch4 + p_gas_co2 + self.p_gas_h2o
        q_gas = self.k_p * (p_gas - self.p_atm) * p_gas / self.p_atm

        s_1 = -self.C_xc + self.f_sI_xc *self.C_sI + self.f_ch_xc * self.C_ch + self.f_pr_xc *self.C_pr + self.f_li_xc * self.C_li + self.f_xI_xc * self.C_xI
        s_2 = -self.C_ch +self. C_su
        s_3 = -self.C_pr + self.C_aa
        s_4 = -self.C_li + (1 - self.f_fa_li) * self.C_su + self.f_fa_li * self.C_fa
        s_5 = -self.C_su + (1 - self.Y_su) * (self.f_bu_su * self.C_bu + self.f_pro_su * self.C_pro + self.f_ac_su * self.C_ac) + self.Y_su * self.C_bac
        s_6 = -self.C_aa + (1 - self.Y_aa) * (self.f_va_aa * self.C_va + self.f_bu_aa * self.C_bu + self.f_pro_aa * self.C_pro + self.f_ac_aa * self.C_ac) + self.Y_aa * self.C_bac
        s_7 = -self.C_fa + (1 - self.Y_fa) * 0.7 * self.C_ac + self.Y_fa * self.C_bac
        s_8 = -self.C_va + (1 - self.Y_c4) * 0.54 * self.C_pro + (1 - self.Y_c4) * 0.31 * self.C_ac + self.Y_c4 * self.C_bac
        s_9 = -self.C_bu + (1 - self.Y_c4) * 0.8 * self.C_ac + self.Y_c4 * self.C_bac
        s_10 = -self.C_pro + (1 - self.Y_pro) * 0.57 * self.C_ac + self.Y_pro * self.C_bac
        s_11 = -self.C_ac + (1 - self.Y_ac) * self.C_ch4 + self.Y_ac * self.C_bac
        s_12 = (1 - self.Y_h2) * self.C_ch4 + self.Y_h2 * self.C_bac
        s_13 = -self.C_bac + self.C_xc

        if (S_IC - S_hco3) < 0:
            S_hco3 = S_IC

        rhoA4 = self.k_A_Bva * (S_va_dis * (self.K_a_va + S_Hplus) - self.K_a_va * S_va)
        rhoA5 = self.k_A_Bbu * (S_bu_dis * (self.K_a_bu + S_Hplus) - self.K_a_bu * S_bu)
        rhoA6 = self.k_A_Bpro * (S_pro_dis * (self.K_a_pro + S_Hplus) - self.K_a_pro * S_pro)
        rhoA7 = self.k_A_Bac * (S_ac_dis * (self.K_a_ac + S_Hplus) - self.K_a_ac * S_ac)
        rhoA10 = self.k_A_B_co2 * (S_hco3 * (self.K_a_co2 + S_Hplus) - self.K_a_co2 * S_IC)
        rhoA11 = self.k_A_BIN * (S_nh3 * (self.K_a_IN + S_Hplus) - self.K_a_IN * S_IN)

        rhoT8 = self.k_L_a * (S_h2 - 16 * self.K_H_h2 * p_gas_h2)
        rhoT9 = self.k_L_a * (S_ch4 - 64 * self.K_H_ch4 * p_gas_ch4)
        rhoT10 = self.k_L_a * (S_co2 - self.K_H_co2 * p_gas_co2)

        y = np.empty(35)

        y[0] = Dil * (self.W_xc_in - W_xc) - rho1 + rho13 + rho14 + rho15 + rho16 + rho17 + rho18 + rho19

        y[1] = Dil * (self.W_ch_in - W_ch) + self.f_ch_xc * rho1 - rho2

        y[2] = Dil * (self.W_pr_in - W_pr) + self.f_pr_xc * rho1 - rho3

        y[3] = Dil * (self.W_li_in - W_li) + self.f_li_xc * rho1 - rho4

        y[4] = Dil * (self.S_su_in - S_su) + rho2 + (1 - self.f_fa_li) * rho4 - rho5

        y[5] = Dil * (self.X_su_in - X_su) + self.Y_su * rho5 - rho13

        y[6] = Dil * (self.S_aa_in - S_aa) + rho3 - rho6

        y[7] = Dil * (self.X_aa_in - X_aa) + self.Y_aa * rho6 - rho14

        y[8] = Dil * (self.S_fa_in - S_fa) + self.f_fa_li * rho4 - rho7

        y[9] = Dil * (self.X_fa_in - X_fa) + self.Y_fa * rho7 - rho15

        y[10] = Dil * (self.S_va_in - S_va) + (1 - self.Y_aa) * self.f_va_aa * rho6 - rho8

        y[11] = Dil * (self.S_bu_in - S_bu) + (1 - self.Y_su) * self.f_bu_su * rho5 + (1 - self.Y_aa) * self.f_bu_aa * rho6 - rho9

        y[12] = Dil * (self.X_c4_in - X_c4) + self.Y_c4 * rho8 + self.Y_c4 * rho9 - rho16

        y[13] = q_in / self.V_liq * (self.S_pro_in - S_pro) + (1 - self.Y_su) * self.f_pro_su * rho5 + (1 - self.Y_aa) * \
                self.f_pro_aa * rho6 + (1 - self.Y_c4) * self.f_pro_va * rho8 - rho10

        y[14] = Dil * (self.X_pro_in - X_pro) + self.Y_pro * rho10 - rho17

        y[15] = Dil * (self.S_ac_in - S_ac) + (1 - self.Y_su) * self.f_ac_su * rho5 + (1 - self.Y_aa) * self.f_ac_aa * rho6 + (1 - self.Y_fa) * \
                self.f_ac_li * rho7 + (1 -self. Y_c4) * self.f_ac_va * rho8 + (1 - self.Y_c4) * self.f_ac_bu * rho9 + (
                            1 - self.Y_pro) * self.f_ac_pro * rho10 - rho11

        y[16] = Dil * (self.X_ac_in - X_ac) + self.Y_ac * rho11 - rho18

        y[17] = Dil * (self.S_h2_in - S_h2) + (1 - self.Y_su) * self.f_h2_su * rho5 + (1 - self.Y_aa) * self.f_h2_aa * rho6 + (1 - self.Y_fa) * \
                self.f_h2_li * rho7 + (1 - self.Y_c4) * self.f_h2_va * rho8 + (1 - self.Y_c4) * self.f_h2_bu * \
                rho9 + (1 - self.Y_pro) * self.f_h2_pro * rho10 - rho12 - rhoT8

        y[18] = Dil * (self.X_h2_in - X_h2) + self.Y_h2 * rho12 - rho19

        y[19] = Dil * (self.S_ch4_in - S_ch4) + (1 - self.Y_ac) * rho11 + (1 - self.Y_h2) * rho12 - rhoT9

        y[20] = -rhoA4

        y[21] = -rhoA5

        y[22] = -rhoA6

        y[23] = -rhoA7

        y[24] = -rhoA10

        y[25] = Dil * (self.S_IC_in - S_IC) - (
                    s_1 * rho1 + s_2 * rho2 + s_3 * rho3 + s_4 * rho4 + s_5 * rho5 + s_6 * rho6 + s_7 * rho7 + s_8 *
                    rho8 + s_9 * rho9 + s_10 * rho10 + s_11 * rho11 + s_12 * rho12 + s_13 * (
                                rho13 + rho14 + rho15 + rho16 + rho17 + rho18 + rho19)) - rhoT10

        y[26] = -S_gas_h2 * q_gas / self.V_gas + rhoT8 * self.V_liq / self.V_gas

        y[27] = -S_gas_ch4 * q_gas / self.V_gas + rhoT9 * self.V_liq / self.V_gas

        y[28] = -S_gas_co2 * q_gas / self.V_gas + rhoT10 * self.V_liq / self.V_gas

        y[29] = Dil * (self.S_IN_in - S_IN) - self.Y_su * self.N_bac * rho5 + (
                    self.N_aa - self.Y_aa * self.N_bac) * rho6 - self.Y_fa * self.N_bac * rho7 - self.Y_c4 * self.N_bac * rho8 - self.Y_c4 * self.N_bac * rho9 - self.Y_pro * self.N_bac * rho10 - \
                self.Y_ac * self.N_bac * rho11 - self.Y_h2 * self.N_bac * rho12 + \
                (self.N_bac - self.N_xc) * (rho13 + rho14 + rho15 + rho16 + rho17 + rho18 + rho19) + \
                (self.N_xc - self.f_xI_xc * self.N_I - self.f_sI_xc * self.N_I - self.f_pr_xc * self.N_aa) * rho1

        y[30] = -rhoA11

        y[31] = Dil * (self.S_I_in - S_I) + self.f_sI_xc * rho1

        y[32] = Dil * (self.W_I_in - W_I) + self.f_xI_xc * rho1

        y[33] = Dil * (self.S_cat_in - S_cat)

        y[34] = Dil * (self.S_an_in - S_an)

        return y

    def ade_SRB(self, x, t):
        Dil = self.action[0]
        q_in = Dil*self.V_liq
        W_xc = x[0]
        W_ch = x[1]
        W_pr = x[2]
        W_li = x[3]
        S_su = x[4]
        X_su = x[5]
        S_aa = x[6]
        X_aa = x[7]
        S_fa = x[8]
        X_fa = x[9]
        S_va = x[10]
        S_bu = x[11]
        X_c4 = x[12]
        S_pro = x[13]
        X_pro = x[14]
        S_ac = x[15]
        X_ac = x[16]
        S_h2 = x[17]
        X_h2 = x[18]
        S_ch4 = x[19]
        S_va_dis = x[20]
        S_bu_dis = x[21]
        S_pro_dis = x[22]
        S_ac_dis = x[23]
        S_hco3 = x[24]
        S_IC = x[25]
        S_gas_h2 = x[26]
        S_gas_ch4 = x[27]
        S_gas_co2 = x[28]
        S_IN = x[29]
        S_nh3 = x[30]
        S_I = x[31]
        W_I = x[32]
        S_cat = x[33]
        S_an = x[34]
        S_so42 = x[35]
        X_acSRB = x[36]
        X_actSRB = x[37]
        X_h2tSRB = x[38]
        p_gas_h2s = x[39]
        S_h2s = x[40]

        S_co2 = S_IC - S_hco3
        S_nh4 = S_IN - S_nh3
        theta = S_cat + S_nh4 - S_hco3 - S_ac_dis / 64 - \
                S_pro_dis / 112 - S_bu_dis / 160 - S_va_dis / 208 - S_an

        S_Hplus = (-theta / 2 + 0.5 * np.sqrt(theta ** 2 + 4 * self.K_w))

        if S_Hplus > 0.0001:
            S_Hplus = 0.00001
        p_H1 = -math.log(S_Hplus, 10)
        self.ph.append(p_H1)

        KpH_aa = 10 ** ((-self.pH_LL_aa + self.pH_UL_aa) / 2)
        KpH_ac = 10 ** ((-self.pH_LL_ac + self.pH_UL_ac) / 2)
        KpH_h2 = 10 ** ((-self.pH_LL_h2 + self.pH_UL_h2) / 2)

        I_pH_aa = KpH_aa ** 2 / (S_Hplus ** 2 + KpH_aa ** 2)
        I_pH_ac = KpH_ac ** 3 / (S_Hplus ** 3 + KpH_ac ** 3)
        I_pH_h2 = KpH_h2 ** 3 / (S_Hplus ** 3 + KpH_h2 ** 3)

        I_h2_fa = 1 / (1 + S_h2 / self.K_Ih2_fa)
        I_h2_c4 = 1 / (1 + S_h2 / self.K_Ih2_c4)
        I_h2_pro = 1 / (1 + S_h2 / self.K_Ih2_pro)
        I_nh3 = 1 / (1 + S_nh3 / self.K_I_nh3)

        I_IN_lim = 1 / (1 + self.K_S_IN / S_IN)
        I5 = I_pH_aa * I_IN_lim
        I6 = I_pH_aa * I_IN_lim
        I7 = I_pH_aa * I_IN_lim * I_h2_fa
        I8 = I_pH_aa * I_IN_lim * I_h2_c4
        I9 = I_pH_aa * I_IN_lim * I_h2_c4
        I10 = I_pH_aa * I_IN_lim * I_h2_pro
        I11 = I_pH_ac * I_IN_lim * I_nh3
        I12 = I_pH_h2 * I_IN_lim

        Sdis_h2s = S_h2s / (self.m6 * (1 + self.K_a1_h2s / S_Hplus + self.K_a1_h2s * self.K_a2_h2s / S_Hplus ** 2))

        rho1 = self.k_dis * W_xc
        rho2 = self.k_hyd_ch * W_ch
        rho3 = self.k_hyd_pr * W_pr
        rho4 = self.k_hyd_li * W_li
        rho5 = self.km_su * S_su * (1 - Sdis_h2s / self.KI_FB) / (self.ks_su + S_su) * X_su * I5
        rho6 = self.km_aa * S_aa * (1 - Sdis_h2s / self.KI_FB) / (self.ks_aa + S_aa) * X_aa * I6
        rho7 = self.km_fa * S_fa * (1 - Sdis_h2s / self.KI_FB) / (self.ks_fa + S_fa) * X_fa * I7
        rho8 = self.km_c4 * S_va * (1 - Sdis_h2s / self.KI_FB) / (self.ks_c4 + S_va) * \
               S_va / (S_bu + S_va + 10 ** (-6)) * X_c4 * I8
        rho9 = self.km_c4 * S_bu * (1 - Sdis_h2s / self.KI_FB) / (self.ks_c4 + S_bu) * \
               S_bu / (S_bu + S_va + 10 ** (-6)) * X_c4 * I9
        rho10 = self.km_pro * S_pro * (1 - Sdis_h2s / self.KI_AB) / (self.ks_pro + S_pro) * X_pro * I10
        rho11 = self.km_ac * S_ac * (1 - Sdis_h2s / self.KI_act) / (self.ks_ac + S_ac) * X_ac * I11
        rho12 = self.km_h2 * S_h2 * (1 - Sdis_h2s / self.KI_h2t) / (self.ks_h2 + S_h2) * X_h2 * I12
        rho13 = self.k_dec_Xsu * X_su
        rho14 = self.k_dec_Xaa * X_aa
        rho15 = self.k_dec_Xfa * X_fa
        rho16 = self.k_dec_Xc4 * X_c4
        rho17 = self.k_dec_Xpro * X_pro
        rho18 = self.k_dec_Xac * X_ac
        rho19 = self.k_dec_Xh2 * X_h2

        p_gas_h2 = S_gas_h2 * self.R_gas * self.T_op / 16
        p_gas_ch4 = S_gas_ch4 * self.R_gas * self.T_op / 64
        p_gas_co2 = S_gas_co2 * self.R_gas * self.T_op
        p_gas = p_gas_h2 + p_gas_ch4 + p_gas_co2 + self.p_gas_h2o + p_gas_h2s
        q_gas = self.k_p * (p_gas - self.p_atm) * p_gas / self.p_atm

        s_1 = -self.C_xc + self.f_sI_xc * self.C_sI + self.f_ch_xc * self.C_ch + self.f_pr_xc * self.C_pr + self.f_li_xc * self.C_li + self.f_xI_xc * self.C_xI
        s_2 = -self.C_ch + self.C_su
        s_3 = -self.C_pr + self.C_aa
        s_4 = -self.C_li + (1 - self.f_fa_li) * self.C_su + self.f_fa_li * self.C_fa
        s_5 = -self.C_su + (1 - self.Y_su) * (self.f_bu_su * self.C_bu + self.f_pro_su * self.C_pro + self.f_ac_su * self.C_ac) + self.Y_su * self.C_bac
        s_6 = -self.C_aa + (1 - self.Y_aa) * (self.f_va_aa * self.C_va + self.f_bu_aa * self.C_bu +
                                    self.f_pro_aa * self.C_pro + self.f_ac_aa * self.C_ac) + self.Y_aa * self.C_bac
        s_7 = -self.C_fa + (1 - self.Y_fa) * 0.7 * self.C_ac + self.Y_fa * self.C_bac
        s_8 = -self.C_va + (1 - self.Y_c4) * 0.54 * self.C_pro + (1 - self.Y_c4) * 0.31 * self.C_ac + self.Y_c4 * self.C_bac
        s_9 = -self.C_bu + (1 - self.Y_c4) * 0.8 * self.C_ac + self.Y_c4 * self.C_bac
        s_10 = -self.C_pro + (1 - self.Y_pro) * 0.57 * self.C_ac + self.Y_pro * self.C_bac
        s_11 = -self.C_ac + (1 - self.Y_ac) * self.C_ch4 + self.Y_ac * self.C_bac
        s_12 = (1 - self.Y_h2) * self.C_ch4 + self.Y_h2 * self.C_bac
        s_13 = -self.C_bac + self.C_xc
        rhoA4 = self.k_A_Bva * (S_va_dis * (self.K_a_va + S_Hplus) - self.K_a_va * S_va)
        rhoA5 = self.k_A_Bbu * (S_bu_dis * (self.K_a_bu + S_Hplus) - self.K_a_bu * S_bu)
        rhoA6 = self.k_A_Bpro * (S_pro_dis * (self.K_a_pro + S_Hplus) - self.K_a_pro * S_pro)
        rhoA7 = self.k_A_Bac * (S_ac_dis * (self.K_a_ac + S_Hplus) - self.K_a_ac * S_ac)
        rhoA10 = self.k_A_B_co2 * (S_hco3 * (self.K_a_co2 + S_Hplus) - self.K_a_co2 * S_IC)
        rhoA11 = self.k_A_BIN * (S_nh3 * (self.K_a_IN + S_Hplus) - self.K_a_IN * S_IN)

        rhoT8 = self.k_L_a * (S_h2 - 16 * self.K_H_h2 * p_gas_h2)
        rhoT9 = self.k_L_a * (S_ch4 - 64 * self.K_H_ch4 * p_gas_ch4)
        rhoT10 =self. k_L_a * (S_co2 - self.K_H_co2 * p_gas_co2)

        Sdis_co2 = S_co2 / (1 + self.K_a1_co2 / S_Hplus + self.K_a1_co2 * self.K_a2_co2 / S_Hplus ** 2)
        G = self.V_liq * self.k_L_a * ((S_h2 - p_gas_h2 / self.K_H_h2) * self.Vs_h2 + (Sdis_h2s - p_gas_h2s / self.H_h2s) *
                             self.Vs_h2s + (S_ch4 - p_gas_ch4 / self.K_H_ch4) * self.Vs_ch4 + (
                                         Sdis_co2 - p_gas_co2 / self.K_H_co2) * self.Vs_co2)

        rho20 =self.km_acSRB * X_acSRB * S_so42 * S_pro * \
                (1 - Sdis_h2s / self.KI_so42_ac) / ((self.Ks_so42_ac + S_pro) * (self.K1 + S_so42))
        rho21 = self.km_actSRB * X_actSRB * S_so42 * S_ac * \
                (1 - Sdis_h2s / self.KI_so42_act) / ((self.Ks_so42_act + S_ac) * (self.K2 + S_so42))
        rho22 = self.km_h2tSRB * X_h2tSRB * S_so42 * S_h2 * \
                (1 - Sdis_h2s / self.KI_so42_h2t) / ((self.Ks_so42_h2t + S_h2) * (self.K3 + S_so42))

        if S_h2s < 0:
            S_h2s = 10 ** (-10)

        y = np.empty(41)

        y[0] = Dil * (self.W_xc_in - W_xc) - rho1 + rho13 + rho14 + \
               rho15 + rho16 + rho17 + rho18 + rho19

        y[1] = Dil * (self.W_ch_in - W_ch) + self.f_ch_xc * rho1 - rho2

        y[2] = Dil * (self.W_pr_in - W_pr) + self.f_pr_xc * rho1 - rho3

        y[3] = Dil * (self.W_li_in - W_li) + self.f_li_xc * rho1 - rho4

        y[4] = Dil * (-S_su) + rho2 + (1 - self.f_fa_li) * rho4 - rho5

        y[5] = Dil * (-X_su) + self.Y_su * rho5 - rho13

        y[6] = Dil * (-S_aa) + rho3 - rho6

        y[7] = Dil * (-X_aa) + self.Y_aa * rho6 - rho14

        y[8] = Dil * (-S_fa) + self.f_fa_li * rho4 - rho7

        y[9] = Dil * (-X_fa) + self.Y_fa * rho7 - rho15

        y[10] = Dil * (-S_va) + (1 - self.Y_aa) * self.f_va_aa * rho6 - rho8

        y[11] = Dil * (-S_bu) + (1 - self.Y_su) * self.f_bu_su * rho5 + (1 - self.Y_aa) * self.f_bu_aa * rho6 - rho9

        y[12] = Dil * (-X_c4) + self.Y_c4 * rho8 + self.Y_c4 * rho9 - rho16

        y[13] = Dil * (-S_pro) + (1 - self.Y_su) * self.f_pro_su * rho5 + \
                (1 - self.Y_aa) * self.f_pro_aa * rho6 + (1 - self.Y_c4) * self.f_pro_va * rho8 - rho10

        y[14] = Dil * (-X_pro) + self.Y_pro * rho10 - rho17

        y[15] = Dil * (-S_ac) + (1 - self.Y_su) * self.f_ac_su * rho5 + (1 - self.Y_aa) * self.f_ac_aa * rho6 + (1 - self.Y_fa) * self.f_ac_li * \
                rho7 + (1 - self.Y_c4) * self.f_ac_va * rho8 + (1 - self.Y_c4) * self.f_ac_bu * \
                rho9 + (1 - self.Y_pro) * self.f_ac_pro * rho10 - rho11 + (1 - self.k * self.Y_ac) * self.f_ac_ac * rho20 - rho21

        y[16] = Dil * (-X_ac) + self.Y_ac * rho11 - rho18

        y[17] = Dil * (-S_h2) + (1 - self.Y_su) * self.f_h2_su * rho5 + (1 - self.Y_aa) * self.f_h2_aa * rho6 + (1 - self.Y_fa) * self.f_h2_li * \
                rho7 + (1 - self.Y_c4) * self.f_h2_va * rho8 + (1 - self.Y_c4) * self.f_h2_bu * \
                rho9 + (1 - self.Y_pro) * self.f_h2_pro * rho10 - rho12 - rhoT8 - rho22

        y[18] = Dil * (-X_h2) + self.Y_h2 * rho12 - rho19

        y[19] = Dil * (-S_ch4) + (1 - self.Y_ac) * rho11 + \
                +(1 - self.Y_h2) * rho12 - rhoT9

        y[20] = -rhoA4

        y[21] = -rhoA5

        y[22] = -rhoA6

        y[23] = -rhoA7

        y[24] = -rhoA10

        y[25] = Dil * (self.S_IC_in - S_IC) - (
                    s_1 * rho1 + s_2 * rho2 + s_3 * rho3 + s_4 * rho4 + s_5 * rho5 + s_6 * rho6 + s_7 * rho7 + s_8 *
                    rho8 + s_9 * rho9 + s_10 * rho10 + s_11 * rho11 + s_12 * rho12 + s_13 * (
                                rho13 + rho14 + rho15 + rho16 + rho17 + rho18 + rho19)) - rhoT10

        y[26] = -S_gas_h2 * q_gas / self.V_gas + rhoT8 * self.V_liq / self.V_gas

        y[27] = -S_gas_ch4 * q_gas / self.V_gas + rhoT9 * self.V_liq / self.V_gas

        y[28] = -S_gas_co2 * q_gas / self.V_gas + rhoT10 * self.V_liq / self.V_gas

        y[29] = Dil * (self.S_IN_in - S_IN) - self.Y_su * self.N_bac * rho5 + (
                    self.N_aa - self.Y_aa * self.N_bac) * rho6 - self.Y_fa * self.N_bac * rho7 - self.Y_c4 * self.N_bac * rho8 - self.Y_c4 * self.N_bac * rho9 - self.Y_pro * self.N_bac * rho10 - \
                self.Y_ac * self.N_bac * rho11 - self.Y_h2 * self.N_bac * rho12 + \
                (self.N_bac - self.N_xc) * (rho13 + rho14 + rho15 + rho16 + rho17 + rho18 + rho19) + \
                (self.N_xc - self.f_xI_xc * self.N_I - self.f_sI_xc * self.N_I - self.f_pr_xc * self.N_aa) * rho1

        y[30] = -rhoA11

        y[31] = Dil * (self.S_I_in - S_I) + self.f_sI_xc * rho1

        y[32] = Dil * (self.W_I_in - W_I) + self.f_xI_xc * rho1

        y[33] = Dil * (self.S_cat_in - S_cat)

        y[34] = Dil * (self.S_an_in - S_an)

        y[35] = Dil * (self.S_so42_in - S_so42) - self.f_so42_ac * (1 - self.k * self.Y_ac) * rho20 - \
                self.f_so42_act * (1 - self.k * self.Y_act) * rho21 - self.f_so42_h2t * (1 - self.k * self.Y_h2t) * rho22

        y[36] = Dil * (-X_acSRB) + rho20 - self.k_dec_acSRB * X_acSRB

        y[37] = Dil * (-X_actSRB) + rho21 - self.k_dec_actSRB * X_actSRB

        y[38] = Dil * (-X_h2tSRB) + rho22 - self.k_dec_h2tSRB * X_h2tSRB

        y[39] = (self.k_L_a * (Sdis_h2s - p_gas_h2s / self.H_h2s) * self.Vs_h2s * self.V_liq * p_gas - G * p_gas_h2s) / self.V_gas

        y[40] = Dil * (self.S_h2s_in - S_h2s) + self.f_h2s_ac * (1 - self.k * self.Y_ac) * rho20 + self.f_h2s_act * (1 - self.k * self.Y_act) * rho21 + self.f_h2s_h2t * (1 - self.k * self.Y_h2t) * rho22 - self.k_L_a * (Sdis_h2s - p_gas_h2s / self.H_h2s) \

        return y




if __name__ == '__main__':
    env = CustomEnv()
    for i in range(env.total_step):
        #action = env.action_space.sample()
        action = -0.5
        next_state, reward, done, info = env.step(action)
    env.render()