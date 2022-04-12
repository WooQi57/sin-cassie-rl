# isolated cassie env
from cassie_m.cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis

from math import floor
import gym
from gym import spaces
import numpy as np
import os
import random
import copy
import pickle

class CassieRefEnv(gym.Env):
    def __init__(self, simrate=60, dynamics_randomization=True,
                 visual=True, config="./cassie_m/model/cassie.xml", **kwargs):
        self.config = config
        self.visual = visual
        self.sim = CassieSim(self.config)
        if self.visual:
            self.vis = CassieVis(self.sim)
        
        self.dynamics_randomization = dynamics_randomization
        self.termination = False
        
        #state buffer
        self.state_buffer = []
        self.buffer_size = 1 # 3
                
        # Observation space and State space
        self.observation_space = spaces.Box(low=-np.inf,high=np.inf,shape=(self.buffer_size*39+2+2,))
        self.action_space = spaces.Box(low=np.array([-1]*10), high=np.array([1]*10))
        
        self.P = np.array([100,  100,  88,  96,  50]) 
        self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])
        self.u = pd_in_t()
        self.foot_pos = [0]*6
        
        self.cassie_state = state_out_t()
        self.simrate = simrate  # simulate X mujoco steps with same pd target. 50 brings simulation from 2000Hz to exactly 40Hz
        self.time    = 0        # number of time steps in current episode
        self.phase   = 0        # portion of the phase the robot is in
        self.counter = 0        # number of phase cycles completed in episode
        self.time_limit = 600
        self.offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])
        self.time_buf = 0
        
        self.max_speed = 4.0
        self.min_speed = -0.3
        self.max_side_speed  = 0.3
        self.min_side_speed  = -0.3
        
        #### Dynamics Randomization ####
        
        self.max_pitch_incline = 0.03
        self.max_roll_incline = 0.03        
        self.encoder_noise = 0.01        
        self.damping_low = 0.3
        self.damping_high = 5.0
        self.mass_low = 0.5
        self.mass_high = 1.5
        self.fric_low = 0.4
        self.fric_high = 1.1
        self.speed = 0.6
        self.side_speed = 0
        self.orient_add = 0

        # Default dynamics parameters
        self.default_damping = self.sim.get_dof_damping()
        self.default_mass = self.sim.get_body_mass()
        self.default_ipos = self.sim.get_body_ipos()
        self.default_fric = self.sim.get_geom_friction()
        self.default_rgba = self.sim.get_geom_rgba()
        self.default_quat = self.sim.get_geom_quat()

        self.motor_encoder_noise = np.zeros(10)
        self.joint_encoder_noise = np.zeros(6)

        # rew_buf
        self.rew_ref = 0
        self.rew_spring = 0
        self.rew_ori = 0
        self.rew_vel = 0
        self.rew_termin = 0
        self.reward = 0
        self.rew_ref_buf = 0
        self.rew_spring_buf = 0
        self.rew_ori_buf = 0
        self.rew_vel_buf = 0
        self.rew_termin_buf = 0
        self.reward_buf = 0
        self.omega_buf = 0

        
    def custom_footheight(self):
        phase = self.phase
        h = 0.15
        h1 = max(0, h*np.sin(2*np.pi/28*phase)-0.2*h) 
        h2 = max(0, h*np.sin(np.pi + 2*np.pi/28*phase)-0.2*h) 
        return [h1,h2]

    def step_simulation(self,action):
        target = action + self.offset
        
        self.u = pd_in_t()
        
        for i in range(5):
            self.u.leftLeg.motorPd.pGain[i]  = self.P[i]
            self.u.rightLeg.motorPd.pGain[i] = self.P[i]
            self.u.leftLeg.motorPd.dGain[i]  = self.D[i]
            self.u.rightLeg.motorPd.dGain[i] = self.D[i]

            self.u.leftLeg.motorPd.torque[i]  = 0  # Feedforward torque
            self.u.rightLeg.motorPd.torque[i] = 0

            self.u.leftLeg.motorPd.pTarget[i]  = target[i]
            self.u.rightLeg.motorPd.pTarget[i] = target[i + 5]

            self.u.leftLeg.motorPd.dTarget[i]  = 0
            self.u.rightLeg.motorPd.dTarget[i] = 0

        self.cassie_state = self.sim.step_pd(self.u)  # cassie_state is different from qpos state???
            
    def step(self, action):
        
        for _ in range(self.simrate):            
            self.step_simulation(action)
            
        self.time  += 1
        self.phase += 1
        if self.phase >= 28:
            self.phase = 0
            self.counter +=1
        
        obs = self.get_state()
        self.sim.foot_pos(self.foot_pos)

        xpos, ypos, height = self.qpos[0], self.qpos[1], self.qpos[2]
        xtarget, ytarget, ztarget = self.ref_pos[0], self.ref_pos[1], self.ref_pos[2] 
        pos2target = (xpos-xtarget)**2 + (ypos-ytarget)**2 + (height-ztarget)**2
        die_radii = 0.6 + (self.speed**2 + self.side_speed**2)**0.5
        self.termination = height < 0.6 or height > 1.2 or pos2target > die_radii**2
        done = self.termination or self.time >= self.time_limit
            
        if self.visual:
            self.render()
        reward = self.compute_reward(action)
        
        return obs, reward, done, {}

    def reset(self):
        if self.time != 0 :
            self.rew_ref_buf = self.rew_ref / self.time
            self.rew_spring_buf = self.rew_spring / self.time
            self.rew_ori_buf = self.rew_ori / self.time
            self.rew_vel_buf = self.rew_vel / self.time
            self.rew_termin_buf = self.rew_termin / self.time
            self.reward_buf = self.reward # / self.time
            self.time_buf = self.time
            self.omega_buf = self.omega / self.time
        
        self.rew_ref = 0
        self.rew_spring = 0
        self.rew_ori = 0
        self.rew_vel = 0
        self.rew_termin = 0
        self.reward = 0
        self.omega = 0

        self.speed = 0.6
        self.side_speed = 0
        self.time = 0
        self.counter = 0
        self.termination = False
        self.phase = int((random.random()>0.5)*14)  # random phase: 0 or 14

        # Randomize dynamics:
        if self.dynamics_randomization:
            damp = self.default_damping

            pelvis_damp_range = [[damp[0], damp[0]],
                                [damp[1], damp[1]],
                                [damp[2], damp[2]],
                                [damp[3], damp[3]],
                                [damp[4], damp[4]],
                                [damp[5], damp[5]]]  # 0->5

            hip_damp_range = [[damp[6]*self.damping_low, damp[6]*self.damping_high],
                              [damp[7]*self.damping_low, damp[7]*self.damping_high],
                              [damp[8]*self.damping_low, damp[8]*self.damping_high]]          # 6->8 and 19->21

            achilles_damp_range = [[damp[9]*self.damping_low, damp[9]*self.damping_high],
                                   [damp[10]*self.damping_low, damp[10]*self.damping_high],
                                   [damp[11]*self.damping_low, damp[11]*self.damping_high]]   # 9->11 and 22->24

            knee_damp_range     = [[damp[12]*self.damping_low, damp[12]*self.damping_high]]   # 12 and 25
            shin_damp_range     = [[damp[13]*self.damping_low, damp[13]*self.damping_high]]   # 13 and 26
            tarsus_damp_range   = [[damp[14]*self.damping_low, damp[14]*self.damping_high]]   # 14 and 27

            heel_damp_range     = [[damp[15], damp[15]]]                                      # 15 and 28
            fcrank_damp_range   = [[damp[16]*self.damping_low, damp[16]*self.damping_high]]   # 16 and 29
            prod_damp_range     = [[damp[17], damp[17]]]                                      # 17 and 30
            foot_damp_range     = [[damp[18]*self.damping_low, damp[18]*self.damping_high]]   # 18 and 31

            side_damp = hip_damp_range + achilles_damp_range + knee_damp_range + shin_damp_range + tarsus_damp_range + heel_damp_range + fcrank_damp_range + prod_damp_range + foot_damp_range
            damp_range = pelvis_damp_range + side_damp + side_damp
            damp_noise = [np.random.uniform(a, b) for a, b in damp_range]

            m = self.default_mass
            pelvis_mass_range      = [[self.mass_low*m[1], self.mass_high*m[1]]]   # 1
            hip_mass_range         = [[self.mass_low*m[2], self.mass_high*m[2]],   # 2->4 and 14->16
                                    [self.mass_low*m[3], self.mass_high*m[3]],
                                    [self.mass_low*m[4], self.mass_high*m[4]]]

            achilles_mass_range    = [[self.mass_low*m[5], self.mass_high*m[5]]]    # 5 and 17
            knee_mass_range        = [[self.mass_low*m[6], self.mass_high*m[6]]]    # 6 and 18
            knee_spring_mass_range = [[self.mass_low*m[7], self.mass_high*m[7]]]    # 7 and 19
            shin_mass_range        = [[self.mass_low*m[8], self.mass_high*m[8]]]    # 8 and 20
            tarsus_mass_range      = [[self.mass_low*m[9], self.mass_high*m[9]]]    # 9 and 21
            heel_spring_mass_range = [[self.mass_low*m[10], self.mass_high*m[10]]]  # 10 and 22
            fcrank_mass_range      = [[self.mass_low*m[11], self.mass_high*m[11]]]  # 11 and 23
            prod_mass_range        = [[self.mass_low*m[12], self.mass_high*m[12]]]  # 12 and 24
            foot_mass_range        = [[self.mass_low*m[13], self.mass_high*m[13]]]  # 13 and 25

            side_mass = hip_mass_range + achilles_mass_range \
                        + knee_mass_range + knee_spring_mass_range \
                        + shin_mass_range + tarsus_mass_range \
                        + heel_spring_mass_range + fcrank_mass_range \
                        + prod_mass_range + foot_mass_range

            mass_range = [[0, 0]] + pelvis_mass_range + side_mass + side_mass
            mass_noise = [np.random.uniform(a, b) for a, b in mass_range]

            delta = 0.0
            com_noise = [0, 0, 0] + [np.random.uniform(val - delta, val + delta) for val in self.default_ipos[3:]]

            fric_noise = []
            translational = np.random.uniform(self.fric_low, self.fric_high)
            torsional = np.random.uniform(1e-4, 5e-4)
            rolling = np.random.uniform(1e-4, 2e-4)
            for _ in range(int(len(self.default_fric)/3)):
                fric_noise += [translational, torsional, rolling]

            self.sim.set_dof_damping(np.clip(damp_noise, 0, None))
            self.sim.set_body_mass(np.clip(mass_noise, 0, None))
            self.sim.set_body_ipos(com_noise)
            self.sim.set_geom_friction(np.clip(fric_noise, 0, None))
        else:
            self.sim.set_body_mass(self.default_mass)
            self.sim.set_body_ipos(self.default_ipos)
            self.sim.set_dof_damping(self.default_damping)
            self.sim.set_geom_friction(self.default_fric)

        self.sim.set_geom_quat(self.default_quat)    
        self.sim.set_const()
        return self.get_state()

    def get_state(self):
        self.qpos = np.copy(self.sim.qpos())  # dim=35 see cassiemujoco.h for details
        self.qvel = np.copy(self.sim.qvel())  # dim=32
        self.state_buffer.append((self.qpos, self.qvel))
        
        if len(self.state_buffer) > self.buffer_size:
            self.state_buffer.pop(0)
        else:
            while len(self.state_buffer) < self.buffer_size:
                self.state_buffer.append((self.qpos, self.qvel))

        pos = np.array([x[0] for x in self.state_buffer])
        vel = np.array([x[1] for x in self.state_buffer])

        self.ref_pos, self.ref_vel = self.get_kin_next_state()
        command = [self.speed, self.side_speed]
        '''
		Position [1], [2] 				-> Pelvis y, z
				 [3], [4], [5], [6] 	-> Pelvis Orientation qw, qx, qy, qz
				 [7], [8], [9]			-> Left Hip Roll (Motor[0]), Yaw (Motor[1]), Pitch (Motor[2])
				 [14]					-> Left Knee   	(Motor[3])
				 [15]					-> Left Shin   	(Joint[0])
				 [16]					-> Left Tarsus 	(Joint[1])
				 [20]					-> Left Foot   	(Motor[4], Joint[2])
				 [21], [22], [23]		-> Right Hip Roll (Motor[5]), Yaw (Motor[6]), Pitch (Motor[7])
				 [28]					-> Rigt Knee   	(Motor[8])
				 [29]					-> Rigt Shin   	(Joint[3])
				 [30]					-> Rigt Tarsus 	(Joint[4])
				 [34]					-> Rigt Foot   	(Motor[9], Joint[5])
		'''
        pos_index = np.array([2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])
        '''
		Velocity [0], [1], [2] 			-> Pelvis x, y, z
				 [3], [4], [5]		 	-> Pelvis Orientation wx, wy, wz
				 [6], [7], [8]			-> Left Hip Roll (Motor[0]), Yaw (Motor[1]), Pitch (Motor[2])
				 [12]					-> Left Knee   	(Motor[3])
				 [13]					-> Left Shin   	(Joint[0])
				 [14]					-> Left Tarsus 	(Joint[1])
				 [18]					-> Left Foot   	(Motor[4], Joint[2])
				 [19], [20], [21]		-> Right Hip Roll (Motor[5]), Yaw (Motor[6]), Pitch (Motor[7])
				 [25]					-> Rigt Knee   	(Motor[8])
				 [26]					-> Rigt Shin   	(Joint[3])
				 [27]					-> Rigt Tarsus 	(Joint[4])
				 [31]					-> Rigt Foot   	(Motor[9], Joint[5])
		'''
        vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])
        # next todo: x,y,z in state -> delta xyz to target + target velo
        return np.concatenate([pos[:,pos_index].reshape(-1), vel[:,vel_index].reshape(-1), 
                                                            [np.sin(self.phase/28*2*np.pi),np.cos(self.phase/28*2*np.pi)],
                                                            # self.ref_pos[pos_index], self.ref_vel[vel_index]
                                                            command
                                                            ])

    def compute_reward(self, action):
        ref_penalty = 0
        custom_footheight = np.array(self.custom_footheight())
        real_footheight = np.array([self.foot_pos[2],self.foot_pos[5]])
        ref_penalty = np.sum(np.square(custom_footheight - real_footheight))
        ref_penalty = ref_penalty/0.0025

        orientation_penalty = (self.qpos[4])**2+(self.qpos[5])**2+(self.qpos[6])**2
        orientation_penalty = orientation_penalty/0.1
        
        vel_penalty = (self.speed - self.qvel[0])**2 + (self.side_speed - self.qvel[1])**2 + (self.qvel[2])**2
        vel_penalty = vel_penalty/max(0.5*(self.speed*self.speed+self.side_speed*self.side_speed),0.01)
        
        spring_penalty = (self.sim.qpos()[15])**2+(self.sim.qpos()[29])**2
        spring_penalty *= 1000
        
        rew_ref = 0.5*np.exp(-ref_penalty)
        rew_spring = 0.1*np.exp(-spring_penalty)
        rew_ori = 0.125*np.exp(-orientation_penalty)
        rew_vel = 0.375*np.exp(-vel_penalty) #
        rew_termin = -10 * self.termination
        
        R_star = 1
        Rp = (0.75 * np.exp(-vel_penalty) + 0.25 * np.exp(-orientation_penalty))/ R_star
        Ri = np.exp(-ref_penalty) / R_star
        Ri = (Ri-0.4)/(1.0-0.4)
        
        omega = 0.5 

        reward = (1 - omega) * Ri + omega * Rp + rew_spring + rew_termin

        self.rew_ref += rew_ref
        self.rew_spring += rew_spring
        self.rew_ori += rew_ori
        self.rew_vel += rew_vel
        self.rew_termin += rew_termin
        self.reward += reward
        self.omega += omega

        return reward

    def render(self):        
        return self.vis.draw(self.sim)

    def get_kin_state(self):
        pose = np.array([0]*3)        
        vel = np.array([0]*3)   
        pose[0] = self.speed  * (self.counter * 28 + self.phase) * (self.simrate / 2000)
        pose[1] = self.side_speed  * (self.counter * 28 + self.phase) * (self.simrate / 2000)
        pose[2] = 1.03 # 
        vel[0] = self.speed
        return pose, vel

    def get_kin_next_state(self):   
        phase = self.phase + 1
        counter = self.counter
        if phase >= 28:
            phase = 0
            counter += 1
        pose = np.array([0]*3)        
        vel = np.array([0]*3)        
        pose[0] = self.speed  * (counter * 28 + phase) * (self.simrate / 2000)
        pose[1] = self.side_speed  * (counter * 28 + phase) * (self.simrate / 2000)
        pose[2] = 1.03  # 
        vel[0] = self.speed
        return pose, vel


