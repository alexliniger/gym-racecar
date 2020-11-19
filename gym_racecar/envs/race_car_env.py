import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from numpy import linalg as LA
import math
import os


class RaceCarEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human', 'rgb_array'],'video.frames_per_second': 25}

    def __init__(self,track_name="TrackCenter",episode_len = 400,model_rand = False,action_obs = True):
        super(RaceCarEnv, self).__init__()
        self.n_state = 6
        self.n_action = 2
        self.dt = 0.02
        # observations
        self.action_obs = action_obs
        if self.action_obs:
            self.n_obs = 9
            self.obs_scale = np.array([1, 1, 0.2, 1, 4, 1, 10, 1, 0.35])
        else:
            self.n_obs = 7
            self.obs_scale = np.array([1, 1, 0.2, 1, 4, 1, 10])
        #
        self.sim_steps = 0
        self.standing_steps = 0
        # state
        self.state = np.zeros(self.n_state)
        self.action = np.zeros(self.n_action)
        #
        self.model_rand = model_rand
        
        self.action_space = spaces.Box(
            low=-1,
            high=1, 
            shape=(self.n_action,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.n_obs,),
            dtype=np.float32
        )
        
        self.track = Track(track_name)
        # bounds for random starting point
        self.s_high = np.array([self.track.L, 0.2, 0.2,2, 0.1, 2])
        self.s_low =  np.array([           0,-0.2,-0.2,0,-0.1,-2])



        # Model Parameters
        self.Cm1 = 0.287
        self.Cm2 = 0.054527
        self.Cr0 = 0.051891
        self.Cr2 = 0.000348

        self.B_r = 3.3852
        self.C_r = 1.2691
        self.D_r = 0.1737

        self.B_f = 2.579
        self.C_f = 1.2
        self.D_f = 0.192

        self.mass = 0.041
        self.I_z = 27.8e-6
        self.l_f = 0.029
        self.l_r = 0.033

        self.L = 0.12
        self.W = 0.06

        self.prog_scale = 1
        self.D_cost = 0.01
        self.delta_cost = 0.1
        self.collision_reward = -1
        self.tire_con = 0.001
        self.beta_cost = 0.000

        self.episode_len = episode_len
        self.max_standing = 5

        self.viewer = None
        self.min_position = -2
        self.max_position =  2

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, a):
        #unsqueeze actions
        a = self.denormalizeAction(a)
        s = self.state

        # simulate the race car
        v_x = s[3]
        # if the velocity is large enough use dynamic bicycle model [2]
        if v_x >= 0.2:
            new_s = self.RK4Dyn(s,a)
            self.standing_steps = 0
        #if the velocity is low use kinematic bicycle model
        else:
            # if the velocity is very low, only allow action that move the car forward
            # if the action is too low the car just stands still
            if v_x <= 0.1 and a[0] <= (self.Cr0+self.Cr2*v_x**2)/(self.Cm1 - self.Cm2*v_x):
                new_s = s
                self.standing_steps += 1
            else:
                new_s = self.RK4Kin(s,a)
                new_v_x = new_s[3]
                # approximate lateral velocity and yaw rate based using kinematic conditions
                new_s[4] = new_v_x*np.tan(a[1])*self.l_r/(self.l_r + self.l_f)
                new_s[5] = new_v_x*np.tan(a[1])/(self.l_r + self.l_f)
                self.standing_steps = 0

        self.sim_steps += 1

        # reward is a progress reward, regularized with a input rate cost
        con_front, con_rear = self.tireConstraints(s,a)

        reward =  self.prog_scale*(new_s[0] - s[0]) \
                - self.D_cost*(self.action[0]-a[0])**2  \
                - self.delta_cost*(self.action[1]-a[1])**2 \
                - self.tire_con*(con_front + con_rear) \
                - self.beta_cost*np.arctan(s[4]/np.maximum(s[3],0.5))**2

        # check if the car is violating the track boundaries 
        con_active = self.constraintActive(new_s)

        # if the track constraints are activated the episode is terminated and the reward is over written
        if con_active:
            reward = self.collision_reward

        # store the new state and action
        self.action = a
        new_s[0] = new_s[0]%self.track.L
        self.state = new_s
        if con_active or \
            self.sim_steps >= self.episode_len or \
            self.standing_steps >= self.max_standing:

            terminate = True
        else:
            terminate = False


        return self._get_obs(), reward, terminate, {}

    def denormalizeAction(self, a):
        a = np.clip(a, -1, 1)
        a[0] = a[0] * 0.6 + 0.4
        a[1] = a[1] * 0.35
        return a

    def constraintActive(self, s):

        # check heading dependent track constraints see [3] for details
        p = s[0]
        d = s[1]
        mu = s[2]
        [d_upper,d_lower,angle_upper,angle_lower] = self.track.getLocalBounds(p)
        track_heading = self.track.getTrackHeading(p)
        angle_border_upper = track_heading - angle_upper
        angle_border_lower = track_heading - angle_lower


        corner_width_upper = self.L/2 * np.sin(np.fabs(mu-angle_border_upper)) + \
                             self.W/2 * np.cos(mu - angle_border_upper)
        corner_width_lower = self.L/2 * np.sin(np.fabs(mu - angle_border_lower)) + \
                             self.W/2 * np.cos(mu - angle_border_lower)

        con_upper =  (d - d_upper + corner_width_upper)
        con_lower = (-d - (-d_lower) + corner_width_lower)

        if con_lower >= 0 or con_upper >= 0:
            return True
        else:
            return False

    def tireConstraints(self,s,a):
        alpha_f, _ = self.slipAngles(s,a)
        F_rx, F_ry, _ = self.forceModel(s, a)

        con_rear = (0.8*F_rx)**2 + F_ry**2 - (0.99*self.D_r)**2
        con_front = np.fabs(alpha_f) - 0.8

        return np.maximum(con_front,0), np.sqrt(np.maximum(con_rear,0))

    # RK4 integrator
    def RK4(self, s, a, dx):

        k1 = dx(s, a)
        k2 = dx(s + self.dt / 2. * k1, a)
        k3 = dx(s + self.dt / 2. * k2, a)
        k4 = dx(s + self.dt * k3     , a)

        s_next = s + self.dt * (k1 / 6. + k2 / 3. + k3 / 3. + k4 / 6.)

        return s_next

    def RK4Dyn(self, s, a):

        if self.model_rand:
            e = np.random.rand(self.n_state)
            e[:3] = 1
            e[3] = 1 + (e[3] - 0.5) * 2 * 1.5
            e[4] = 1 + (e[4] - 0.5) * 2 * 2.5
            e[5] = 1 + (e[5] - 0.5) * 2 * 2
        else:
            e = np.ones(self.n_state)

        k1 = self.dxCurv(s, a)
        k2 = self.dxCurv(s + self.dt / 2. * k1, a)
        k3 = self.dxCurv(s + self.dt / 2. * k2, a)
        k4 = self.dxCurv(s + self.dt * k3     , a)

        s_next = s + e * self.dt * (k1 / 6. + k2 / 3. + k3 / 3. + k4 / 6.)

        return s_next

    def RK4Kin(self, s, a):
        if self.model_rand:
            e = np.random.rand(self.n_state)
            e[:3] = 1
            e[3] = 1 + (e[3] - 0.5) * 2 * 1.5
            e[4] = 1 + (e[4] - 0.5) * 2 * 2.5
            e[5] = 1 + (e[5] - 0.5) * 2 * 2
        else:
            e = np.ones(self.n_state)

        k1 = self.dxCurvKin(s, a)
        k2 = self.dxCurvKin(s + self.dt / 2. * k1, a)
        k3 = self.dxCurvKin(s + self.dt / 2. * k2, a)
        k4 = self.dxCurvKin(s + self.dt * k3     , a)

        s_next = s + e * self.dt * (k1 / 6. + k2 / 3. + k3 / 3. + k4 / 6.)

        return s_next

    # dynamic bicycle model in curvilinear coordinates
    def dxCurv(self, s, a):

        f = np.empty(self.n_state)

        p = s[0]
        d = s[1]
        mu = s[2]
        v_x = s[3]
        v_y = s[4]
        r = s[5]

        delta = a[1]

        [F_rx, F_ry, F_fy] = self.forceModel(s, a)

        kappa = self.track.getCurvature(p)

        f[0] = (v_x * np.cos(mu) - v_y * np.sin(mu))/(1.0 - kappa*d)
        f[1] =  v_x * np.sin(mu) + v_y * np.cos(mu)
        f[2] = r - kappa*((v_x * np.cos(mu) - v_y * np.sin(mu))/(1.0 - kappa*d))
        f[3] = 1 / self.mass * (F_rx - F_fy * np.sin(delta) + self.mass * v_y * r)
        f[4] = 1 / self.mass * (F_ry + F_fy * np.cos(delta) - self.mass * v_x * r)
        f[5] = 1 / self.I_z * (F_fy * self.l_f * np.cos(delta) - F_ry * self.l_r)

        return f

    # kinematic bicycle model in curvilinear coordinates
    def dxCurvKin(self, s, a):

        f = np.empty(self.n_state)

        p = s[0]
        d = s[1]
        mu = s[2]
        v = s[3]

        D = a[0]
        delta = a[1]


        kappa = self.track.getCurvature(p)
        beta = np.arctan(np.tan(delta)*self.l_r/(self.l_r+self.l_f))

        v_x = v*np.cos(beta)
        v_y = v*np.sin(beta)
        r = v/self.l_r * np.sin(beta)

        F_rx = self.Cm1 * D - self.Cm2*v_x*D - self.Cr2*v_x**2 - self.Cr0

        f[0] = (v_x * np.cos(mu) - v_y * np.sin(mu))/(1.0 - kappa*d)
        f[1] =  v_x * np.sin(mu) + v_y * np.cos(mu)
        f[2] = r - kappa*((v_x * np.cos(mu) - v_y * np.sin(mu))/(1.0 - kappa*d))
        f[3] = 1 / self.mass * F_rx
        
        return f

    def slipAngles(self, s, a):
        v_x = s[3]
        v_y = s[4]
        r = s[5]

        D = a[0]
        delta = a[1]

        alpha_f = -np.arctan((self.l_f * r + v_y) / v_x) + delta
        alpha_r =  np.arctan((self.l_r * r - v_y) / v_x)

        return alpha_f, alpha_r

    # tire and motor model
    def forceModel(self, s, a):
        v_x = s[3]
        v_y = s[4]
        r = s[5]

        D = a[0]
        delta = a[1]

        alpha_f = -np.arctan((self.l_f * r + v_y) / v_x) + delta
        alpha_r =  np.arctan((self.l_r * r - v_y) / v_x)

        F_rx = self.Cm1 * D - self.Cm2*v_x*D - self.Cr2*v_x**2 - self.Cr0
        F_ry = self.D_r * np.sin(self.C_r * np.arctan(self.B_r * alpha_r))
        F_fy = self.D_f * np.sin(self.C_f * np.arctan(self.B_f * alpha_f))

        return F_rx, F_ry, F_fy

    # reset state to random place around the track
    def reset(self):
        self.sim_steps = 0
        self.standing_steps = 0
        for i in range(10):
            self.state = self.np_random.uniform(low=self.s_low, high=self.s_high)
            if not self.constraintActive(self.state):
                break
        self.action = np.zeros(self.n_action)

        return self._get_obs()

    # retunred -1/1 observation
    def _get_obs(self):
        s = self.state
        p = s[0]
        p_angle = p/self.track.L*2*np.pi
        d = s[1]
        mu = s[2]
        v_x = s[3]
        v_y = s[4]
        r = s[5]
        a = self.action
        D = a[0]
        delta = a[1]

        if self.action_obs:
            obs = np.array([np.cos(p_angle), np.sin(p_angle), d, mu, v_x, v_y, r, D, delta])/self.obs_scale
        else:
            obs = np.array([np.cos(p_angle), np.sin(p_angle), d, mu, v_x, v_y, r])/self.obs_scale

        return np.clip(obs,-1,1)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 600


        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carlength= self.L*scale
        carwidth = self.W*scale

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            xy_center = list(zip((self.track.X-self.min_position)*scale, (self.track.Y-self.min_position)*scale))

            self.track_center = rendering.make_polyline(xy_center)
            self.track_center.set_linewidth(1)
            self.viewer.add_geom(self.track_center)

            xy_o = list(zip((self.track.X_boarder_outer-self.min_position)*scale, (self.track.Y_boarder_outer-self.min_position)*scale))

            self.track_outer = rendering.make_polyline(xy_o)
            self.track_outer.set_linewidth(4)
            self.viewer.add_geom(self.track_outer)

            xy_i = list(zip((self.track.X_boarder_inner-self.min_position)*scale, (self.track.Y_boarder_inner-self.min_position)*scale))

            self.track_inner = rendering.make_polyline(xy_i)
            self.track_inner.set_linewidth(4)
            self.viewer.add_geom(self.track_inner)

            clearance = 0

            l, r, t, b = -carlength / 2, carlength / 2 ,-carwidth / 2, carwidth / 2
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, 0)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            car.set_color(1, 0, 0)
            self.viewer.add_geom(car)
            

        pos = self.track.fromLocaltoGlobal(self.state)
        self.cartrans.set_translation(
            (pos[0]-self.min_position) * scale, (pos[1]-self.min_position) * scale
        )
        self.cartrans.set_rotation(pos[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class Track:

    def __init__(self,track):
        dirname = os.path.dirname(__file__)
        if track == "TrackCenter":
            self.path = os.path.join(dirname, track)
            self.X = np.loadtxt(self.path + "/x_center.txt")[:, 0]
            self.Y = np.loadtxt(self.path + "/x_center.txt")[:, 1]
            self.s = np.loadtxt(self.path + "/s_center.txt")
            self.phi = np.loadtxt(self.path + "/phi_center.txt")
            self.kappa = np.loadtxt(self.path + "/kappa_center.txt")
            self.diff_s = np.mean(np.diff(self.s))
            self.N = len(self.s) - 1
            self.L = self.s[-1]

            self.d_upper =  0.18*np.ones_like(self.s)
            self.d_lower = -0.18*np.ones_like(self.s)
            self.border_angle_upper = np.zeros_like(self.s)
            self.border_angle_lower = np.zeros_like(self.s)

            #only for plotting
            self.X_boarder_outer = np.loadtxt(self.path + "/x_outer.txt")[:, 0]
            self.Y_boarder_outer = np.loadtxt(self.path + "/x_outer.txt")[:, 1]
            self.X_boarder_inner = np.loadtxt(self.path + "/x_inner.txt")[:, 0]
            self.Y_boarder_inner = np.loadtxt(self.path + "/x_inner.txt")[:, 1]

        if track == "TrackIdeal":
            self.path = os.path.join(dirname, track)
            self.X = np.loadtxt(self.path + "/x_center.txt")[:, 0]
            self.Y = np.loadtxt(self.path + "/x_center.txt")[:, 1]
            self.s = np.loadtxt(self.path + "/s_center.txt")
            self.phi = np.loadtxt(self.path + "/phi_center.txt")
            self.kappa = np.loadtxt(self.path + "/kappa_center.txt")
            self.diff_s = np.mean(np.diff(self.s))
            self.N = len(self.s) - 1
            self.L = self.s[-1]

            self.d_upper = np.loadtxt(self.path + "/con_inner.txt")
            self.d_lower = np.loadtxt(self.path + "/con_outer.txt")
            self.border_angle_upper = np.loadtxt(self.path + "/con_angle_inner.txt")
            self.border_angle_lower = np.loadtxt(self.path + "/con_angle_outer.txt")

            # only for plotting
            self.X_boarder_outer = np.loadtxt(self.path + "/x_outer.txt")[:, 0]
            self.Y_boarder_outer = np.loadtxt(self.path + "/x_outer.txt")[:, 1]
            self.X_boarder_inner = np.loadtxt(self.path + "/x_inner.txt")[:, 0]
            self.Y_boarder_inner = np.loadtxt(self.path + "/x_inner.txt")[:, 1]



    def posAtIndex(self, i):
        return np.array([self.X[i], self.Y[i]])

    def vecToPoint(self, index, x):
        return np.array([x[0] - self.X[index], x[1] - self.Y[index]])

    def vecTrack(self, index):
        if index >= self.N - 1:
            next_index = 0
        else:
            next_index = index + 1

        return np.array([self.X[next_index] - self.X[index], self.Y[next_index] - self.Y[index]])

    def interpol(self, name, index, rela_proj):
        if index == self.N:
            index = 0
            next_index = 1
        else:
            next_index = index + 1

        if name == "s":
            return self.s[index] + (rela_proj * (self.s[next_index] - self.s[index]))
        if name == "phi":
            return self.phi[index] + (rela_proj * (self.phi[next_index] - self.phi[index]))
        if name == "kappa":
            return self.kappa[index] + (rela_proj * (self.kappa[next_index] - self.kappa[index]))

    def fromStoPos(self, s):

        index = math.floor(s / self.diff_s)
        rela_proj = (s - self.s[index]) / self.diff_s
        pos = [self.X[index], self.Y[index]] + self.vecTrack(index) * rela_proj
        return pos

    def fromStoIndex(self, s):
        if s > self.L:
            s = s - self.L
        elif s < 0:
            s = s + self.L

        s = max(s, 0)
        s = min(s, self.L )

        index = math.floor(s / self.diff_s)
        rela_proj = (s - self.s[index]) / self.diff_s
        return [index, rela_proj]

    def getCurvature(self,s):
        index, rela_proj = self.fromStoIndex(s)
        return self.interpol("kappa",index,rela_proj)

    def getTrackHeading(self,s):
        index, rela_proj = self.fromStoIndex(s)
        return self.interpol("phi",index,rela_proj)

    def fromLocaltoGlobal(self, x_local):
        s = x_local[0]
        d = x_local[1]
        mu = x_local[2]

        [index, rela_proj] = self.fromStoIndex(s)
        pos_center = [self.X[index], self.Y[index]] + self.vecTrack(index) * rela_proj
        phi = self.interpol("phi", index, rela_proj)

        pos_global = pos_center + d * np.array([-np.sin(phi), np.cos(phi)])
        heading = phi + mu
        return [pos_global[0], pos_global[1], heading]

    def wrapMu(self, mu):
        if mu < -np.pi:
            mu = mu + 2 * np.pi
        elif mu > np.pi:
            mu = mu - 2 * np.pi
        return mu

    def compLocalCoordinates(self, x):
        dist = np.zeros(self.N)
        for i in range(self.N):
            dist[i] = LA.norm(self.vecToPoint(i, x))

        min_index = np.argmin(dist)
        min_dist = dist[min_index]

        if min_dist <= 1e-13:
            s = self.s[min_index]
            d = 0
            mu = x[2] - self.phi[min_index]
            kappa = self.kappa[min_index]
            phi = self.phi[min_index]
        else:
            a = self.vecToPoint(min_index, x)
            b = self.vecTrack(min_index)

            cos_theta = (np.dot(a, b) / (LA.norm(a) * LA.norm(b)))

            if cos_theta < 0:
                min_index = min_index - 1
                if min_index < 0:
                    min_index = self.N - 1
                a = self.vecToPoint(min_index, x)
                b = self.vecTrack(min_index)

                cos_theta = (np.dot(a, b) / (LA.norm(a) * LA.norm(b)))

            if cos_theta >= 1:
                cos_theta = 0.99999999

            rela_proj = LA.norm(a) * cos_theta / LA.norm(b)
            rela_proj = max(min(rela_proj, 1), 0)
            theta = np.arccos(cos_theta)

            error_sign = -np.sign(a[0] * b[1] - a[1] * b[0])
            error = error_sign * LA.norm(a) * np.sin(theta)
            error_dist = error_sign * LA.norm(
                self.posAtIndex(min_index) + b * LA.norm(a) * cos_theta / LA.norm(b) - [x[0], x[1]])

            s = self.interpol("s", min_index, rela_proj)
            d = error
            mu = self.wrapMu(x[2] - self.interpol("phi", min_index, rela_proj))
            kappa = self.interpol("kappa", min_index, rela_proj)
            phi = self.interpol("phi", min_index, rela_proj)

        return np.array([s, d, mu, x[3], x[4], x[5], kappa, phi])

    def getLocalBounds(self,s):
        index, rela_proj = self.fromStoIndex(s)
        if index == self.N:
            index = 0
            next_index = 1
        else:
            next_index = index + 1

        d_upper = self.d_upper[index] + \
                  rela_proj * (self.d_upper[next_index] - self.d_upper[index])
        d_lower = self.d_lower[index] +\
                  rela_proj * (self.d_lower[next_index] - self.d_lower[index])

        angle_upper = self.border_angle_upper[index] + \
                      rela_proj * (self.border_angle_upper[next_index] - self.border_angle_upper[index])
        angle_lower = self.border_angle_lower[index] + \
                      rela_proj * (self.border_angle_lower[next_index] - self.border_angle_lower[index])

        return d_upper, d_lower,angle_upper,angle_lower