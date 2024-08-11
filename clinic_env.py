import gym
from gym import spaces
import numpy as np

class ClinicEnv(gym.Env):
    def __init__(self):
        super(ClinicEnv, self).__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        self.time = 0
        self.doctors = [True] * 5  # 5 doctors
        self.patients = [{'assigned_doctor': None} for _ in range(3)]  # 3 patients
        self.rooms = [True] * 3  # 3 rooms

    def reset(self):
        self.time = 0
        self.doctors = [True] * 5
        self.patients = [{'assigned_doctor': None} for _ in range(3)]
        self.rooms = [True] * 3
        return self._get_observation()

    def step(self, action):
        done = False
        reward = 0
        
        if self.action_space.contains(action):
            available_patients = [i for i, p in enumerate(self.patients) if p['assigned_doctor'] is None]
            if available_patients:
                patient_idx = available_patients[0]
                if self.doctors[action]:
                    self.doctors[action] = False
                    self.patients[patient_idx]['assigned_doctor'] = action
                    reward += 10
                else:
                    reward -= 5
        
        if self.time > 100:
            done = True

        self.time += 1
        obs = self._get_observation()
        return obs, reward, done, {}

    def _get_observation(self):
        return np.array([self.time] + self.doctors + self.rooms, dtype=np.float32)

    def render(self, mode='human'):
        pass
