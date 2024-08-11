from stable_baselines3 import DQN
import gym
from clinic_env import ClinicEnv

def main():
    env = ClinicEnv()
    
    model = DQN(
        policy='MlpPolicy',
        env=env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=64,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        max_grad_norm=10,
    )

    model.learn(total_timesteps=50000)
    model.save("clinic_dqn_model")

if __name__ == "__main__":
    main()
