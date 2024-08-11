import pygame
import numpy as np
from stable_baselines3 import DQN
from clinic_env import ClinicEnv

pygame.init()

WINDOW_SIZE = (800, 600)
BACKGROUND_COLOR = (255, 255, 255)
DOCTOR_COLOR = (0, 255, 0)
DOCTOR_BUSY_COLOR = (200, 200, 200)
PATIENT_COLOR = (255, 0, 0)
ROOM_COLOR = (0, 0, 255)
ROOM_OCCUPIED_COLOR = (200, 200, 200)
TEXT_COLOR = (0, 0, 0)

screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption('Clinic Environment Visualization')

model = DQN.load("clinic_dqn_model")

def draw_environment(env, screen):
    screen.fill(BACKGROUND_COLOR)

    # Draw doctors
    for i, available in enumerate(env.doctors):
        color = DOCTOR_COLOR if available else DOCTOR_BUSY_COLOR
        pygame.draw.rect(screen, color, (50 + i * 100, 50, 80, 30))
        pygame.draw.rect(screen, (0, 0, 0), (50 + i * 100, 50, 80, 30), 2)
        font = pygame.font.Font(None, 24)
        text = font.render(f'Doctor {i+1}', True, TEXT_COLOR)
        screen.blit(text, (50 + i * 100 + 10, 55))

    # Draw rooms
    for i, available in enumerate(env.rooms):
        color = ROOM_COLOR if available else ROOM_OCCUPIED_COLOR
        pygame.draw.rect(screen, color, (50 + i * 100, 100, 80, 30))
        pygame.draw.rect(screen, (0, 0, 0), (50 + i * 100, 100, 80, 30), 2)
        font = pygame.font.Font(None, 24)
        text = font.render(f'Room {i+1}', True, TEXT_COLOR)
        screen.blit(text, (50 + i * 100 + 10, 105))

    # Draw patients
    patient_positions = [(WINDOW_SIZE[0] // 2, 200 + i * 30) for i in range(len(env.patients))]
    for i, (patient_pos, patient) in enumerate(zip(patient_positions, env.patients)):
        pygame.draw.circle(screen, PATIENT_COLOR, patient_pos, 10)
        font = pygame.font.Font(None, 24)
        text = font.render(f'Patient {i+1}', True, TEXT_COLOR)
        screen.blit(text, (patient_pos[0] + 15, patient_pos[1] - 10))

        if patient['assigned_doctor'] is not None:
            doctor_pos = (50 + patient['assigned_doctor'] * 100 + 40, 65)
            pygame.draw.line(screen, (0, 0, 0), patient_pos, doctor_pos, 2)

    pygame.display.flip()

def main():
    env = ClinicEnv()
    obs = env.reset()
    done = False
    clock = pygame.time.Clock()

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        
        draw_environment(env, screen)
        
        font = pygame.font.Font(None, 36)
        text = font.render(f'Reward: {reward}', True, TEXT_COLOR)
        screen.blit(text, (10, 10))
        
        pygame.display.flip()
        clock.tick(1)

        print(f"Action taken: {action}")
        print(f"Current Observation: {obs}")
        print(f"Reward received: {reward}")
        print(f"Done: {done}")
        print(f"Info: {info}")

    pygame.quit()

if __name__ == "__main__":
    main()
