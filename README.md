﻿# Clinic Resource Allocation using DQN

## Project Overview

This project uses a reinforcement learning agent that uses Deep Q-Learning (DQN) to optimize resource allocation in a simulated clinic environment. The goal of the agent is designed to efficiently manage patient appointments by assigning doctors to patients.

## Environment Description

- **Clinic Layout:**
  - 5 doctors
  - 3 patients
  - 3 rooms

- **Agent:**
  - Role: Clinic Manager

- **Actions:**
  - Assign Patient 1 to Doctor 1
  - Assign Patient 2 to Doctor 2
  - Assign Patient 3 to Doctor 3
  - Delay Assignment
  - Prioritize an urgent patient

- **Observation Space:**
  - 9-dimensional vector representing current time, doctor availability, and room usage status.

- **Rewards:**
  - Efficient patient assignment: Positive reward (e.g., +10)
  - Idle doctors or unassigned patients: Negative reward (e.g., -5)
  - Each time step: Small penalty (e.g., -0.1) to encourage efficiency.

- **Termination Conditions:**
  - All patients treated successfully (success)
  - Maximum time steps reached (100 steps) without treating all patients (failure)
  - All doctors occupied and unable to assign new patients (failure)

## Usage

### To Visualize:

[Demo](https://drive.google.com/file/d/1NADlgSNauQVjzNHKeewdjH03S0Q2pBXL/view?usp=sharing)

```bash
python play.py

