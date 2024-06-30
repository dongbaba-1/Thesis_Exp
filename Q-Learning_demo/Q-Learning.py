import airsim
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar

# Q-Learning parameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate


# Discretize state space
def discretize_state(x, target_x, bins=30):
    return np.digitize([x, abs(target_x - x)], bins=np.linspace(0, 30, bins))


# Initialize Q-table
state_space_size = 31  # Number of bins for discretization
action_space_size = 2  # Two actions: forward, backward
Q_table = np.zeros((state_space_size, state_space_size, action_space_size))

# Action mapping
actions = {
    0: 'forward',
    1: 'backward'
}


# Reward function
def compute_reward(current_x, target_x, next_x):
    if abs(next_x - target_x) < abs(current_x - target_x):
        return 1
    elif abs(next_x - target_x) <= 0:
        return 10
    else:
        return -1


# AirSim client setup
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Training loop
target_x = 30  # Target x position
num_episodes = 10  # Number of episodes to train for
episodes_per_progress = 1  # Number of episodes per progress bar segment
num_progress_bars = num_episodes // episodes_per_progress  # Number of progress bars

# To store the rewards
rewards = []


for progress in range(num_progress_bars):
    print("Progress ", progress + 1, "/", num_progress_bars)
    for episode in tqdm(range(episodes_per_progress), desc="Progress %d" % progress):
        client.takeoffAsync().join()

        # Reset position
        client.moveToPositionAsync(0, 0, -10, 5).join()
        state = client.getMultirotorState().kinematics_estimated.position
        current_x = state.x_val

        done = False
        total_reward = 0  # To accumulate the rewards for this episode

        while not done:
            state = discretize_state(current_x, target_x)
            # print("State: ", state)

            # Choose action
            if np.random.rand() < epsilon:
                action = np.random.choice(action_space_size)
            else:
                action = np.argmax(Q_table[state[0], state[1]])

            # Execute action
            if actions[action] == 'forward':
                client.moveByVelocityAsync(1, 0, 0, 3).join()
            else:
                client.moveByVelocityAsync(-1, 0, 0, 3).join()

            time.sleep(1)

            # Get new state
            new_state = client.getMultirotorState().kinematics_estimated.position
            next_x = new_state.x_val
            new_state_discrete = discretize_state(next_x, target_x)

            # Compute reward
            reward = compute_reward(current_x, target_x, next_x)
            total_reward += reward  # Accumulate reward for this episode

            # Update Q-table
            Q_table[state[0], state[1], action] = Q_table[state[0], state[1], action] + \
                                                  alpha * (reward + gamma * np.max(
                Q_table[new_state_discrete[0], new_state_discrete[1]]) - Q_table[state[0], state[1], action])

            current_x = next_x

            if abs(current_x - target_x) < 1:  # Threshold for reaching target
                done = True
                print("Episode {episode + 1}: Reached the target")

        rewards.append(total_reward)  # Store total reward for this episode
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)

# Plotting the results
plt.plot(range(num_episodes), rewards)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Total Reward vs Episodes')
plt.show()