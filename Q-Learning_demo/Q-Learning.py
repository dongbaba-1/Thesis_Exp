import airsim
import numpy as np
import time

# Q-Learning parameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate


# Discretize state space
def discretize_state(x, target_x, bins=50):
    return np.digitize([x, abs(target_x - x)], bins=np.linspace(0, 50, bins))


# Initialize Q-table
state_space_size = 51  # Number of bins for discretization
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
    elif abs(next_x - target_x) == 0:
        return 10
    else:
        return -1


# AirSim client setup
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Training loop
target_x = 50  # Target x position
num_episodes = 1000

for episode in range(num_episodes):
    client.takeoffAsync().join()

    # Reset position
    client.moveToPositionAsync(0, 0, -10, 5).join()

    state = client.getMultirotorState().kinematics_estimated.position
    current_x = state.x_val

    done = False

    while not done:
        state = discretize_state(current_x, target_x)
        print("State: ", state)

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

        # Update Q-table
        Q_table[state[0], state[1], action] = Q_table[state[0], state[1], action] + \
                                              alpha * (reward + gamma * np.max(
            Q_table[new_state_discrete[0], new_state_discrete[1]]) - Q_table[state[0], state[1], action])

        current_x = next_x

        if abs(current_x - target_x) < 1:  # Threshold for reaching target
            done = True
            print("Episode {episode + 1}: Reached the target")

client.armDisarm(False)
client.enableApiControl(False)
