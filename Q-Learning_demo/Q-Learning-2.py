import airsim
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar

# Q-Learning parameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
initial_epsilon = 0.5  # Initial exploration rate
min_epsilon = 0.01  # Minimum exploration rate
decay_rate = 0.9  # Rate at which exploration rate decays
epsilon = initial_epsilon  # Initialize exploration rate


# Discretize state space
# def discretize_state(x, target_x, bins=30):
#     return np.digitize([x, abs(target_x - x)], bins=np.linspace(0, 30, bins))
# 　改成只要目标的距离作为状态
def discretize_state(target_x_distance, bins=40):
    return np.digitize(target_x_distance, bins=np.linspace(0, 40, bins))


# Initialize Q-table
state_space_size = 41  # Number of bins for discretization
action_space_size = 2  # Two actions: forward, backward
Q_table = np.zeros((state_space_size, action_space_size))

# Action mapping
actions = {
    0: 'forward',
    1: 'backward'
}


# Reward function
def compute_reward(current_x, target_x, next_x):
    if abs(next_x - target_x) < abs(current_x - target_x):
        return 1
    elif next_x >= target_x:
        return 10
    else:
        return -3


# AirSim client setup
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Training loop
target_x = 20  # Target x position
num_episodes = 125   # Number of episodes to train for
episodes_per_progress = 1  # Number of episodes per progress bar segment
num_progress_bars = num_episodes // episodes_per_progress  # Number of progress bars

# To store the rewards
rewards = []

try:
    for progress in range(num_progress_bars):
        print("Progress ", progress + 1, "/", num_progress_bars)
        for episode in tqdm(range(episodes_per_progress), desc="Progress %d" % progress):
            client.takeoffAsync().join()

            # Reset position
            client.moveToPositionAsync(0, 0, -10, 5).join()
            time.sleep(1)
            state = client.getMultirotorState().kinematics_estimated.position
            current_x = state.x_val

            done = False
            total_reward = 0  # To accumulate the rewards for this episode

            while not done:

                state = discretize_state(target_x - current_x)
                z_position = client.getMultirotorState().kinematics_estimated.position.z_val
                # print("State: ", state)
                if z_position >= -1 or z_position <= -20:
                    client.moveToZAsync(-10, 1).join()

                # Choose action
                if np.random.rand() < epsilon:
                    action = np.random.choice(action_space_size)
                else:
                    action = np.argmax(Q_table[state])

                # Execute action
                if actions[action] == 'forward':
                    client.moveByVelocityAsync(1, 0, 0, 2).join()
                else:
                    client.moveByVelocityAsync(-1, 0, 0, 2).join()

                # time.sleep(1)

                # Get new state
                new_state = client.getMultirotorState().kinematics_estimated.position
                next_x = new_state.x_val
                new_state_discrete = discretize_state(target_x - next_x)

                # Compute reward
                reward = compute_reward(current_x, target_x, next_x)
                total_reward += reward  # Accumulate reward for this episode

                # Update Q-table
                Q_table[state, action] = Q_table[state, action] + \
                                         alpha * (reward + gamma * np.max(
                    Q_table[new_state_discrete]) - Q_table[state, action])

                print("Q-table after action:", actions[action])
                print("Q_table[", state, "]:", Q_table[state])
                current_x = next_x

                if current_x >= target_x:  # Threshold for reaching target
                    done = True
                    print("Episode %d: Reached the target" % (episode + 1))

            rewards.append(total_reward)  # Store total reward for this episode
            print("Total reward for this episode:", total_reward)
            # updata epsilon
            epsilon = max(min_epsilon, epsilon * decay_rate)

except KeyboardInterrupt:
    print("Interrupted")

finally:

    # save Q-table
    np.save("Q_table.npy", Q_table)
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)

    # Plotting the results
    plt.plot(range(num_episodes), rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Total Reward vs Episodes')
    plt.savefig("training_rewards.png")
    plt.show()

