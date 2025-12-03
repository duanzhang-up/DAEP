import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from velodyne_env import GazeboEnv


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


# TD3 network
class TD3(object):
    def __init__(self, state_dim, action_dim):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        # Function to load network parameters
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )


# Set the parameters for the implementation
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
device = torch.device("cpu")
seed = 0  # Random seed number
max_ep = 500  # maximum number of steps per episode
file_name = "TD3_velodyne(DAEP-m5k5)"  # name of the file to load the policy from


# Create the testing environment
environment_dim = 20
robot_dim = 4
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
print(
    "Environment initialization completed. Please use 2D Nav Goal in RViz or publish goal point to /move_base_simple/goal."
)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim
action_dim = 2

# Create the network
network = TD3(state_dim, action_dim)
try:
    network.load(file_name, "./pytorch_models")
except:
    raise ValueError("Could not load the stored model parameters")

done = False
episode_timesteps = 0
print("Waiting for first manual goal...")
state, last_distance, last_x, last_y = env.reset()
last_action = [0, 0]
start_time = time.time()
num = 0 # Total attempts
goal_num = 0 # Successful goals
distance_num = 0 # Total distance traveled by robot

# Begin the testing loop
try:
    while True:
        action = network.get_action(np.array(state))

        # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
        a_in = [(action[0] + 1) / 2, action[1]]
        next_state, reward, done, target, distance, done_target, collision, self_distance = env.step(a_in, last_distance, last_action, last_x, last_y)
        last_distance = distance
        last_action = a_in
        done = 1 if episode_timesteps + 1 == max_ep else int(done)
        distance_num += self_distance
        last_x, last_y = env.odom_x, env.odom_y

        # On termination of episode
        if done:
            num += 1
            print("Run ended. Please re-set goal point and wait for environment reset...")
            state, last_distance, last_x, last_y = env.reset()
            done = False
            last_action = [0, 0]
            episode_timesteps = 0
            end_time = time.time()
            elapsed_time = end_time - start_time
            if done_target:
                goal_num += 1
                result = "Success"
            elif collision:
                result = "Failed"
            else:
                result = "Failed"
            percentage_goal = goal_num / num * 100
            print(f"Run {num} took {elapsed_time:.2f} seconds, traveled {distance_num:.2f} meters, result: {result}, success rate: {percentage_goal:.2f}%")
            distance_num = 0
            start_time = time.time()
        else:
            state = next_state
            episode_timesteps += 1
except KeyboardInterrupt:
    print("Manual interrupt detected. Testing process stopped.")
