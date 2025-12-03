import os
import time
import math
import statistics

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.writer import SummaryWriter

from replay_buffer import ReplayBuffer
from velodyne_env import GazeboEnv

def evaluate(timestep, network, epoch, eval_episodes=10):

    avg_reward = 0.0
    col = 0 # Counter
    for _ in range(eval_episodes):
        count = 0
        state, last_distance, _, _ = env.reset() # Reset environment and get initial state
        last_action = [0, 0]
        done = False # Initialize a boolean variable done to indicate whether current evaluation set is completed
        while not done and count < 501:
            action = network.get_action(np.array(state)) # Use policy network to get action to be executed in current state
            a_in = [(action[0] + 1) / 2, action[1]] # Preprocess network output action, here convert first action value from [-1, 1] range to [0, 1] range
            state, reward, done, _, distance, _, _, _ = env.step(a_in, last_distance, last_action, 0, 0)
            last_distance = distance
            last_action = a_in 
            avg_reward += reward
            count += 1
            if reward < -90:
                col += 1
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    writer.add_scalar("Average Reward per Episode", avg_reward, timestep)
    print("..............................................")
    print(
        "Average Reward over %i Evaluation Episodes, Epoch %i: %f, %f"
        % (eval_episodes, epoch, avg_reward, avg_col)
    )
    print("..............................................")
    return avg_reward


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        # self.layer_1 = nn.Linear(state_dim, 384)
        # self.layer_2 = nn.Linear(384, action_dim)
        # self.tanh = nn.Tanh()
        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        # s = F.relu(self.layer_1(s))
        # s = F.relu(self.layer_2(s))
        # s = F.dropout(s)
        # a = self.tanh(s)
        # return a
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2_s = nn.Linear(800, 600)
        self.layer_2_a = nn.Linear(action_dim, 600)
        self.layer_3 = nn.Linear(600, 1)

        self.layer_4 = nn.Linear(state_dim, 800)
        self.layer_5_s = nn.Linear(800, 600)
        self.layer_5_a = nn.Linear(action_dim, 600)
        self.layer_6 = nn.Linear(600, 1)
        # self.layer_1 = nn.Linear(state_dim + action_dim, 384)
        # self.layer_2 = nn.Linear(384, 64)
        # self.layer_3 = nn.Linear(64, 1)
    def forward(self, s, a):
        s1 = F.relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(a)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        s2 = F.relu(self.layer_4(s))
        self.layer_5_s(s2)
        self.layer_5_a(a)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(a, self.layer_5_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)

        return q1, q2


# TD3 network
class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        # Initialize the Critic networks
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action
        self.writer = SummaryWriter()
        self.iter_count = 0

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    # training cycle
    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=100,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,  # discount=0.99
        noise_clip=0.5,
        policy_freq=2,
    ):
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        num_reward = 0 # Average reward
        for it in range(iterations):
            # sample a batch from the replay buffer
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            # Obtain the estimated action from the next state by using the actor-target
            next_action = self.actor_target(next_state)
            num_reward = discount * reward + num_reward
            av_reward = num_reward / it
            # Add noise to the action
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Calculate the Q values from the critic-target network for the next state-action pair
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Select the minimal Q value from the 2 calculated values
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += torch.mean(target_Q)
            max_Q = max(max_Q, torch.max(target_Q).item()) 
            # Calculate the final Q value from the target network parameters by using Bellman equation
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Get the Q values of the basis networks with the current parameters
            current_Q1, current_Q2 = self.critic(state, action)

            # Calculate the loss between the current Q value and the target Q value
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Perform the gradient descent
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            if it % policy_freq == 0:
                # Maximize the actor output value by performing gradient descent on negative Q values
                # (essentially perform gradient ascent)
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()

                # Use soft update to update the actor-target network parameters by
                # infusing small amount of current parameters
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                # Use soft update to update the critic-target network parameters by infusing
                # small amount of current parameters
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

            av_loss += loss
        self.iter_count += 1
        # Write new values for tensorboard
        self.writer.add_scalar("loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("Av. Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("Max. Q", max_Q, self.iter_count)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )




# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use CUDA if available, otherwise use CPU
device = torch.device("cpu")
seed = 42  # Random seed number
eval_freq = 5e3  # Evaluate after how many steps
max_ep = 500  # Maximum steps per episode
eval_ep = 10  # Number of episodes for evaluation
max_timesteps = 1e6 + 1 # Maximum steps to execute
expl_noise = 1  # Initial exploration noise starting value, in range [expl_min ... 1]

expl_decay_steps = 1000000  # How many steps the initial exploration noise will decay over, originally 500000
expl_min = 0.1  # Decayed exploration noise, in range [0 ... expl_noise]
batch_size = 40  # Mini-batch size
discount = 0.99999  # Discount factor for calculating discounted future rewards (should be close to 1)
tau = 0.005  # Soft target update variable (should be close to 0)
policy_noise = 0.2  # Noise added during exploration
noise_clip = 0.5  # Maximum limit for noise
policy_freq = 2  # Policy network update frequency
buffer_size = 1e6  # Maximum size of buffer
file_name = "TD3_velodyne"  # File name to store policy
save_model = True  # Whether to save model
load_model = False  # Whether to load stored model
random_near_obstacle = True  # Whether to take random actions near obstacles

# Create the network storage folders11
if not os.path.exists("./results"):
    os.makedirs("./results")
if save_model and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

# Create the training enviepisode_num >= evaluations_num
# robot_dim = 4
# env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
# time.sleep(5)
# torch.manual_seed(seed)
# np.random.seed(seed)
# state_dim = environment_dim + robot_dim
# action_dim = 2
# max_action = 1

environment_dim = 20  # Size of environment dimension, typically represents number of features in environment state space
robot_dim = 4  # Size of robot dimension, angular velocity, linear velocity, distance from robot to goal, angle difference from robot to goal
env = GazeboEnv("multi_robot_scenario.launch", environment_dim) # Create a Gazebo environment instance, load specified robot scenario launch file, and set environment dimension
time.sleep(5) # Wait 5 seconds to ensure Gazebo environment is fully started and initialized
torch.manual_seed(seed) # Set seed for PyTorch random number generator to ensure reproducibility
np.random.seed(seed) # Set seed for NumPy random number generator to ensure reproducibility
state_dim = environment_dim + robot_dim # Dimension of state space, sum of environment dimension and robot dimension
action_dim = 2 # Dimension of action space, may represent number or type of actions robot can execute
max_action = 1 # Maximum possible value of action, typically used to scale neural network output action values to actual action ranges


# Create the network
network = TD3(state_dim, action_dim, max_action)
# Create a replay buffer
replay_buffer = ReplayBuffer(buffer_size, seed)
if load_model:
    try:
        network.load(file_name, "./pytorch_models")
    except:
        print(
            "Could not load the stored model parameters, initializing training with random parameters"
        )

# Create evaluation data store
evaluations = [] # Initialize a list to store key data from evaluation process

timestep = 0 # Initialize timestep counter to track total timesteps in training process
timesteps_since_eval = 0 # Initialize timestep counter since last evaluation to determine when to perform next evaluation
episode_num = 0 # Initialize completed training episode count
done = True # Initialize boolean variable to indicate whether current episode is finished
epoch = 1 # Initialize training epoch counter to track current training epoch

count_rand_actions = 0 # Initialize counter to record number of random actions taken
random_action = [] # Initialize a list to store information about random actions

beta_noise = 1 # Balance coefficient to adapt noise scale to current training stability
min_reward = 50000.0
max_reward = -50000.0
evaluations_num = 10 # Number of episodes to measure training stability, previously set to 9 
beta_freq = 10 # Delayed update of balance correction factor
R = [] # Used to store normalized rewards
# Begin the training loop
# Main training loop, continues until maximum timestep is reached
writer = SummaryWriter()
success_goals =[] 
goal_rate = 0
a = 10 # Variance change coefficient
while timestep < max_timesteps:

    # If current episode is finished
    if done:
        # If current timestep is not 0, train the network
        if timestep != 0:
            network.train(
                replay_buffer,
                episode_timesteps,
                batch_size,
                discount,
                tau,
                policy_noise,
                noise_clip,
                policy_freq,
            )
            if episode_reward > max_reward:
                max_reward = episode_reward
            if episode_reward < min_reward:
                min_reward = episode_reward
            if min_reward != max_reward: 
                R.append((episode_reward - min_reward) / (max_reward - min_reward)) # Min-max normalization processing
            else:
                R.append(1)
            writer.add_scalar("episode_reward", episode_reward, timestep)    
            if episode_num % beta_freq == 0 and episode_num >= evaluations_num:
                last_R = R[-evaluations_num:] # Get last evaluations_num values from list R
                variance_of_last = a*statistics.variance(last_R) # Get variance value
                writer.add_scalar("variance_of_last", variance_of_last, timestep) 
                beta_noise = math.exp(-variance_of_last) # Get beta value
                writer.add_scalar("beta_noise", beta_noise, timestep) 


        # If timesteps since last evaluation reach evaluation frequency, perform evaluation
        if timesteps_since_eval >= eval_freq:
            print("Validating")
            # Reset evaluation counter
            timesteps_since_eval %= eval_freq
            # Add evaluation results
            evaluations.append(
                evaluate(timestep, network=network, epoch=epoch, eval_episodes=eval_ep)
            )
            # Save network model
            network.save(file_name, directory="./pytorch_models")
            # Save evaluation results
            np.save("./results/%s" % (file_name), evaluations)
            # Increase epoch count
            epoch += 1

        # Reset environment and get initial state
        state, last_distance, _, _ = env.reset()
        last_action = [0, 0] # Velocity of previous state
        # Set done flag to False
        done = False

        # Initialize episode reward and timestep count

        episode_reward = 0
        episode_timesteps = 0
        # Increase episode count
        episode_num += 1

    # Gradually decrease exploration noise
    if expl_noise > expl_min:
        expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)

    # Get action under current state and add exploration noise
    action = network.get_action(np.array(state))
    action = (action + beta_noise*np.random.normal(0, expl_noise, size=action_dim)).clip(
        -max_action, max_action
    )


    # If encountering obstacles, randomly force consistent random actions to increase exploration
    if random_near_obstacle:
        if (
            np.random.uniform(0, 1) > 0.85 # Generate random number
            and min(state[4:-8]) < 0.6 # Select part from state, calculate minimum of these values
            and count_rand_actions < 1 # 
        ):
            # Set duration of random action
            count_rand_actions = np.random.randint(8, 15)
            # Generate random action
            random_action = np.random.uniform(-1, 1, 2)

        if count_rand_actions > 0:
            # Reduce remaining time of random action
            count_rand_actions -= 1
            action = random_action
            action[0] = -1

    # Map action to range accepted by environment
    a_in = [(action[0] + 1) / 2, action[1]]


    # Execute action and get next state, reward and done flag
    next_state, reward, done, target, distance, done_target, collision, _ = env.step(a_in, last_distance, last_action, 0, 0) # Added to get robot's distance to goal
    # Get distance from previous state to goal point
    last_distance = distance
    # Get velocity of previous state
    last_action = a_in
    # If current timestep is not last timestep of episode, done flag is False
    done_bool = 1 if episode_timesteps + 1 == max_ep else int(done) # Changed done_bool=0 to =1,
    # If current timestep is last timestep of episode, done flag is True
    done = 1 if episode_timesteps + 1 == max_ep else int(done)
    if reward > 90:
        success_goals.append(1)
    elif reward < -90:
        success_goals.append(0)
    elif episode_timesteps + 1 == max_ep:
        success_goals.append(0)    
    if episode_num % 100 == 0 and episode_num > 0:
        success_goals = success_goals[-100:]
        count = 0
        for it in success_goals:
            if it:
                count += 1
        goal_rate = count / 100
        writer.add_scalar("goal_rate", goal_rate, timestep)  


    # Accumulate episode reward
    episode_reward += reward

    # Store experience tuple in experience replay buffer
    replay_buffer.add(state, action, reward, done_bool, next_state)

    # Update state and timestep counter
    state = next_state
    episode_timesteps += 1
    timestep += 1
    timesteps_since_eval += 1


# After the training is done, evaluate the network and save it
evaluations.append(evaluate(timestep, network=network, epoch=epoch, eval_episodes=eval_ep))
if save_model:
    network.save("%s" % file_name, directory="./models")
np.save("./results/%s" % file_name, evaluations)
