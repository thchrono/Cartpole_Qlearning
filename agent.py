import gymnasium as gym
from qlearning import Qlearning

# create environment
env = gym.make('CartPole-v1', render_mode="rgb_array")

# parameters for discretization
upper_bound = env.observation_space.high
lower_bound = env.observation_space.low

# need to overwrite the boundary values of velocity and
# angular_velocity because the range of values for both
# of them is [-inf, inf] and they can't be discretized
velocity_min, velocity_max = -3.5, 3.5
angular_velocity_min, angular_velocity_max = -3.5, 3.5
upper_bound[1], upper_bound[3] = velocity_max, angular_velocity_max
lower_bound[1], lower_bound[3] = velocity_min, angular_velocity_min

# hyperparameters
ALPHA= 0.1
GAMMA = 1
EPSILON = 1
NUM_EPISODES = 25000
NUM_BINS = [10,10,10,10]

# create agent
agent = Qlearning(env,ALPHA,GAMMA,EPSILON,NUM_EPISODES,NUM_BINS,lower_bound,upper_bound)

# train agent
agent.training()

# run learned policy
agent.save_policy_video(env, output_filename="learned_policy.mp4", episodes=8)