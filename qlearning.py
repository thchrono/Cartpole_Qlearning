import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt


class Qlearning:
    def __init__(self,env,alpha,gamma,epsilon,num_episodes,numBins,lower_bound,upper_bound):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_num = env.action_space.n
        self.num_episodes = num_episodes
        self.numBins = numBins
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # sum of all rewards in each episode
        self.episode_tot_reward = []

        # Q-matrix
        self.q_matrix = np.zeros((numBins[0],numBins[1],numBins[2],numBins[3],self.action_num))


    def get_discrete_state(self,state):
        '''
        Discretizes a continuous state into bin indices for use in a Q-table.

        Input:
            state (tuple): A tuple with 4 continuous values representing
                           (position, velocity, angle, angular_velocity).

        Output:
            tuple: A 4-tuple of integers (position_index, velocity_index, angle_index, angular_velocity_index),
                   where each index corresponds to the bin of the respective state variable.
        '''

        # get state
        position, velocity, angle, angular_velocity = state[0], state[1], state[2], state[3]

        # create bins for each state
        position_bin = np.linspace(start=self.lower_bound[0],stop=self.upper_bound[0],num=self.numBins[0])
        velocity_bin = np.linspace(start=self.lower_bound[1],stop=self.upper_bound[1],num=self.numBins[1])
        angle_bin = np.linspace(start=self.lower_bound[2],stop=self.upper_bound[2],num=self.numBins[2])
        angular_velocity_bin = np.linspace(start=self.lower_bound[3],stop=self.upper_bound[3],num=self.numBins[3])

        # get index and convert to 0-based
        position_index = np.maximum(np.digitize(position,position_bin) - 1, 0)
        velocity_index = np.maximum(np.digitize(velocity,velocity_bin) - 1, 0)
        angle_index = np.maximum(np.digitize(angle,angle_bin) - 1, 0)
        angular_velocity_index = np.maximum(np.digitize(angular_velocity,angular_velocity_bin) - 1, 0)

        return position_index, velocity_index, angle_index, angular_velocity_index


    def select_action(self,state):
        '''
        Selects an action using the epsilon-greedy strategy.

        Input:
            state (array-like): A list or array with 4 continuous values representing
                                [position, velocity, angle, angular_velocity].

        Output:
            int: Selected action index (either random or greedy based on epsilon).
        '''

        # this random number is used in the epsilon-greedy approach
        random_number = np.random.rand()

        # if the random number is smaller than epsilon,
        # then select a random action
        # otherwise select the action with the highest value
        # in the Q-matrix for the current state
        # if multiple actions have the same Q value,
        # select one of them at random
        if random_number < self.epsilon:
            return self.env.action_space.sample()

        else:
            # get state index
            state_index = self.get_discrete_state(state)
            # get q values for current state
            q_values = self.q_matrix[state_index]
            # find max q values and pick best action
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]

            return np.random.choice(best_actions)

    def training(self):

        for episode in range(self.num_episodes):
            # reset the environment
            state, _ = self.env.reset()

            terminal_state = False
            total_reward = 0
            # store all transitions for this episode
            episode_transitions = []
            while not terminal_state:
                # discretize current state
                state_index = self.get_discrete_state(state)

                # select action
                action = self.select_action(state)

                state_prime, reward, terminated, truncated, _ = self.env.step(action)
                terminal_state = terminated or truncated

                total_reward += reward

                # discretize next state
                state_prime_index = self.get_discrete_state(state_prime)

                episode_transitions.append((state_index, action, reward, state_prime_index, terminal_state))
                state = state_prime


            reward_threshold = 5
            # Q-learning update
            if total_reward > reward_threshold:
                for state_index, action, reward, state_prime_index, terminal_state in episode_transitions:
                    if not terminal_state:
                        error = reward + self.gamma * np.max(self.q_matrix[state_prime_index]) - self.q_matrix[state_index + (action,)]

                    else:
                        error = reward - self.q_matrix[state_index + (action,)]

                    self.q_matrix[state_index + (action,)] += self.alpha * error

                self.epsilon = np.maximum(0.01, self.epsilon * 0.999)

            self.episode_tot_reward.append(total_reward)

            if episode % 100 == 0:
                print(f"Episode {episode}: Total Reward = {total_reward}, Epsilon = {self.epsilon:.4f}")

            if episode % 200 == 0 and total_reward <= 450:
                reward_threshold += 5

            if episode >= 5000:
                self.alpha = 0.05


        # save Q-matrix at the end of training
        np.save("q_matrix_2.npy", self.q_matrix)
        print("Q-matrix saved to 'q_matrix.npy'")

        # average rewards per 100 episodes
        window_size = 100
        averaged_rewards = []
        for i in range(0, len(self.episode_tot_reward), window_size):
            averaged_rewards.append(np.mean(self.episode_tot_reward[i:i+window_size]))

        averaged_episodes = list(range(0, len(self.episode_tot_reward), window_size))

        plt.figure(figsize=(12, 6))
        plt.plot(averaged_episodes, averaged_rewards, label="Average Reward (per 100 episodes)", linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Q-learning Performance Over Episodes (smoothed)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("training_rewards.png")
        plt.show()

    def save_policy_video(self, env, output_filename="learned_policy.mp4", episodes=1):
        """
        Runs the learned policy and saves the simulation as an MP4 video.

        Args:
            env: Gymnasium environment created with render_mode="rgb_array"
            output_filename (str): File path for saving the video
            episodes (int): Number of episodes to record

        Returns:
            None
        """

        frames = []
        labels = []
        rewards_per_episode = []
        self.epsilon = 0
        self.q_matrix = np.load("q_matrix.npy")

        for ep in range(episodes):
            done = False
            total_reward = 0

            state, _ = env.reset()
            frame = env.render()
            frames.append(frame)
            labels.append(f"Reward: {int(total_reward)}")
            state_index = self.get_discrete_state(state)

            while not done:

                q_values = self.q_matrix[state_index]
                action = np.random.choice(np.where(q_values == np.max(q_values))[0])

                state_prime, reward, terminated, truncated, _ = env.step(action)

                state_prime_index = self.get_discrete_state(state_prime)
                state, state_index = state_prime, state_prime_index

                total_reward += reward

                done = terminated or truncated

            rewards_per_episode.append(total_reward)
            print(f"Episode {ep}: Total Reward = {total_reward}")


        env.close()

        # save frames as video
        fig, ax = plt.subplots()
        img = ax.imshow(frames[0])
        text = ax.text(10, 10, "", color="white", fontsize=12, weight="bold", backgroundcolor="black")
        plt.axis("off")

        def animate(i):
            img.set_array(frames[i])
            text.set_text(labels[i])
            return [img, text]

        ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=50)
        ani.save(output_filename, writer="ffmpeg", fps=30)
        print(f"Saved policy video to '{output_filename}'")