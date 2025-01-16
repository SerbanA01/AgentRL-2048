from rl.callbacks import Callback
import matplotlib.pyplot as plt

class TrainingMetricsCallback(Callback):
    def __init__(self):
        self.episode_rewards = []
        self.episode_steps = []
        self.losses = []
        self.max_tiles = []
        self.current_episode_steps = 0
        self.episode_count = 0

    def on_episode_begin(self, episode, logs):
        """Reset step counter at the beginning of each episode."""
        self.current_episode_steps = 0

    def on_step_end(self, step, logs):
        """Count steps and capture losses."""
        self.current_episode_steps += 1
        if 'loss' in logs and logs['loss'] is not None:
            self.losses.append(logs['loss'])

    def on_episode_end(self, episode, logs):
        """Record episode metrics."""
        self.episode_rewards.append(logs['episode_reward'])
        self.episode_steps.append(self.current_episode_steps)
        self.episode_count += 1

        # Log the max tile from the environment
        if hasattr(self.env, 'last_info') and 'max_tile' in self.env.last_info:
            self.max_tiles.append(self.env.last_info['max_tile'])
        else:
            self.max_tiles.append(0)  # Placeholder if no max tile info is available



def plot_training_metrics(metrics_callback):
    """Visualize the training metrics."""
    plt.figure(figsize=(12, 8))

    # Plot episode rewards
    plt.subplot(2, 2, 1)
    plt.plot(metrics_callback.episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    # Plot episode steps
    plt.subplot(2, 2, 2)
    plt.plot(metrics_callback.episode_steps)
    plt.title('Episode Steps')
    plt.xlabel('Episode')
    plt.ylabel('Steps')

    # Plot losses
    if metrics_callback.losses:
        plt.subplot(2, 2, 3)
        plt.plot(metrics_callback.losses)
        plt.title('Losses')
        plt.xlabel('Step')
        plt.ylabel('Loss')

    # Plot max tiles
    plt.subplot(2, 2, 4)
    plt.plot(metrics_callback.max_tiles)
    plt.title('Max Tile Reached')
    plt.xlabel('Episode')
    plt.ylabel('Max Tile')

    plt.tight_layout()
    plt.show()

