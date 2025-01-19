import numpy as np
import random
import time
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


class Game2048:
    def __init__(self):
        self.score = 0
        self.invalidMoves = 0
        self.is_valid_move = True
        self.highest_tile = 0
        self.consecutive_invalid_moves = 0
        self.board = self.reset()


    def reset(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.invalidMoves = 0
        self.highest_tile = 0
        self.is_valid_move = True
        self.consecutive_invalid_moves = 0
        self.add_new_tile()
        self.add_new_tile()
        return self.get_board()

    def add_new_tile(self):
        empty_positions = np.argwhere(self.board == 0)
        if empty_positions.size > 0:
            row, col = random.choice(empty_positions)
            self.board[row, col] = 2 if random.random() < 0.9 else 4

    def move_left(self):
        changed = False
        for i in range(4):
            row = self.board[i][self.board[i] != 0]
            merged_row = []
            skip = False
            for j in range(len(row)):
                if skip:
                    skip = False
                    continue
                if j + 1 < len(row) and row[j] == row[j + 1]:
                    merged_row.append(row[j] * 2)
                    self.score += row[j] * 2
                    skip = True
                    changed = True
                else:
                    merged_row.append(row[j])
            merged_row += [0] * (4 - len(merged_row))
            if not np.array_equal(self.board[i], merged_row):
                changed = True
            self.board[i] = merged_row
        if changed:
            self.add_new_tile()
            self.highest_tile = np.max(self.board)
            self.is_valid_move = True
            self.consecutive_invalid_moves = 0
        else:
            self.invalidMoves += 1
            self.consecutive_invalid_moves += 1
            self.is_valid_move = False


    def move_right(self):
        self.board = np.fliplr(self.board)
        self.move_left()
        self.board = np.fliplr(self.board)

    def move_up(self):
        self.board = np.rot90(self.board)
        self.move_left()
        self.board = np.rot90(self.board, -1)

    def move_down(self):
        self.board = np.rot90(self.board)
        self.move_right()
        self.board = np.rot90(self.board, -1)

    def get_board(self):
        return self.board

    def get_score(self):
        return self.score

    def get_invalid_moves(self):
        return self.invalidMoves

    def get_highest_tile(self):
        return self.highest_tile

    def is_move_valid(self):
        return self.is_valid_move

    def can_move(self):
        return( np.any(self.board == 0) or np.any(
            self.board[:-1, :] == self.board[1:, :]
        ) or np.any(self.board[:, :-1] == self.board[:, 1:]) )and self.invalidMoves < 50

    def get_reward(self, prev_board):
        zero_old = np.count_nonzero(prev_board == 0)
        zero_new = np.count_nonzero(self.board == 0)
        if not self.is_valid_move:
            return -10*self.consecutive_invalid_moves
        elif not self.can_move():
            return -100
        elif self.highest_tile > np.max(prev_board):
            return 100
        elif zero_new > zero_old:
            return 50
        elif zero_new == zero_old:
            return 10
        else:
            return 0


def preprocess_state(board):
    return np.log2(np.maximum(board, 1)).flatten() / 11.0  # Normalize to [0, 1]

def build_ddqn_model(input_size=16, output_size=4, hidden_sizes=[128, 128]):
    model = Sequential()
    model.add(Input(shape=(input_size,)))  # Input layer
    for hidden_size in hidden_sizes:
        model.add(Dense(hidden_size, activation="relu"))  # Hidden layers
    model.add(Dense(output_size, activation="linear"))  # Output layer
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model

class PrioritizedReplayBuffer:
    def __init__(self, max_size=20000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.max_size = max_size
        self.buffer = []
        self.priorities = []
        self.alpha = alpha  # Priority exponent
        self.beta = beta  # Importance-sampling exponent
        self.beta_increment = beta_increment
        self.epsilon = 1e-6  # Small constant to avoid zero priority

    def add(self, experience, td_error):
        priority = (abs(td_error) + self.epsilon) ** self.alpha
        self.buffer.append(experience)
        self.priorities.append(priority)

        # Remove oldest experience if buffer exceeds max size
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
            self.priorities.pop(0)

    def sample(self, batch_size):
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()  # Normalize priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        # Importance-sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights

        # Increment beta for stability
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Get sampled experiences
        samples = [self.buffer[idx] for idx in indices]
        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        for i, td_error in zip(indices, td_errors):
            self.priorities[i] = (abs(td_error) + self.epsilon) ** self.alpha



class DDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.main_model = build_ddqn_model(state_size, action_size)
        self.target_model = build_ddqn_model(state_size, action_size)
        self.update_target_model()
        self.replay_buffer = PrioritizedReplayBuffer()
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.01
        self.batch_size = 64

    def update_target_model(self):
        self.target_model.set_weights(self.main_model.get_weights())

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        q_values = self.main_model.predict(state[np.newaxis, :], verbose=0)
        return np.argmax(q_values[0])

    def train(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return

        # Sample from prioritized replay buffer
        batch, indices, weights = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        # Predict Q-values
        q_values = self.main_model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)

        # Compute TD errors and targets
        td_errors = []
        for i in range(self.batch_size):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * np.max(next_q_values[i])
            td_errors.append(target - q_values[i, actions[i]])
            q_values[i, actions[i]] = target

        # Update the model
        self.main_model.fit(states, q_values, sample_weight=weights, verbose=0, batch_size=self.batch_size)

        # Update priorities in the replay buffer
        self.replay_buffer.update_priorities(indices, td_errors)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_ddqn(agent, episodes=200, update_target_every=10):
    game = Game2048()
    print("Starting training loop...")
    for episode in range(1, episodes + 1):
        start_time = time.time()
        state = preprocess_state(game.reset())
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            prev_board = game.get_board().copy()
            if action == 0:
                game.move_left()
            elif action == 1:
                game.move_right()
            elif action == 2:
                game.move_up()
            elif action == 3:
                game.move_down()

            next_state = preprocess_state(game.get_board())
            reward = game.get_reward(prev_board)

            if not game.can_move():
                done = True

            # Calculate initial TD error
            q_values = agent.main_model.predict(state[np.newaxis, :], verbose=0)
            target = reward
            if not done:
                target += agent.gamma * np.max(agent.target_model.predict(next_state[np.newaxis, :], verbose=0))
            td_error = target - q_values[0][action]

            # Add experience to PER buffer
            agent.replay_buffer.add((state, action, reward, next_state, done), td_error)
            state = next_state
            total_reward += reward

            # Train the agent
            agent.train()
        # Update target network periodically
        if episode % update_target_every == 0:
            agent.update_target_model()

        print( f"Episode {episode}/{episodes}, Score: {game.get_score()}, Invalid-Moves:{game.get_invalid_moves()}, Highest-tile: {game.get_highest_tile()} Reward: {total_reward}, time: {time.time() - start_time:.2f} seconds")
        if episode % 100 == 0:
            agent.main_model.save(f"ddqn_model_v2_{episode}.h5")


if __name__ == '__main__':
    print("Starting DDQN")
    trained_agent = DDQNAgent(state_size=16, action_size=4)  # Ensure state_size matches your game's input
    print("Starting DDQN training")
    train_ddqn(trained_agent)
    trained_agent.main_model.save("trained_ddqn_model_v2.h5")




