## TO DO ##
# fix bug in models with more that one layer (grad var)
# add n_step method

import gymnasium as gym
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import FeatureUnion
from typing import List, Union


class Layer:
    def __init__(self, output_dim: int, input_dim: int=None, gamma: float=None, lmbda: float=None, e_trace: np.ndarray=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.lmbda = lmbda
        self.e_trace = e_trace       

    def _init_weights(self):
        # Initialize the weights
        self.w = np.random.randn(self.input_dim, self.output_dim)

    def forward(self, x):
        # Forward pass
        pass

    def backward(self, grad, learning_rate):
        # Backward pass
        pass

class DenseLayer(Layer):
    def __init__(self, output_dim: int, input_dim: int=None, gamma: float=None, lmbda: float=None, init_trace: bool=False):
        super().__init__(output_dim, input_dim)

    def forward(self, x):
        # Forward pass
        self.x = x
        return self.x.dot(self.w)

    def backward(self, grad, learning_rate, action):
        # Backward pass & update weights
        if self.e_trace is not None:
            # update eligibility trace
            self.e_trace[:,action] = np.clip((self.gamma * self.lmbda * self.e_trace[:,action]) + self.x, -10.0, 10.0)
            self.w[:,action] += learning_rate * grad * self.e_trace[:,action]
            # print(f'grad:{grad}')
            # print(f'e_trace[:,action]:{self.e_trace[:,action]}')
            # self.w[:,action] += learning_rate * grad.dot(self.x)

        else:
            self.w[:,action] -= learning_rate * grad.dot(self.x)

class Loss:
    def __init__(self):
        pass

    def forward(self, pred, y):
        # Forward pass
        pass

    def backward(self, pred, y):
        # Backward pass
        pass

class MeanSquaredError(Loss):
    def __init__(self):
        pass

    def forward(self, pred, y):
        # Forward pass
        return np.mean((pred-y)**2)

    def backward(self, pred, y):
        # Backward pass
        return pred-y
    
class TDError(Loss):
    def __init__(self, gamma: float=0.97, lmbda: float=0.5):
        self.gamma = gamma
        self.lmbda = lmbda
    
    def forward(self, pred, y):
        # Forward pass
        return y-pred

    def backward(self, pred, y):
        # Backward pass
        return y-pred

# Model class
class Model:
    def __init__(self, layers: List[Layer], loss: Loss, input_data: np.ndarray, learning_rate: float=0.1, learning_rate_decay: Union[bool, str]=False, decay_rate: float=0.97, min_learning_rate: float=0.001):
        self.layers = layers
        self.loss = loss
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.decay_rate = decay_rate
        self.min_learning_rate = min_learning_rate
        self._compile(input_data)

    def _compile(self, input_data):
        input_dim = input_data.shape[1]
        for layer in self.layers:
            # set input dimensions and initialize weights
            layer.input_dim = input_dim
            layer._init_weights()
            input_dim = layer.output_dim
            # if loss is TD Error, initialize gamma, lambda, and eligibility trace
            if isinstance(self.loss, TDError):
                for layer in self.layers:
                    layer.gamma = self.loss.gamma
                    layer.lmbda = self.loss.lmbda
                    layer.e_trace = np.zeros((layer.input_dim, layer.output_dim))


    def get_learning_rate(self, episode: int):
        if self.learning_rate_decay == 'constant':
            return self.learning_rate
        elif self.learning_rate_decay == 'exponential':
            return max(self.min_learning_rate, self.learning_rate * np.exp(-self.decay_rate*episode))
        elif self.learning_rate_decay == 'linear':
            return max(self.min_learning_rate, self.learning_rate - self.decay_rate * episode)
        elif self.learning_rate_decay == 'inverse':
            return max(self.min_learning_rate, self.learning_rate / (1 + self.decay_rate * episode))
        else:
            return self.learning_rate



    def predict(self, x):
        # Forward pass
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def update(self, x, y, action, episode):
        # Backward pass
        grad = self.loss.backward(self.predict(x)[:,action], y)
        for layer in reversed(self.layers):
            grad = layer.backward(grad, self.get_learning_rate(episode), action)

# Agent class
class Agent:
    def __init__(self, env, model: Model, transformer=None, epsilon=0.1, min_epsilon=0.1, epsilon_decay=0.99, gamma=0.9, step_reward=None, term_reward=None):
        self.env = env
        self.model = model
        self.transformer = transformer
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.step_reward = step_reward
        self.term_reward = term_reward

    def get_epsilon(self, episode):
        return max(self.min_epsilon, self.epsilon*np.exp(-self.epsilon_decay*episode))

    def get_action(self, state, episode):
        if np.random.random() < self.get_epsilon(episode):
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state))

    def play(self, n_episodes: int, render: bool=False)-> list[int]:
        rewards = []
        for episode in range(n_episodes):
            # re-instantiate environment with render mode set to 'human'
            self.env = gym.make(self.env.spec.id, render_mode="human")
            # Reset the environment
            state, _ = self.env.reset()
            # make sure state is 2d arrray
            state = np.atleast_2d(state)
            if self.transformer:
                    state = self.transformer.transform(state)
            done = False
            total_reward = 0
            while not done:
                # set the environment to render in real time
                if render:
                    self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, term, trunc, _ = self.env.step(action)
                # make sure next state is a 2D array
                next_state = np.atleast_2d(next_state)
                # transform next state if exists
                if self.transformer:
                    next_state = self.transformer.transform(next_state)
                if term or trunc:
                    done = True
                total_reward += reward
                # update state
                state = next_state
            # close the renderer if done
            if render:
                self.env.close()
            rewards.append(total_reward)
        return rewards


    def train(self, n_episodes, streamlit=False, update_freq=1):
        rewards = []
        for episode in range(n_episodes):
            # Reset the environment
            state, _ = self.env.reset()
            # make sure state is 2d arrray
            state = np.atleast_2d(state)
            if self.transformer:
                    state = self.transformer.transform(state)
            done = False
            total_reward = 0
            while not done:
                action = self.get_action(state, episode)
                next_state, reward, term, trunc, _ = self.env.step(action)
                if self.step_reward:
                    reward += self.step_reward
                total_reward += reward
                # make sure next state is a 2D array
                next_state = np.atleast_2d(next_state)
                # transform next state if exists
                if self.transformer:
                    next_state = self.transformer.transform(next_state)
                if term or trunc:
                    done = True
                    if term and self.term_reward:
                        reward += self.term_reward
                # update model
                G = reward + self.gamma*np.max(self.model.predict(next_state))
                self.model.update(state, G, action, episode)
                # update state
                state = next_state
            rewards.append(total_reward)
            if streamlit:
                if episode % update_freq == 0:
                    # print({
                    #     'episode': episode,
                    #     'total_reward': total_reward,
                    #     'avg_reward': np.mean(rewards[-100:]),
                    #     'epsilon': self.get_epsilon(episode),
                    #     'learning_rate': self.model.get_learning_rate(episode)
                    # })  # Print values for debugging
                    yield {
                        'episode': episode,
                        'total_reward': total_reward,
                        'avg_reward': np.mean(rewards[-100:]),
                        'epsilon': self.get_epsilon(episode),
                        'learning_rate': self.model.get_learning_rate(episode)
                    }
            elif episode % 100 == 0:
                print(f'episode {episode}: reward={total_reward} AVG/100={np.mean(rewards[-100:])} epsilon={self.get_epsilon(episode)}')

class Transformer:
    def __init__(self):
        self.transformers = []

    def add(self, transformer):
        self.transformers.append(transformer)

    def transform(self, x):
        for transformer in self.transformers:
            x = transformer.transform(x)
        return x

    def fit(self, x):
        for transformer in self.transformers:
            x = transformer.fit_transform(x)
        # return x

# write a function to gather sample states from env by taking random actions over episodes
def get_sample_states(env, n_episodes):
    """
    Plays n_episode number of episodes using random actions and returns states.
    Parameters:
    -env: gym environment.
    -n_episodes: number of episodes to play.

    Returns:
    -states: list of states.
    """

    states = []

    for i in range(n_episodes):
        done = False
        state, _ = env.reset()
        states.append(state)
        while not done:
            action = env.action_space.sample()
            next_state, reward, term, trunc, _ = env.step(action)
            states.append(next_state)
            if term or trunc:
                done = True
    return np.array(states)

def create_rbf_samplers(sigmas, n_components):
    rbf_samplers = []
    for e,(s,n) in enumerate(zip(sigmas, n_components)):
        rbf_samplers.append((f'rbf{e}',RBFSampler(gamma=s, n_components=n)))
    return rbf_samplers

# code the main function
# def main():
#     # set seed
#     np.random.seed(42)

#     #instantiate cartpole env
#     env = gym.make('CartPole-v1')

#     # collect sample states to fit scaler and rbf kernels
#     sample_states = get_sample_states(env, 10000)

#     # instantiate scaler
#     scaler = StandardScaler()

#     # instantiate rbf kernels and merge into a feature union
#     rbf_samplers = create_rbf_samplers([0.05, 0.1, 0.5, 1.0], [1000, 1000, 1000, 1000])
#     # create a FeatureUnion object of RBFSamplers
#     rbf_union = FeatureUnion([*rbf_samplers])

#     # instantiate a transformer class
#     transformer = Transformer()
#     # add scaler and rbf union
#     transformer.add(scaler)
#     transformer.add(rbf_union)
#     # fit transformer to sample states
#     transformer.fit(sample_states)

#     # create the layers of the model
#     layers = [
#         DenseLayer(4000, 2)
#     ]

#     loss = MeanSquaredError()

#     # create a model object
#     model = Model(layers=layers,
#                 loss=loss,
#                 learning_rate_decay='inverse',
#                 decay_rate=0.01)

#     # create agent
#     agent = Agent(env, model, transformer, epsilon=1.0, min_epsilon=0.0, gamma=0.97)

#     agent.train(1000)

#     rewards = agent.play(100)
#     print(rewards)

# if __name__ == "__main__":
#     main()
