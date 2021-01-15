import random
import numpy as np
import torch
from matplotlib import pyplot as plt
import fire
from tqdm import tqdm


class ContextBanditEnv(object):
    def __init__(self,
                 max_score=10,
                 num_states=10,
                 num_actions=10):
        """
        ContextBanditEnv can be initialized with specific number of states and actions
        :param num_actions: Number of actions
        """
        self.state = None
        self.probability_matrix = None

        self.num_actions = num_actions
        self.num_states = num_states
        self.max_score = max_score

        self.init_reward_probability_distribution(num_states=num_states, num_actions=num_actions)
        self.update_state()

    def init_reward_probability_distribution(self, num_actions, num_states):
        """

        :param num_actions:
        :param num_states:
        :return:
        """
        # each row represents a state, each column an action
        # The higher the action probability in that state, the higher the reward by taking that action
        # Eg: 4 x 4
        # [
        #     [s1a1, s1a2, s1a3, s1a4],
        #     [s2a1, s2a2, s2a3, s2a4],
        #     [s3a1, s3a2, s3a3, s3a4],
        #     [s4a1, s4a2, s4a3, s4a4]
        # ]
        self.probability_matrix = np.random.rand(num_states, num_actions)

    def update_state(self):
        self.state = np.random.randint(0, self.num_states)

    """
    The way we’ve chosen to implement our reward probability distributions for each
    arm is this: Each arm will have a probability, e.g., 0.7, and the maximum reward is self.max_score.
    
    We will set up a for loop going to max_score, and at each step it will add 1 to the reward if a
    random float is less than the arm’s probability. Thus, on the first loop it makes up a
    random float (e.g., 0.4). 0.4 is less than 0.7, so reward += 1. On the next iteration, it
    makes up another random float (e.g., 0.6) which is also less than 0.7, so reward += 1.
    This continues until we complete max_score iterations, and then we return the final total
    reward, which could be anything between 0 and max_score. With an arm probability of 0.7,
    the average reward of doing this to infinity would be 0.7 * self.max_score, 
    but on any single play it could be more or less.
    """
    def reward(self, prob, n):
        """
        Rewards based on given probability and the number of actions.
        :param prob: The greater the probability the more the reward will be!
        :return:
        """
        reward = 0
        for i in range(n):
            if random.random() < prob:
                reward += 1
        return reward

    def get_state(self):
        """
        Returns a random state
        :return:
        """
        return self.state

    def get_reward(self, action):
        state = self.get_state()
        return self.reward(self.probability_matrix[state][action], n=self.max_score)

    def choose_action(self, action):
        reward = self.get_reward(action)
        self.update_state()
        return reward


def softmax(av, tau=1.12):
    softm = ( np.exp(av / tau) / np.sum( np.exp(av / tau) ) )
    return softm


def one_hot(N, pos, val=1):
    one_hot_vec = np.zeros(N)
    one_hot_vec[pos] = val
    return one_hot_vec


def running_mean(x,N=50):
    c = x.shape[0] - N
    y = np.zeros(c)
    conv = np.ones(N)
    for i in range(c):
        y[i] = (x[i:i+N] @ conv)/N
    return y


def get_model(input_dimension, hidden_dim, out_dim):
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dimension, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, out_dim),
        torch.nn.ReLU()
    )
    return model


def main(num_actions=10,
         num_states=10,
         max_score=100,
         batch_size=1,
         num_hidden_neurons=100,
         epochs=5000,
         learning_rate=1e-2):
    """

    :param num_actions:
    :param num_states:
    :param batch_size:
    :param num_hidden_neurons:
    :return:
    """
    rewards = []
    model = get_model(input_dimension=num_states,
                      hidden_dim=num_hidden_neurons,
                      out_dim=num_actions)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    env = ContextBanditEnv(num_actions=num_actions,
                           num_states=num_states,
                           max_score=max_score)

    current_state = torch.Tensor(one_hot(num_states, env.get_state()))

    # state -> model -> y_pred -> softmax ->  action_probabilities -> action -> reward
    # target = action_probabilities
    # target[action] = reward
    # loss(y_pred, target)
    for current_epoch in tqdm(range(1, epochs+1)):
        # Runs neural net forward get reward predictions
        y_pred = model(current_state)  # [num_actions,]
        # Converts reward predictions to probability distribution with softmax
        action_value = softmax(y_pred.data.numpy(), tau=2)
        # Normalizes the distribution to make sure it sums to 1
        action_value /= action_value.sum()
        # action_value : list of probabilities
        random_action_choice = np.random.choice(num_actions, p=action_value)
        current_reward = env.choose_action(random_action_choice)
        rewards.append(current_reward)

        target = y_pred.data.numpy().copy()  # [num_actions,]
        # Array index update
        target[random_action_choice] = current_reward
        target = torch.Tensor(target)
        # target is same as prediction except the action index value
        loss = loss_fn(y_pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current_state = torch.Tensor(one_hot(num_states, env.get_state()))

    rewards = np.array(rewards)
    plt.plot(running_mean(rewards, N=int(epochs/100)))
    plt.show()
    print(f"Mean reward : {rewards.mean()}")


if __name__ == '__main__':
    fire.Fire(main)

"""
python context_bandits_with_pytorch.py --num_actions=10 --num_hidden_neurons=100 --num_states=10 --max_score=100 --epochs=50000
python context_bandits_with_pytorch.py --num_actions=100 --num_hidden_neurons=1000 --num_states=100 --epochs=50000

"""
