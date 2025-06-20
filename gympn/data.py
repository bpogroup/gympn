import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import numpy as np


class GraphDataLoader(torch.utils.data.Dataset):
    def __init__(self, batch_size, states, actions, logprobs, advantages, logpis, values, data_type='hetero', force_batch_size=True):
        self.states = states
        self.actions = actions
        self.logprobs = logprobs
        self.advantages = advantages
        self.values = values
        self.logpis = logpis

        if data_type == 'hetero':
            self.data_list = []
            for index in range(len(states)):
                temp_h_data = states[index]['graph']
                temp_h_data.y = actions[index]
                temp_h_data.advantage = advantages[index]
                temp_h_data.logprobs = logprobs[index]
                temp_h_data.value = values[index]
                temp_h_data.logpis = logpis[index].squeeze(-1)
                self.data_list.append(temp_h_data)
        elif data_type == 'homogeneous':
            self.data_list = [Data(x=torch.from_numpy(states[index]['graph'].nodes), edge_index=torch.from_numpy(states[index]['graph'].edge_links),
                             y=actions[index], advantage=advantages[index], logprobs=logprobs[index], value=values[index])
                        for index in range(len(states))]
        else:
            raise ValueError("data_type must be either 'hetero' or 'homogeneous'")
        #self.batch = Batch.from_data_list(self.data_list)

        #Cut the data list to be a multiple of the batch size
        if force_batch_size:
            if len(self.data_list) % batch_size != 0:
                self.data_list = self.data_list[:len(self.data_list) - (len(self.data_list) % batch_size)]

        self.loader = DataLoader(self.data_list, batch_size=batch_size, shuffle=False) #shuffle to true would disrupt the normalized advantages

    def __getitem__(self, index):
        """
        Extract a data element and convert it to a PyTorch Geometric Data object.

        :param index: the index of the element to extract
        :return: the pytorch_geometric data object
        """
        data = Data(
            x=self.states[index][0],
            edge_index=self.states[index][1],
            y=self.actions[index],
            advantage=self.advantages[index],
            logprobs=self.logprobs[index],
            value=self.values[index],
            logpis=torch.tensor(self.logpis[index])
        )
        return data

    def __len__(self):
        return len(self.data_list)

def discount_rewards(rewards, gam):
    """Return discounted rewards-to-go computed from inputs.

    Parameters
    ----------
    :param rewards : array_like
        List or 1D array of rewards from a single complete trajectory.
    :param gam : float
        Discount rate.

    Returns
    -------
    rewards : ndarray
        1D array of discounted rewards-to-go.

    Examples
    --------
    >>> rewards = [1, 2, 3, 4, 5]
    >>> discount_rewards(rewards, 0.5)
    [1, 2, 6.25, 6.5, 5]

    """
    cumulative_reward = 0
    discounted_rewards = rewards.clone()
    for i in reversed(range(len(rewards))):
        cumulative_reward = rewards[i] + gam * cumulative_reward
        discounted_rewards[i] = cumulative_reward
    return discounted_rewards


def compute_advantages(rewards, values, gam, lam):
    """Return generalized advantage estimates computed from inputs.

    Parameters
    ----------
    :param rewards : Tensor
        1D tensor of rewards from a single complete trajectory.
    :param values : Tensor
        1D tensor of value predictions from a single complete trajectory.
    :param gam : float
        Discount rate.
    :param lam : float
        Parameter for generalized advantage estimation.
    :param end_flag : Tensor, optional
        1D tensor indicating whether the episode ends at each step.

    Returns
    -------
    advantages : Tensor
        1D tensor of computed advantage scores.
    """
    advantages = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):

        next_value = 0 if t + 1 >= len(values) else values[t + 1]
        delta = rewards[t] + gam * next_value - values[t]
        gae = delta + gam * lam * gae
        advantages[t] = gae
    return advantages

class TrajectoryBuffer:
    """A buffer to store and compute with trajectories.

    The buffer is used to store information from each step of interaction
    between the agent and environment. When a trajectory is finished it
    computes the discounted rewards and generalized advantage estimates. After
    some number of trajectories are finished it can return a tf.Dataset of the
    training data for policy gradient algorithms.

    Parameters
    ----------
    :param gam : float, optional
        Discount rate.
    :param lam : float, optional
        Parameter for generalized advantage estimation.

    See Also
    --------
    discount_rewards : Discount the list or array of rewards by gamma in-place.
    compute_advantages : Return generalized advantage estimates computed from inputs.

    """

    def __init__(self, gam=1, lam=1, data_type = 'hetero', action_mode="node_selection"):
        self.gam = gam
        self.lam = lam
        self.states = []
        self.actions = []
        self.rewards = torch.tensor([], dtype=torch.float32, requires_grad=True)
        self.logprobs = torch.tensor([], dtype=torch.float32, requires_grad=True)
        self.values =  torch.tensor([], dtype=torch.float32, requires_grad=True)
        self.logpis = []
        self.start = 0  # index to start of current episode
        self.end = 0  # index to one past end of current episode
        self.action_mode = action_mode # "node_selection" or "edge_selection"

        self.data_type = data_type

        #logging utilities
        self.prev_policy_loss = 0


    def store(self, state, action, reward, logprob, value, logpis):
        """Store the information from one interaction with the environment.

        Parameters
        ----------
        :param state : ndarray
           Observation of the state.
        :param action : int
           Chosen action in this trajectory.
        :param reward : float
           Reward received in the next transition.
        :param logprob : float
           Agent's logged probability of picking the chosen action.
        :param value : float
           Agent's computed value of the state.
        :param logpis : float
              Agent's logged probabilities of picking any action.

        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards = torch.cat((self.rewards, torch.tensor([reward])))
        self.logprobs = torch.cat((self.logprobs, torch.tensor([logprob])))
        self.values = torch.cat((self.values, torch.tensor([value])))
        self.logpis.append(logpis)
        self.end += 1

    def finish(self):
        """
        Finish an episode and compute advantages and discounted rewards.
        Advantages are stored in place of `values` and discounted rewards are
        stored in place of `rewards` for the current trajectory.
        """
        tau = slice(self.start, self.end)

        # Create end_flag tensor: True for the last step of the trajectory
        end_flag = torch.zeros_like(self.rewards[tau], dtype=torch.bool)
        end_flag[-1] = True  # Mark the last step as terminal
        values = compute_advantages(self.rewards[tau], self.values[tau], self.gam, self.lam)
        rewards = discount_rewards(self.rewards[tau], self.gam)
        self.rewards[tau] = rewards
        self.values[tau] = values
        self.start = self.end

    def clear(self):
        """Reset the buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards = torch.tensor([], dtype=torch.float32, requires_grad=True)
        self.logprobs = torch.tensor([], dtype=torch.float32, requires_grad=True)
        self.values = torch.tensor([], dtype=torch.float32, requires_grad=True)
        self.logpis = []
        self.start = 0
        self.end = 0

    def get(self, batch_size=64, normalize_advantages=True, sort=False, shuffle=True, drop_remainder=False):
        """Return a tf.Dataset of training data from this TrajectoryBuffer, along with the desired batch size.

        Parameters
        ----------
        :param batch_size : int, optional
            Batch size in the returned tf.Dataset.
        :param normalize_advantages : bool, optional
            Whether to normalize the returned advantages.
        :param sort : bool, optional
            Whether to sort by state shape before batching to minimize padding.
        :param drop_remainder : bool, optional
            Whether to drop the last batch if it has fewer than batch_size elements.

        Returns
        -------
        dataset : tf.Dataset
        batch_size : int

        """

        if shuffle:
            indices = np.random.permutation(len(self.states[:self.start]))
            self.states = [self.states[i] for i in indices]
            self.actions = list(np.array(self.actions)[indices])  # Convert to NumPy array for indexing
            self.logprobs = self.logprobs[indices]
            self.values = self.values[indices]
            self.rewards = self.rewards[indices]
            self.logpis = [self.logpis[i] for i in indices]


        actions = np.array(self.actions[:self.start], dtype=np.int32)
        logprobs = self.logprobs[:self.start]
        advantages = self.values[:self.start]
        values = self.rewards[:self.start]
        logpis = self.logpis[:self.start]



        if self.states: #and self.states[0].ndim == 2:

            # filter out any states with only one action available
            if self.action_mode == "node_selection":
                if self.data_type == 'hetero':
                    #no need to filter anything
                    states = self.states[:self.start]
                    pass

                elif self.data_type == 'homogeneous':
                    indices = [i for i in range(len(self.states[:self.start])) if len(self.states[i]['graph'].nodes) != 1]
                    states = [self.states[i] for i in indices]
                    actions = actions[indices]
                    logprobs = logprobs[indices]
                    advantages = advantages[indices]
                    values = values[indices]

                    if sort:
                        indices = np.argsort([s.shape[0] for s in states])
                        states = [states[i] for i in indices]
                        actions = actions[indices]
                        logprobs = logprobs[indices]
                        advantages = advantages[indices]
                        values = values[indices]

            elif self.action_mode == "edge_selection":
                indices = [i for i in range(len(self.states[:self.start])) if len(self.states[0]['graph'].edge_links) != 1]
            else:
                raise ValueError("Action_mode must be either 'node_selection' or 'edge_selection'")

            dataloader = GraphDataLoader(batch_size, states, actions, logprobs, advantages, logpis, values).loader

            #if normalize_advantages:
            #    for batch in dataloader:
            #        batch_advantages = batch['advantage']
            #        batch['advantage'] = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
            if normalize_advantages:
                # Compute global mean and std of advantages
                all_advantages = torch.cat([batch['advantage'] for batch in dataloader])
                global_mean = all_advantages.mean()
                global_std = all_advantages.std() + 1e-8

                # Normalize advantages for each batch
                for batch in dataloader:
                    batch['advantage'] = (batch['advantage'] - global_mean) / global_std


        else:
            raise ValueError("States must be non-empty.")

        return dataloader

    def __len__(self):
        return len(self.states)


def print_status_bar(i, epochs, history, verbose=1):
    """Print a status line."""
    metrics = "".join([" - {}: {:.4f}".format(m, history[m][i])
                       for m in ['mean_returns']])
    end = "\n" if verbose == 2 or i+1 == epochs else ""
    print("\rEpoch {}/{}".format(i+1, epochs) + metrics, end=end)