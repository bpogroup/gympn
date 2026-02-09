"""
RUDDER (RUlE-based Data-driven Explanation of Rewards) implementation for credit assignment.
Redistributes rewards based on learned importance of trajectory steps.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import List, Dict, Tuple, Any
import warnings


class RUDDERNetwork(nn.Module):
    """
    RUDDER reward predictor network using BiLSTM.
    Learns to predict cumulative returns from state sequences.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # BiLSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Output head for reward prediction
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(
        self,
        states: torch.Tensor,
        lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through RUDDER network.

        Args:
            states: (batch_size, seq_len, state_dim) - padded state sequences
            lengths: (batch_size,) - actual sequence lengths

        Returns:
            step_rewards: (batch_size, seq_len) - predicted reward contribution at each step
            total_reward: (batch_size,) - predicted total return per episode
        """
        # Pack padded sequences
        packed = nn.utils.rnn.pack_padded_sequence(
            states, lengths, batch_first=True, enforce_sorted=False
        )

        # LSTM forward pass
        lstm_out, _ = self.lstm(packed)

        # Unpack sequences
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # Predict rewards at each step
        step_rewards = self.output_head(unpacked).squeeze(-1)  # (batch_size, seq_len)

        # Mask padding and sum for total prediction
        mask = self._create_mask(lengths, device=states.device)
        masked_rewards = step_rewards * mask
        total_reward = masked_rewards.sum(dim=1)  # (batch_size,)

        return step_rewards, total_reward

    @staticmethod
    def _create_mask(lengths: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Create mask for variable length sequences."""
        batch_size, max_len = lengths.shape[0], lengths.max().item()
        mask = torch.arange(max_len, device=device).expand(batch_size, max_len)
        mask = (mask < lengths.unsqueeze(1)).float()
        return mask


class RUDDERCreditAssignment:
    """
    RUDDER credit assignment engine.
    Trains a network to predict returns and uses this to redistribute rewards.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        learning_rate: float = 1e-3,
        device: str = 'cpu'
    ):
        self.device = device
        self.state_dim = state_dim

        self.network = RUDDERNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        ).to(device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.training_losses = []

    def prepare_batch(
        self,
        trajectories: List[Dict[str, np.ndarray]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare trajectory batch for training.

        Args:
            trajectories: List of dicts with 'states' key

        Returns:
            padded_states: (batch_size, max_seq_len, state_dim)
            lengths: (batch_size,)
            state_sequences: List of tensors (original)
        """
        state_sequences = []

        for traj in trajectories:
            states = np.array(traj['states'], dtype=np.float32)
            if states.ndim == 1:
                states = states.reshape(-1, 1)
            state_sequences.append(torch.tensor(states, dtype=torch.float32))

        # Pad sequences
        padded_states = pad_sequence(state_sequences, batch_first=True)

        # Get lengths
        lengths = torch.tensor(
            [len(s) for s in state_sequences],
            dtype=torch.long
        )

        return padded_states, lengths, state_sequences

    def train_step(
        self,
        trajectories: List[Dict[str, np.ndarray]],
        returns: np.ndarray
    ) -> float:
        """
        Train RUDDER network to predict returns.

        Args:
            trajectories: List of trajectory dicts with 'states' key
            returns: (batch_size,) - actual episode returns

        Returns:
            loss: Training loss value
        """
        if len(trajectories) == 0:
            return 0.0

        # Prepare batch
        padded_states, lengths, _ = self.prepare_batch(trajectories)
        padded_states = padded_states.to(self.device)
        lengths = lengths.to(self.device)

        # Convert returns to tensor
        target_returns = torch.tensor(
            returns, dtype=torch.float32
        ).to(self.device)

        # Forward pass
        step_rewards, predicted_returns = self.network(padded_states, lengths)

        # Compute loss
        loss = self.criterion(predicted_returns, target_returns)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        loss_value = loss.item()
        self.training_losses.append(loss_value)

        return loss_value

    def redistribute_rewards(
        self,
        trajectories: List[Dict[str, np.ndarray]],
        method: str = 'contribution'
    ) -> List[np.ndarray]:
        """
        Redistribute rewards based on RUDDER predictions.

        Args:
            trajectories: List of trajectory dicts with 'states' and 'rewards' keys
            method: 'contribution' (relative importance) or 'direct' (predicted values)

        Returns:
            redistributed_rewards: List of redistributed reward arrays
        """
        self.network.eval()
        redistributed = []

        with torch.no_grad():
            for traj in trajectories:
                states = np.array(traj['states'], dtype=np.float32)
                if states.ndim == 1:
                    states = states.reshape(-1, 1)

                states_tensor = torch.tensor(states, dtype=torch.float32).unsqueeze(0).to(self.device)
                lengths = torch.tensor([len(states)], dtype=torch.long).to(self.device)

                # Get RUDDER predictions
                step_rewards, total_pred = self.network(states_tensor, lengths)
                step_rewards = step_rewards[0, :lengths[0]]

                # Get original rewards
                original_rewards = np.array(traj['rewards'], dtype=np.float32)
                original_return = original_rewards.sum()
                predicted_return = total_pred[0].item()

                if method == 'contribution':
                    # Use relative importance: normalize by sum and scale to match original return
                    abs_rewards = torch.abs(step_rewards)

                    if abs_rewards.sum() > 1e-6:
                        importance = abs_rewards / abs_rewards.sum()
                        redistributed_rewards = (importance * original_return).cpu().numpy()
                    else:
                        # Fallback: uniform distribution
                        redistributed_rewards = np.full_like(original_rewards,
                                                            original_return / len(original_rewards))

                elif method == 'direct':
                    # Scale predictions to match original return
                    if abs(predicted_return) > 1e-6:
                        scaling = original_return / predicted_return
                    else:
                        scaling = 1.0

                    redistributed_rewards = (step_rewards * scaling).cpu().numpy()

                else:
                    raise ValueError(f"Unknown redistribution method: {method}")

                redistributed.append(redistributed_rewards)

        self.network.train()
        return redistributed


class RUDDERAgent:
    """
    RUDDER-based agent for credit assignment in reinforcement learning.
    Trains alongside a regular RL agent to improve credit assignment.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        device: str = 'cpu',
        training_frequency: int = 1,
        redistribution_method: str = 'contribution'
    ):
        """
        Initialize RUDDER agent.

        Args:
            state_dim: Dimension of state vectors
            hidden_dim: Hidden dimension for LSTM
            learning_rate: Learning rate for RUDDER network
            device: Device to use ('cpu' or 'cuda')
            training_frequency: Train RUDDER every N epochs
            redistribution_method: Method for redistributing rewards
        """
        self.state_dim = state_dim
        self.device = device
        self.training_frequency = training_frequency
        self.redistribution_method = redistribution_method

        self.credit_assignment = RUDDERCreditAssignment(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            device=device
        )

        self.trajectory_buffer = []
        self.returns_buffer = []
        self.epoch_counter = 0

    def add_trajectory(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        episode_return: float
    ):
        """
        Add a trajectory to the training buffer.

        Args:
            states: (seq_len, state_dim) - state sequence
            actions: (seq_len,) - action sequence
            rewards: (seq_len,) - reward sequence
            episode_return: Total return for episode
        """
        trajectory = {
            'states': states,
            'actions': actions,
            'rewards': rewards
        }
        self.trajectory_buffer.append(trajectory)
        self.returns_buffer.append(episode_return)

    def train(self, num_epochs: int = 5) -> float:
        """
        Train RUDDER on buffered trajectories.

        Args:
            num_epochs: Number of training epochs

        Returns:
            average_loss: Average training loss
        """
        if len(self.trajectory_buffer) == 0:
            warnings.warn("No trajectories in buffer for RUDDER training")
            return 0.0

        total_loss = 0.0

        for epoch in range(num_epochs):
            loss = self.credit_assignment.train_step(
                self.trajectory_buffer,
                np.array(self.returns_buffer)
            )
            total_loss += loss

        avg_loss = total_loss / num_epochs

        # Clear buffers after training
        self.trajectory_buffer = []
        self.returns_buffer = []

        return avg_loss

    def redistribute_rewards(
        self,
        states: np.ndarray,
        rewards: np.ndarray
    ) -> np.ndarray:
        """
        Redistribute rewards for a trajectory using RUDDER.

        Args:
            states: (seq_len, state_dim) - state sequence
            rewards: (seq_len,) - reward sequence

        Returns:
            redistributed: (seq_len,) - redistributed rewards
        """
        trajectory = {
            'states': states,
            'rewards': rewards
        }

        redistributed = self.credit_assignment.redistribute_rewards(
            [trajectory],
            method=self.redistribution_method
        )

        return redistributed[0]

    def should_train(self) -> bool:
        """Check if RUDDER should be trained this epoch."""
        return self.epoch_counter % self.training_frequency == 0

    def step_epoch(self):
        """Increment epoch counter."""
        self.epoch_counter += 1

