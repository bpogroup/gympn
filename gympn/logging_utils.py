"""
Structured logging utilities for training and evaluation.
Provides clean, organized output for training progress and metrics.
"""

import sys
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    epoch: int
    mean_return: float
    std_return: float
    mean_length: float
    policy_loss: Optional[float] = None
    kld: Optional[float] = None
    entropy: Optional[float] = None


@dataclass
class TestMetrics:
    """Container for test metrics."""
    mean_return: float
    std_return: float
    mean_length: float
    min_return: float
    max_return: float


@dataclass
class EpisodeMetrics:
    """Container for per-episode metrics."""
    episode: int
    length: int
    return_value: float
    advantage_mean: Optional[float] = None
    advantage_std: Optional[float] = None


class Logger:
    """Structured logger for training process."""

    def __init__(self, verbose: int = 1, use_color: bool = True):
        """
        Initialize logger.

        Parameters
        ----------
        verbose : int
            Verbosity level (0=silent, 1=important, 2=detailed)
        use_color : bool
            Whether to use colored output
        """
        # Ensure verbose is an integer
        if isinstance(verbose, str):
            # If a string is passed (like __name__), use default
            self.verbose = 1
        else:
            self.verbose = int(verbose) if verbose is not None else 1
        self.use_color = use_color and sys.stdout.isatty()

    # Colors for terminal output
    _COLORS = {
        'HEADER': '\033[95m',
        'BLUE': '\033[94m',
        'CYAN': '\033[96m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'RED': '\033[91m',
        'ENDC': '\033[0m',
        'BOLD': '\033[1m',
    }

    def _colorize(self, text: str, color: str) -> str:
        """Add color to text if colors are enabled."""
        if not self.use_color:
            return text
        return f"{self._COLORS.get(color, '')}{text}{self._COLORS['ENDC']}"

    def _header(self, text: str) -> str:
        """Format header text."""
        return self._colorize(f"{'='*60}\n{text}\n{'='*60}", 'BOLD')

    def training_start(self, algorithm: str, epochs: int, episodes_per_epoch: int):
        """Log training start."""
        if self.verbose >= 1:
            msg = self._header(f"Training {algorithm}")
            print(msg)
            print(f"  Epochs: {epochs}, Episodes/epoch: {episodes_per_epoch}")
            print()

    def causal_rl_enabled(self):
        """Log causal RL mode is enabled."""
        if self.verbose >= 1:
            print(self._colorize("🔗 Causal RL Mode: Credits redistributed via causal traces", 'GREEN'))
            print()

    def epoch_start(self, epoch: int, total_epochs: int):
        """Log epoch start."""
        if self.verbose >= 2:
            print(self._colorize(f"[Epoch {epoch}/{total_epochs}]", 'CYAN'))

    def epoch_metrics(self, metrics: TrainingMetrics):
        """Log epoch metrics."""
        if self.verbose >= 1:
            msg = f"Epoch {metrics.epoch:3d} | Return: {metrics.mean_return:7.2f} ± {metrics.std_return:5.2f}"

            if metrics.policy_loss is not None:
                msg += f" | Loss: {metrics.policy_loss:7.4f}"
            if metrics.kld is not None:
                msg += f" | KLD: {metrics.kld:6.4f}"
            if metrics.entropy is not None:
                msg += f" | Ent: {metrics.entropy:6.4f}"

            print(msg)

    def test_metrics(self, metrics: TestMetrics, epoch: Optional[int] = None):
        """Log test/evaluation metrics."""
        if self.verbose >= 1:
            prefix = f"[Epoch {epoch}] " if epoch else ""
            print(self._colorize(
                f"{prefix}Test: Return {metrics.mean_return:7.2f} ± {metrics.std_return:5.2f} "
                f"[{metrics.min_return:6.2f}, {metrics.max_return:6.2f}]",
                'GREEN'
            ))

    def episode_advantage_stats(self, episode: int, num_steps: int,
                               adv_mean: float, adv_std: float):
        """Log episode advantage statistics."""
        if self.verbose >= 2:
            print(f"  Episode {episode}: {num_steps} steps | "
                  f"Adv: μ={adv_mean:7.4f}, σ={adv_std:7.4f}")

    def training_step_info(self, batch_num: int, total_batches: int,
                          loss: float, kld: Optional[float] = None):
        """Log training step info."""
        if self.verbose >= 2:
            msg = f"    Batch {batch_num}/{total_batches}: Loss={loss:7.4f}"
            if kld is not None:
                msg += f", KLD={kld:6.4f}"
            print(msg)

    def no_batches_warning(self):
        """Log warning when no batches to process."""
        if self.verbose >= 1:
            print(self._colorize("  ⚠ No complete batches to process", 'YELLOW'))

    def training_end(self, best_metric: float, metric_name: str = "return"):
        """Log training end."""
        if self.verbose >= 1:
            msg = self._header(f"Training Complete")
            print(msg)
            print(f"  Best {metric_name}: {best_metric:.4f}")
            print()

    def test_start(self, num_episodes: int):
        """Log test start."""
        if self.verbose >= 1:
            print(self._colorize(f"Testing ({num_episodes} episodes)...", 'BLUE'))

    def best_policy_saved(self, path: str, metric_value: float):
        """Log best policy save."""
        if self.verbose >= 1:
            print(self._colorize(f"  [BEST] Best policy saved: {metric_value:.4f}", 'GREEN'))

    def error(self, message: str):
        """Log error message."""
        print(self._colorize(f"ERROR: {message}", 'RED'))

    def warning(self, message: str):
        """Log warning message."""
        if self.verbose >= 1:
            print(self._colorize(f"WARNING: {message}", 'YELLOW'))

    def info(self, message: str):
        """Log info message."""
        if self.verbose >= 1:
            print(self._colorize(f"[INFO] {message}", 'BLUE'))

    def debug(self, message: str):
        """Log debug message."""
        if self.verbose >= 2:
            print(self._colorize(f"[DEBUG] {message}", 'CYAN'))

    def separator(self):
        """Print separator line."""
        if self.verbose >= 2:
            print("-" * 60)


# Global logger instance
_default_logger: Optional[Logger] = None


def get_logger(verbose: int = 1) -> Logger:
    """Get or create default logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = Logger(verbose=verbose)
    return _default_logger


def set_logger(logger: Logger):
    """Set the global logger."""
    global _default_logger
    _default_logger = logger

