"""
Weights & Biases (W&B) integration for experiment tracking and visualization.

Provides a clean interface for logging metrics, charts, and hyperparameters
to W&B with local caching to avoid localhost issues.

Supports automatic opening of local visualization dashboard.
"""

import os
import wandb
import subprocess
import webbrowser
import time
from typing import Optional, Dict, Any
from pathlib import Path


class WandBLogger:
    """Wrapper around W&B for logging training metrics and hyperparameters."""

    def __init__(self,
                 project: str = "gympn-training",
                 entity: Optional[str] = None,
                 run_name: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 mode: str = "online",
                 save_locally: bool = True,
                 open_dashboard: bool = True,
                 dashboard_port: int = 8080):
        """
        Initialize W&B logger.

        Parameters
        ----------
        project : str
            W&B project name
        entity : str, optional
            W&B entity (team/username)
        run_name : str, optional
            Name for this training run
        config : dict, optional
            Hyperparameters and config to log
        mode : str
            "online" (sync to cloud), "offline" (local only), or "disabled"
        save_locally : bool
            Whether to save W&B run locally in addition to syncing
        open_dashboard : bool
            Whether to automatically open local W&B dashboard in browser
        dashboard_port : int
            Port for local W&B dashboard (default: 8080)
        """
        self.project = project
        self.entity = entity
        self.run_name = run_name
        self.config = config or {}
        self.mode = mode
        self.save_locally = save_locally
        self.open_dashboard = open_dashboard
        self.dashboard_port = dashboard_port
        self.run = None
        self.enabled = mode != "disabled"
        self.dashboard_process = None

        if self.enabled:
            self._initialize_wandb()

    def _initialize_wandb(self):
        """Initialize W&B run with local caching and optional dashboard."""
        try:
            # Set offline dir to avoid localhost port conflicts
            wandb_dir = Path(".wandb_offline")
            wandb_dir.mkdir(exist_ok=True)

            # Initialize run
            self.run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.run_name,
                config=self.config,
                mode=self.mode,
                dir=str(wandb_dir),  # Store artifacts locally
                settings=wandb.Settings(
                    code_dir=".",
                    console="auto",
                    quiet=False
                )
            )
            print(f"✓ W&B initialized: {self.project}/{self.run_name or 'unnamed'}")
            if self.mode == "offline":
                print(f"  (Offline mode - synced later via 'wandb sync .wandb_offline')")

            # Start local dashboard if requested
            if self.open_dashboard:
                self._start_local_dashboard()

        except Exception as e:
            print(f"⚠ Warning: Could not initialize W&B: {e}")
            self.enabled = False

    def _start_local_dashboard(self):
        """Start local W&B dashboard server and open in browser."""
        try:
            # First, ensure .wandb_offline directory exists and has content
            wandb_offline_dir = Path(".wandb_offline")
            if not wandb_offline_dir.exists():
                print("⚠ Warning: .wandb_offline directory doesn't exist yet.")
                print("  Dashboard will be available once data is logged.")
                return

            # Check if wandb CLI is available
            try:
                result = subprocess.run(['wandb', '--version'],
                                      capture_output=True,
                                      timeout=5,
                                      text=True)
                if result.returncode != 0:
                    print(f"⚠ Warning: wandb CLI check failed: {result.stderr}")
                    print("  Make sure wandb is installed: pip install wandb")
                    return

                # Check wandb version for serve command
                version_str = result.stdout.strip()
                print(f"  wandb version: {version_str}")

            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                print(f"⚠ Warning: wandb command not found in PATH: {e}")
                print("  Make sure wandb is installed: pip install --upgrade wandb")
                return

            # Try to use wandb serve
            print(f"\n🚀 Starting local W&B dashboard on localhost:{self.dashboard_port}...")
            print(f"  (This may take 10-15 seconds to start...)")

            # Start wandb serve process with output capture for debugging
            self.dashboard_process = subprocess.Popen(
                ['wandb', 'serve',
                 '--port', str(self.dashboard_port),
                 '--dir', str(wandb_offline_dir)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait for server to start and check if it's actually running
            max_retries = 15
            dashboard_started = False

            for i in range(max_retries):
                time.sleep(1)

                # Check if process is still alive
                if self.dashboard_process.poll() is not None:
                    # Process died, get the error
                    stdout, stderr = self.dashboard_process.communicate()

                    # Check if error is "No such command"
                    if "No such command" in stderr and "serve" in stderr:
                        print(f"✗ 'wandb serve' command not available")
                        print(f"  (wandb serve was removed or not installed)")
                        print(f"\n📊 ALTERNATIVE OPTIONS:")
                        print(f"\n  Option 1: Sync to W&B cloud (recommended)")
                        print(f"    After training: wandb sync .wandb_offline")
                        print(f"    Then view at: https://wandb.ai")
                        print(f"\n  Option 2: Use W&B online mode for real-time sync")
                        print(f"    Update config: 'wandb_mode': 'online'")
                        print(f"\n  Option 3: Install older wandb with serve support")
                        print(f"    pip install 'wandb==0.15.3'")
                        return
                    else:
                        print(f"✗ Dashboard failed to start!")
                        print(f"  Error: {stderr[:200]}")
                        return

                # Try to connect to the port
                try:
                    import socket
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    result = sock.connect_ex(('127.0.0.1', self.dashboard_port))
                    sock.close()

                    if result == 0:
                        print(f"✓ Dashboard is running!")
                        dashboard_started = True
                        break
                except Exception:
                    pass

                if i == max_retries - 1:
                    print(f"⚠ Dashboard didn't respond after {max_retries} seconds")
                    return

            if dashboard_started:
                # Open in browser
                dashboard_url = f"http://localhost:{self.dashboard_port}"
                print(f"📊 Dashboard URL: {dashboard_url}")
                try:
                    webbrowser.open(dashboard_url)
                except Exception:
                    print(f"  Open manually: {dashboard_url}")

                print(f"  Data dir: {wandb_offline_dir.absolute()}")
                print()

        except Exception as e:
            print(f"⚠ Error: {e}")


    def log_config(self, config: Dict[str, Any]):
        """Log hyperparameters and configuration."""
        if not self.enabled or not self.run:
            return
        try:
            self.run.config.update(config)
        except Exception as e:
            print(f"⚠ Warning: Could not log config to W&B: {e}")

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        Log training metrics.

        Parameters
        ----------
        metrics : dict
            Dictionary of metric_name -> value pairs
        step : int, optional
            Training step/epoch number (W&B auto-increments if not provided)
        """
        if not self.enabled or not self.run:
            return
        try:
            if step is not None:
                metrics = {**metrics, "epoch": step}
            self.run.log(metrics)
        except Exception as e:
            print(f"⚠ Warning: Could not log metrics to W&B: {e}")

    def log_epoch(self,
                  epoch: int,
                  mean_return: float,
                  std_return: float,
                  policy_loss: Optional[float] = None,
                  kld: Optional[float] = None,
                  entropy: Optional[float] = None,
                  value_loss: Optional[float] = None):
        """
        Log epoch training metrics.

        Parameters
        ----------
        epoch : int
            Epoch number
        mean_return : float
            Mean episode return
        std_return : float
            Std dev of episode returns
        policy_loss : float, optional
            Policy loss
        kld : float, optional
            KL divergence
        entropy : float, optional
            Policy entropy
        value_loss : float, optional
            Value function loss
        """
        metrics = {
            "train/mean_return": mean_return,
            "train/std_return": std_return,
        }

        if policy_loss is not None:
            metrics["train/policy_loss"] = policy_loss
        if kld is not None:
            metrics["train/kld"] = kld
        if entropy is not None:
            metrics["train/entropy"] = entropy
        if value_loss is not None:
            metrics["train/value_loss"] = value_loss

        self.log_metrics(metrics, step=epoch)

    def log_test(self,
                 epoch: int,
                 mean_return: float,
                 std_return: float,
                 min_return: Optional[float] = None,
                 max_return: Optional[float] = None):
        """
        Log test/evaluation metrics.

        Parameters
        ----------
        epoch : int
            Epoch number
        mean_return : float
            Mean test return
        std_return : float
            Std dev of test returns
        min_return : float, optional
            Minimum return in test set
        max_return : float, optional
            Maximum return in test set
        """
        metrics = {
            "test/mean_return": mean_return,
            "test/std_return": std_return,
        }

        if min_return is not None:
            metrics["test/min_return"] = min_return
        if max_return is not None:
            metrics["test/max_return"] = max_return

        self.log_metrics(metrics, step=epoch)

    def log_artifact(self, path: str, name: str = None):
        """
        Log an artifact (model, plot, etc.).

        Parameters
        ----------
        path : str
            Path to file to log
        name : str, optional
            Name for the artifact
        """
        if not self.enabled or not self.run:
            return
        try:
            self.run.log_artifact(path, name=name)
        except Exception as e:
            print(f"⚠ Warning: Could not log artifact to W&B: {e}")

    def log_summary(self, summary: Dict[str, Any]):
        """Log final summary statistics."""
        if not self.enabled or not self.run:
            return
        try:
            for key, value in summary.items():
                self.run.summary[key] = value
            self.run.summary.update()
        except Exception as e:
            print(f"⚠ Warning: Could not update W&B summary: {e}")

    def finish(self):
        """Finish the W&B run and cleanup dashboard."""
        if self.enabled and self.run:
            try:
                self.run.finish()
                print("✓ W&B run finished")
            except Exception as e:
                print(f"⚠ Warning: Could not finish W&B run: {e}")

        # Keep dashboard running for user to view results
        # (they can close it manually or press Ctrl+C)
        if self.dashboard_process:
            if self.dashboard_process.poll() is None:  # Still running
                print(f"\n📊 Local W&B dashboard is still running at http://localhost:{self.dashboard_port}")
                print("   You can view your training metrics in real-time")
                print("   Press Ctrl+C to stop the training script (dashboard stays open)")
                print(f"   Or manually stop it with: kill {self.dashboard_process.pid}  (Linux/Mac)")
                print(f"                             taskkill /PID {self.dashboard_process.pid}  (Windows)")
            else:
                print("⚠ Warning: Dashboard process terminated unexpectedly")
        elif self.open_dashboard:
            print(f"\n📊 To view results, start the dashboard manually:")
            print(f"   wandb serve --port {self.dashboard_port} --dir .wandb_offline")

    def cleanup(self):
        """Properly cleanup the dashboard process."""
        if self.dashboard_process and self.dashboard_process.poll() is None:
            try:
                self.dashboard_process.terminate()
                self.dashboard_process.wait(timeout=5)
                print("✓ Dashboard process terminated")
            except subprocess.TimeoutExpired:
                try:
                    self.dashboard_process.kill()
                    print("✓ Dashboard process killed")
                except Exception as e:
                    print(f"⚠ Warning: Could not cleanup dashboard process: {e}")


    def get_run_url(self) -> Optional[str]:
        """Get the W&B run URL."""
        if self.enabled and self.run:
            return self.run.get_url()
        return None


def init_wandb(project: str = "gympn-training",
               entity: Optional[str] = None,
               run_name: Optional[str] = None,
               config: Optional[Dict[str, Any]] = None,
               mode: str = "online",
               open_dashboard: bool = True,
               dashboard_port: int = 8080) -> WandBLogger:
    """
    Convenience function to initialize W&B logger.

    Parameters
    ----------
    project : str
        W&B project name
    entity : str, optional
        W&B entity (team/username)
    run_name : str, optional
        Name for the run
    config : dict, optional
        Hyperparameters
    mode : str
        "online", "offline", or "disabled"
    open_dashboard : bool
        Whether to open local W&B dashboard in browser (default: True)
    dashboard_port : int
        Port for local dashboard (default: 8080)

    Returns
    -------
    WandBLogger
        Initialized logger instance
    """
    return WandBLogger(
        project=project,
        entity=entity,
        run_name=run_name,
        config=config,
        mode=mode,
        save_locally=True,
        open_dashboard=open_dashboard,
        dashboard_port=dashboard_port
    )


def sync_offline_runs():
    """
    Sync offline W&B runs to cloud.

    Call this after training to upload all locally saved runs.
    """
    try:
        print("Syncing W&B offline runs...")
        os.system("wandb sync .wandb_offline")
        print("✓ Sync complete")
    except Exception as e:
        print(f"⚠ Could not sync W&B runs: {e}")

