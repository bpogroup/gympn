#!/usr/bin/env python
"""Test TensorBoard launch with the new wrapper."""

import sys
import os
import time
import subprocess
import tempfile

# Add gympn to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gympn.train import launch_tensorboard

# Create a temporary directory for logs
with tempfile.TemporaryDirectory() as tmpdir:
    print(f"Testing TensorBoard launch with logdir: {tmpdir}")

    # Create a dummy event file so TensorBoard has something to show
    event_dir = os.path.join(tmpdir, 'events')
    os.makedirs(event_dir, exist_ok=True)

    # Write a dummy event file
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(event_dir)
    writer.add_scalar('test/metric', 1.0, 0)
    writer.add_scalar('test/metric', 2.0, 1)
    writer.flush()
    writer.close()

    print("✓ Created dummy event files")

    # Try launching TensorBoard
    print("\nAttempting to launch TensorBoard...")
    process = launch_tensorboard(event_dir, port=6007, wait_time=3)

    if process is None:
        print("✗ TensorBoard launch failed")
        sys.exit(1)

    print("✓ TensorBoard process created successfully")

    # Clean up
    if process and process.poll() is None:
        print("ℹ Terminating TensorBoard process...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        print("✓ Process terminated")

print("\n✓ All tests passed!")

