#!/usr/bin/env python3
"""
Easy way to switch W&B modes without editing config files manually.

Usage:
    python switch_wandb_mode.py online username
    python switch_wandb_mode.py offline
    python switch_wandb_mode.py status
"""

import argparse
import json
from pathlib import Path
import re

def find_training_scripts():
    """Find training scripts in examples folder."""
    examples_dir = Path("examples")
    if not examples_dir.exists():
        return []

    scripts = []
    for py_file in examples_dir.glob("*.py"):
        if "train" in py_file.name.lower() or "SCG" in py_file.name:
            scripts.append(py_file)
    return scripts

def read_config(filepath):
    """Read Python file and extract default_args dict."""
    with open(filepath, 'r') as f:
        content = f.read()
    return content

def switch_to_online(filepath, entity):
    """Switch config to online mode."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Find and update default_args
    # Look for 'use_wandb': True or add it
    if "'use_wandb': True" not in content and '"use_wandb": True' not in content:
        # Add use_wandb if not present
        content = content.replace(
            "default_args = {",
            "default_args = {\n    'use_wandb': True,"
        )

    # Update mode to online
    content = re.sub(
        r"'wandb_mode':\s*'offline'",
        "'wandb_mode': 'online'",
        content
    )
    content = re.sub(
        r'"wandb_mode":\s*"offline"',
        '"wandb_mode": "online"',
        content
    )

    # Add entity if not present
    if 'wandb_entity' not in content:
        # Find line with wandb_mode and add entity after it
        content = re.sub(
            r"('wandb_mode':\s*'online')",
            r"\1,\n    'wandb_entity': '" + entity + "'",
            content
        )
    else:
        # Update existing entity
        content = re.sub(
            r"'wandb_entity':\s*'[^']*'",
            f"'wandb_entity': '{entity}'",
            content
        )

    with open(filepath, 'w') as f:
        f.write(content)

    return True

def switch_to_offline(filepath):
    """Switch config to offline mode."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Update mode to offline
    content = re.sub(
        r"'wandb_mode':\s*'online'",
        "'wandb_mode': 'offline'",
        content
    )
    content = re.sub(
        r'"wandb_mode":\s*"online"',
        '"wandb_mode": "offline"',
        content
    )

    with open(filepath, 'w') as f:
        f.write(content)

    return True

def check_mode(filepath):
    """Check current mode in config."""
    with open(filepath, 'r') as f:
        content = f.read()

    if "'wandb_mode': 'online'" in content or '"wandb_mode": "online"' in content:
        return "online"
    elif "'wandb_mode': 'offline'" in content or '"wandb_mode": "offline"' in content:
        return "offline"
    else:
        # Check if use_wandb is present
        if "'use_wandb': True" in content or '"use_wandb": True' in content:
            return "offline (default)"
        else:
            return "wandb disabled"

def main():
    parser = argparse.ArgumentParser(description="Switch W&B mode easily")
    parser.add_argument('mode', choices=['online', 'offline', 'status'],
                       help='Mode to switch to')
    parser.add_argument('--username', help='W&B username (required for online mode)')
    parser.add_argument('--script', help='Specific script to modify (default: all)')

    args = parser.parse_args()

    scripts = find_training_scripts()

    if not scripts:
        print("✗ No training scripts found in examples/")
        return

    if args.script:
        scripts = [Path(args.script)]

    print(f"\nFound {len(scripts)} training script(s):")
    for s in scripts:
        print(f"  • {s}")

    print()

    if args.mode == 'status':
        print("Current W&B configuration:")
        for script in scripts:
            mode = check_mode(script)
            print(f"  {script.name}: {mode}")

    elif args.mode == 'online':
        if not args.username:
            print("✗ --username is required for online mode")
            print("  Usage: python switch_wandb_mode.py online --username your-username")
            return

        print(f"🔄 Switching to online mode (syncing to {args.username})...")
        for script in scripts:
            try:
                switch_to_online(script, args.username)
                print(f"  ✓ {script.name}")
            except Exception as e:
                print(f"  ✗ {script.name}: {e}")

        print("\n✓ Done! Configuration updated.")
        print(f"\nNext steps:")
        print(f"  1. Run training: python examples/{scripts[0].name}")
        print(f"  2. When prompted, paste your W&B API key")
        print(f"  3. Watch results at: https://wandb.ai/{args.username}/gympn-training")

    elif args.mode == 'offline':
        print("🔄 Switching to offline mode...")
        for script in scripts:
            try:
                switch_to_offline(script)
                print(f"  ✓ {script.name}")
            except Exception as e:
                print(f"  ✗ {script.name}: {e}")

        print("\n✓ Done! Configuration updated.")
        print(f"\nNext steps:")
        print(f"  1. Run training: python examples/{scripts[0].name}")
        print(f"  2. After training: wandb sync .wandb_offline")
        print(f"  3. View results at: https://wandb.ai/your-username/gympn-training")

if __name__ == "__main__":
    main()

