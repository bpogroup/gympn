#!/usr/bin/env python
"""
Wrapper script for TensorBoard that patches imghdr for Python 3.13+ compatibility.
This script applies the imghdr fix before importing TensorBoard.
"""

import sys
import os

# Apply imghdr fix for Python 3.13+ BEFORE any other imports
if sys.version_info >= (3, 13):
    import types

    def what(file, h=None):
        """Minimal imghdr.what implementation."""
        return None

    imghdr_module = types.ModuleType('imghdr')
    imghdr_module.what = what
    imghdr_module.tests = []
    sys.modules['imghdr'] = imghdr_module

# Now import and run TensorBoard
if __name__ == '__main__':
    from tensorboard.main import run_main
    sys.exit(run_main())

