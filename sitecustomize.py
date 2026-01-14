"""
Sitecustomize module to handle missing imghdr in Python 3.13+
This provides a compatibility shim for TensorBoard.
"""

import sys
import os

# Only apply this patch if imghdr is missing or incomplete (Python 3.13+)
if sys.version_info >= (3, 13):
    try:
        import imghdr
        # Check if imghdr has the 'tests' attribute (for TensorBoard)
        if not hasattr(imghdr, 'tests'):
            raise AttributeError("imghdr missing 'tests' attribute")
    except (ModuleNotFoundError, AttributeError):
        # Create a minimal imghdr compatibility module
        import types

        # Try to use PIL if available
        try:
            from PIL import Image

            def what(file, h=None):
                """Identify image file type (minimal implementation using PIL)."""
                if h is None:
                    if isinstance(file, str):
                        try:
                            img = Image.open(file)
                            fmt = img.format
                            return fmt.lower() if fmt else None
                        except:
                            return None
                    return None
                else:
                    # Try to identify from bytes
                    try:
                        from io import BytesIO
                        img = Image.open(BytesIO(h))
                        fmt = img.format
                        return fmt.lower() if fmt else None
                    except:
                        return None
        except ImportError:
            # Fallback without PIL - just return None
            def what(file, h=None):
                """Minimal imghdr fallback."""
                return None

        # Create and inject the module
        imghdr_module = types.ModuleType('imghdr')
        imghdr_module.what = what
        imghdr_module.tests = []  # TensorBoard appends to this list
        sys.modules['imghdr'] = imghdr_module
