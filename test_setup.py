#!/usr/bin/env python
"""Test script to verify environment setup"""

import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

print("\nChecking installed packages:")
packages = ['pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn', 'jupyter', 'notebook', 'ipykernel']

for package in packages:
    try:
        if package == 'sklearn':
            import sklearn
            print(f"[OK] scikit-learn {sklearn.__version__}")
        else:
            module = __import__(package)
            print(f"[OK] {package} {module.__version__}")
    except ImportError as e:
        print(f"[FAIL] {package} - Not installed")
    except AttributeError:
        print(f"[OK] {package} - Installed (no version info)")

print("\nSetup complete! You can now open prompt_III.ipynb in VS Code.")