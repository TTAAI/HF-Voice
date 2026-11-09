#!/usr/bin/env python3
"""R72 training, prediction and evaluation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from pipeline import main

if __name__ == '__main__':
    main()
