"""Launcher. Model code lives in sdm.py so Windows worker processes can import it."""
import os
from pathlib import Path

os.environ["PYTHONPATH"] = str(Path(__file__).resolve().parent)

if __name__ == "__main__":
    from sdm import main
    main()
