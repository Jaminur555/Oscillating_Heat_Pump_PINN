"""Put src/ on sys.path so tests run without installing the package
(python -m unittest discover -s tests, from the repo root)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
