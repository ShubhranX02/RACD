"""
pytest configuration — adds src/ to the path for all tests.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
