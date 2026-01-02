"""Utility scripts package.

This file exists so that DataLoader worker processes on Windows can import
scripts like `scripts.train_teacher` by module name (needed for pickling
custom transforms such as CLAHETransform).
"""
