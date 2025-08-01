#!/usr/bin/env python3
"""
Test script to verify imports and dependencies
"""
import sys
import os

print("Python version:", sys.version)
print("Python executable:", sys.executable)

# Test basic imports
try:
    import cv2
    print("✓ OpenCV imported successfully")
    print("  OpenCV version:", cv2.__version__)
except ImportError as e:
    print("✗ OpenCV import failed:", e)

try:
    import numpy as np
    print("✓ NumPy imported successfully")
    print("  NumPy version:", np.__version__)
except ImportError as e:
    print("✗ NumPy import failed:", e)

# Test moviepy import
try:
    import moviepy
    print("✓ MoviePy imported successfully")
    print("  MoviePy version:", moviepy.__version__)
    
    try:
        from moviepy import editor
        print("✓ MoviePy editor imported successfully")
    except ImportError as e:
        print("✗ MoviePy editor import failed:", e)
        
except ImportError as e:
    print("✗ MoviePy import failed:", e)

# Test FER import
try:
    from fer import FER
    print("✓ FER imported successfully")
except ImportError as e:
    print("✗ FER import failed:", e)
    print("  This is likely due to MoviePy dependency issues")

print("\nAll tests completed.")
