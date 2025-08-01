#!/usr/bin/env python3
"""
Simple test to verify FER with TensorFlow works
"""

print("Testing TensorFlow...")
try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__} imported successfully")
except ImportError as e:
    print(f"✗ TensorFlow import failed: {e}")
    exit(1)

print("\nTesting FER...")
try:
    from fer import FER
    print("✓ FER imported successfully")
    
    # Test FER initialization
    detector = FER()
    print("✓ FER detector initialized successfully")
    
except ImportError as e:
    print(f"✗ FER import failed: {e}")
    exit(1)
except Exception as e:
    print(f"✗ FER initialization failed: {e}")
    exit(1)

print("\nTesting OpenCV...")
try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__} imported successfully")
except ImportError as e:
    print(f"✗ OpenCV import failed: {e}")
    exit(1)

print("\n🎉 All dependencies are working correctly!")
print("You can now run the facial expression recognition system.")
