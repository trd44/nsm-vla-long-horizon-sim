#!/usr/bin/env python3
"""
Example script showing how to use dual resolution video recording.

This script demonstrates how to run your main script with full resolution video recording
while keeping model observations at the required size.
"""

import subprocess
import sys
from pathlib import Path

def run_with_full_res_video():
    """Run the main script with full resolution video recording enabled."""
    
    # Example command with full resolution video recording
    cmd = [
        "python", 
        "openpi/examples/libero/main_hanoi_multi_config_fixed.py",
        "--save-full-res-video",  # Enable full resolution video recording
        "--full-res-height", "640",  # Full resolution height
        "--full-res-width", "480",   # Full resolution width
        "--camera-height", "256",    # Model observation height (keep as is)
        "--camera-width", "256",     # Model observation width (keep as is)
        "--episodes", "1",           # Just run 1 episode for testing
        "--render-mode", "headless", # Save videos, don't display
    ]
    
    print("Running with dual resolution recording:")
    print("  - Model observations: 256x256 (for policy)")
    print("  - Full resolution videos: 640x480 (for data collection)")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Successfully completed!")
        print("Check the 'data/robosuite_videos' directory for:")
        print("  - *_full_res.mp4 (agentview camera)")
        print("  - *_full_res_wrist.mp4 (wrist camera)")
        print("  - Original .mp4 (model observation size)")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running script: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    
    return True

def show_output_structure():
    """Show what files will be generated."""
    print("\n📁 Expected output structure:")
    print("data/robosuite_videos/")
    print("├── Hanoi_multi_config_seed3_ep1_123456.mp4              # Model observation video (256x256)")
    print("├── Hanoi_multi_config_seed3_ep1_123456_full_res.mp4     # Full resolution agentview (640x480)")
    print("└── Hanoi_multi_config_seed3_ep1_123456_full_res_wrist.mp4 # Full resolution wrist (640x480)")
    print("\n🎯 Benefits:")
    print("  ✅ Model gets correctly sized observations (256x256)")
    print("  ✅ You get high-quality videos for data collection (640x480)")
    print("  ✅ Both cameras recorded at full resolution")
    print("  ✅ Frames are flipped correctly (no upside-down videos)")

if __name__ == "__main__":
    print("🚀 Dual Resolution Video Recording Example")
    print("=" * 50)
    
    show_output_structure()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        print("\n🏃 Running the script...")
        run_with_full_res_video()
    else:
        print("\n💡 To actually run the script, use:")
        print("python example_dual_resolution_usage.py --run")
        print("\nOr run the main script directly with:")
        print("python openpi/examples/libero/main_hanoi_multi_config_fixed.py --save-full-res-video --full-res-height 640 --full-res-width 480")
