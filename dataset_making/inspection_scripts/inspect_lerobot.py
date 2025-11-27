import os
from pathlib import Path
import numpy as np
from PIL import Image
from itertools import islice

DATASET_PATH = "/home/hrilab/.cache/huggingface/lerobot/tduggan93/hanoi_50"
SAVE_GIF_DIR = Path("inspection_videos/lerobot")
SAVE_GIF_DIR.mkdir(exist_ok=True)

def save_frames_as_gif(frames, out_path, duration=100):
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0
    )

def load_lerobot_dataset(dataset_path):
    """Load LeRobot dataset from HuggingFace cache directory."""
    # Try using lerobot library first (preferred method)
    try:
        # Try both possible import paths
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        
        # Extract repo_id from path (e.g., /path/to/lerobot/tduggan93/hanoi_50 -> tduggan93/hanoi_50)
        dataset_path_str = str(dataset_path)
        if "lerobot" in dataset_path_str:
            # Extract the repo_id part after "lerobot/"
            parts = dataset_path_str.split("lerobot/")
            if len(parts) > 1:
                repo_id = parts[1]
            else:
                # Fallback: get last two path components
                path_parts = dataset_path_str.rstrip("/").split("/")
                repo_id = "/".join(path_parts[-2:])
        else:
            # Fallback: get last two path components
            path_parts = dataset_path_str.rstrip("/").split("/")
            repo_id = "/".join(path_parts[-2:])
        
        print(f"Loading LeRobot dataset with repo_id: {repo_id}")
        dataset = LeRobotDataset(repo_id)
        return dataset, None
    except ImportError as e:
        print(f"lerobot library not available: {e}")
        print("Please install lerobot: pip install lerobot")
        raise
    except Exception as e:
        print(f"Failed to load with lerobot library: {e}")
        import traceback
        traceback.print_exc()
        raise

def validate_lerobot_dataset():
    """Validate LeRobot dataset and check that action[3:6] are all zeros."""
    print(f"Loading dataset from: {DATASET_PATH}")
    
    try:
        dataset_or_episodes, meta = load_lerobot_dataset(DATASET_PATH)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Handle LeRobotDataset object
    # LeRobotDataset is indexed by frame, not episode
    dataset = dataset_or_episodes
    num_frames = len(dataset)
    print(f"Loaded LeRobot dataset with {num_frames} frames")
    print(f"Processing frames one at a time to avoid memory issues...")
    
    n_checked = 0
    n_broken = 0
    error_counts = {}
    
    required_obs_keys = ["image", "wrist_image", "state"]
    required_step_keys = ["action"]
    
    expected_shapes = {
        "image": (256, 256, 3),
        "wrist_image": (256, 256, 3),
        "state": (8,),
        "action": (7,),
    }
    expected_dtypes = {
        "image": np.uint8,
        "wrist_image": np.uint8,
        "state": np.float32,
        "action": np.float32,
    }
    
    # Process frames directly from dataset (one at a time to save memory)
    for i in range(num_frames):
        frame = dataset[i]
        # LeRobot format: observations and actions are at top level of frame dict
        if not isinstance(frame, dict):
            print(f"Frame {i}: Frame is not a dict, skipping")
            n_broken += 1
            error_counts["bad_frame_format"] = error_counts.get("bad_frame_format", 0) + 1
            continue
        
        # Extract observations and action
        obs = {}
        if "image" in frame:
            obs["image"] = frame["image"]
        if "wrist_image" in frame:
            obs["wrist_image"] = frame["wrist_image"]
        if "state" in frame:
            obs["state"] = frame["state"]
        # Also check nested observation structure
        if "observation" in frame:
            obs.update(frame["observation"])
        
        action = frame.get("action", frame.get("actions", None))
        
        if action is None:
            print(f"Frame {i}: missing 'action' or 'actions'")
            n_broken += 1
            error_counts["missing_action"] = error_counts.get("missing_action", 0) + 1
            continue
            
        # Check required observation keys
        for k in required_obs_keys:
            if k not in obs:
                print(f"Frame {i}: missing observation key '{k}'")
                n_broken += 1
                error_counts[f"missing_{k}"] = error_counts.get(f"missing_{k}", 0) + 1
        
        # Check shapes and dtypes
        try:
            # Convert to numpy arrays if needed
            action = np.asarray(action)
            
            for k in expected_shapes:
                if k == "action":
                    value = action
                else:
                    if k not in obs:
                        continue
                    value = np.asarray(obs[k])
                
                # if value.shape != expected_shapes[k]:
                #     print(f"Frame {i}: '{k}' shape {value.shape} != expected {expected_shapes[k]}")
                #     n_broken += 1
                #     error_counts[f"bad_shape_{k}"] = error_counts.get(f"bad_shape_{k}", 0) + 1
                
                # if value.dtype != expected_dtypes[k]:
                #     print(f"Frame {i}: '{k}' dtype {value.dtype} != expected {expected_dtypes[k]}")
                #     n_broken += 1
                #     error_counts[f"bad_dtype_{k}"] = error_counts.get(f"bad_dtype_{k}", 0) + 1
            
            # Check for NaN/Inf in state/action
            state = np.asarray(obs.get("state", []))
            if len(state) > 0:
                if np.any(np.isnan(state)):
                    print(f"Frame {i}: NaN found in 'state'")
                    n_broken += 1
                    error_counts["nan_state"] = error_counts.get("nan_state", 0) + 1
                if np.any(np.isinf(state)):
                    print(f"Frame {i}: Inf found in 'state'")
                    n_broken += 1
                    error_counts["inf_state"] = error_counts.get("inf_state", 0) + 1
            
            if np.any(np.isnan(action)):
                print(f"Frame {i}: NaN found in 'action'")
                n_broken += 1
                error_counts["nan_action"] = error_counts.get("nan_action", 0) + 1
            
            if np.any(np.isinf(action)):
                print(f"Frame {i}: Inf found in 'action'")
                n_broken += 1
                error_counts["inf_action"] = error_counts.get("inf_action", 0) + 1

            print(f"Action: {action}")
            
            # Check that rotation components (action[3:6]) are all zeros
            if len(action) >= 6:
                rotation_components = action[3:6]
                if not np.allclose(rotation_components, 0.0, atol=1e-6):
                    print(f"Frame {i}: Rotation components (action[3:6]) are not all zeros: {rotation_components}")
                    n_broken += 1
                    error_counts["non_zero_rotation"] = error_counts.get("non_zero_rotation", 0) + 1
            elif len(action) < 4:
                print(f"Frame {i}: Action length {len(action)} is too short (expected at least 4)")
                n_broken += 1
                error_counts["action_too_short"] = error_counts.get("action_too_short", 0) + 1
            
        except Exception as e:
            print(f"Frame {i}: Exception during checking: {e}")
            import traceback
            traceback.print_exc()
            n_broken += 1
            error_counts["exception"] = error_counts.get("exception", 0) + 1
        
        n_checked += 1
        
        # Progress update every 1000 frames
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{num_frames} frames... ({n_broken} issues found so far)")
    
    print(f"\nDone. Checked {n_checked} frames. Found {n_broken} issues.")
    if n_broken:
        print("Error counts:")
        for k, v in sorted(error_counts.items()):
            print(f"  {k}: {v}")
    else:
        print("✓ All checks passed!")

if __name__ == "__main__":
    validate_lerobot_dataset()

