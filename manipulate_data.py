#!/usr/bin/env python3
import argparse
import pickle
import numpy as np
import os

def convert_episode(episode):
    """
    Convert a single episode (a list of alternating actions and images) into RLDS format.
    
    Each episode is assumed to possibly have an initial header (if the first element 
    is not a numpy array), then alternating pairs:
      action (shape: (4,)) followed by image (flat array of 196608 elements).
      
    The action is transformed from [dx, dy, dz, gripper] to 
    [dx, dy, dz, d_roll, d_pitch, d_yaw, gripper] by inserting three zeros.
    
    The image is reshaped to (256, 256, 3) assuming RGB images.
    """
    # Determine starting index (skip header if necessary)
    start_idx = 1
    if not isinstance(episode[0], np.ndarray):
        start_idx = 2

    # Compute number of complete (action, image) pairs
    num_steps = (len(episode) - start_idx) // 2

    observations = []
    actions = []
    rewards = []
    terminals = []
    infos = []

    for i in range(num_steps):
        # Extract the original action and image
        action_orig = episode[start_idx + 2 * i]  # shape: (4,)
        image_flat = episode[start_idx + 2 * i + 1] # flat array, expected size: 196608

        # Convert action: insert three zeros between translation and gripper.
        # Resulting order: [dx, dy, dz, d_roll, d_pitch, d_yaw, gripper]
        action_new = np.concatenate((action_orig[:3], np.zeros(3), action_orig[3:]))

        # Reshape the image to (256, 256, 3)
        try:
            image = image_flat.reshape((256, 256, 3))
        except Exception as e:
            print(f"Error reshaping image at step {i}: {e}")
            continue  # Skip this step if reshaping fails

        observations.append({"image": image})
        actions.append(action_new.tolist())
        rewards.append(0.0)      # Default reward (adjust if needed)
        terminals.append(False)  # Will mark the final step as terminal later
        infos.append({})         # No additional info

    # Mark the last step as terminal (if there is at least one step)
    if terminals:
        terminals[-1] = True

    # Return the episode dictionary in RLDS format.
    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "infos": infos
    }

def main(input_file, output_file, max_episodes=None):
    # Load the raw dataset from the pickle file.
    with open(input_file, "rb") as f:
        episodes_data = pickle.load(f)

    # Optionally limit the number of episodes to convert.
    if max_episodes is not None:
        episodes_data = episodes_data[:max_episodes]

    converted_episodes = []
    for idx, episode in enumerate(episodes_data):
        converted = convert_episode(episode)
        converted_episodes.append(converted)
        num_steps = len(converted.get("actions", []))
        print(f"Converted episode {idx+1}/{len(episodes_data)} with {num_steps} steps.")

    # Save the RLDS-formatted dataset.
    with open(output_file, "wb") as f:
        pickle.dump(converted_episodes, f)
    print(f"Saved converted dataset with {len(converted_episodes)} episodes to '{output_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert raw list data to RLDS format for OpenVLA fine-tuning."
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input pickle file with the raw episodes list.")
    parser.add_argument("--output_file", type=str, default="rlds_dataset.pkl",
                        help="Path to the output RLDS-format pickle file.")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Optional: maximum number of episodes to convert.")
    args = parser.parse_args()
    main(args.input_file, args.output_file, args.max_episodes)
