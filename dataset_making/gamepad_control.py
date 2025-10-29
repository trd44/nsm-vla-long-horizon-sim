# gamepad_control.py

import os
import time
import argparse
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from utils import to_datestring
from robosuite.utils.detector import KitchenDetector, NutAssemblyDetector
from panda_hanoi_detector import PandaHanoiDetector as HanoiDetector
import pygame
import threading
from collections import deque
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

import threading
import queue


class GamepadController:
    """Gamepad controller for robot teleoperation."""
    
    def __init__(self, deadzone=0.1, max_velocity=0.3, orientation_enabled=False):
        """
        Initialize gamepad controller.
        
        Args:
            deadzone: Minimum joystick value to register as input
            max_velocity: Maximum velocity for robot movement
            orientation_enabled: Whether to enable roll/pitch/yaw controls
        """
        self.deadzone = deadzone
        self.max_velocity = max_velocity
        self.orientation_enabled = orientation_enabled
        self.running = False
        self.action_queue = deque(maxlen=10)  # Buffer for smooth actions
        
        # Initialize pygame for gamepad support
        pygame.init()
        pygame.joystick.init()
        
        # Check for connected gamepads
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No gamepad detected! Please connect a gamepad.")
        
        # Initialize the first gamepad
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"Gamepad initialized: {self.joystick.get_name()}")
        print(f"Number of axes: {self.joystick.get_numaxes()}")
        print(f"Number of buttons: {self.joystick.get_numbuttons()}")
        
        # Test all axes to see what's available
        print("Testing all axes (should all be 0 when not touched):")
        for i in range(self.joystick.get_numaxes()):
            val = self.joystick.get_axis(i)
            print(f"  Axis {i}: {val:.3f}")
        
        # Button mappings (can be customized)
        self.button_mappings = {
            'grasp': 0,      # A button (Xbox) / X button (PlayStation)
            'reset': 1,      # B button (Xbox) / Circle button (PlayStation)
            'record': 2,     # X button (Xbox) / Square button (PlayStation)
            'stop': 3,       # Y button (Xbox) / Triangle button (PlayStation)
        }
        
        # PlayStation 4 controller axis mappings - based on actual test
        # Axis 0: Left stick horizontal (delta X)
        # Axis 1: Left stick vertical (delta Y)
        # Axis 2: Left trigger (L2) - range -1 to 1
        # Axis 3: Right stick vertical (delta Z)
        # Axis 4: Right stick horizontal (delta roll)
        # Axis 5: Right trigger (R2) - range -1 to 1
        self.axis_mappings = {
            'x': 1,          # Left stick horizontal (delta X)
            'y': 0,          # Left stick vertical (delta Y)
            'z': 4,          # Right stick vertical (delta Z)
            'roll': 3,       # Right stick horizontal (delta roll)
            # 'pitch': 3,      # Left trigger (delta pitch)
            # 'yaw': 3,        # Right trigger (delta yaw)
            'grasp_trigger': 5,  # Right trigger for grasp control
        }
        
        # State tracking
        self.last_grasp_state = False
        self.grasp_toggle = False
        self.recording = False
        self.episode_data = []
        
    def start(self):
        """Start the gamepad controller thread."""
        self.running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        
    def stop(self):
        """Stop the gamepad controller."""
        self.running = False
        pygame.quit()
        
    def _control_loop(self):
        """Main control loop for processing gamepad input."""
        clock = pygame.time.Clock()
        
        while self.running:
            pygame.event.pump()  # Process pygame events
            
            # Get joystick values for delta commands
            x_axis = self.joystick.get_axis(self.axis_mappings['x'])
            y_axis = self.joystick.get_axis(self.axis_mappings['y'])
            z_axis = self.joystick.get_axis(self.axis_mappings['z'])
            
            # Use axis 3 for roll control
            roll_axis = self.joystick.get_axis(self.axis_mappings['roll'])
            
            # Use D-pad for pitch and yaw control (PS4 D-pad is accessed via hat)
            try:
                hat = self.joystick.get_hat(0)  # Get D-pad state (x, y)
                dpad_x, dpad_y = hat
                pitch_axis = -dpad_y  # Up = positive pitch, Down = negative pitch
                yaw_axis = dpad_x     # Right = positive yaw, Left = negative yaw
            except:
                # Fallback if hat doesn't work
                pitch_axis = 0.0
                yaw_axis = 0.0
            
            grasp_trigger = self.joystick.get_axis(self.axis_mappings['grasp_trigger'])
            
            # Apply deadzone
            x_axis = 0 if abs(x_axis) < self.deadzone else x_axis
            y_axis = 0 if abs(y_axis) < self.deadzone else y_axis
            z_axis = 0 if abs(z_axis) < self.deadzone else z_axis
            roll_axis = 0 if abs(roll_axis) < self.deadzone else roll_axis
            
            # Scale to max velocity for delta commands [dx, dy, dz, droll, dpitch, dyaw]
            if self.orientation_enabled:
                action = np.array([
                    x_axis * self.max_velocity,      # delta X
                    y_axis * self.max_velocity,      # delta Y
                    z_axis * self.max_velocity,      # delta Z
                    roll_axis * self.max_velocity,   # delta roll
                    pitch_axis * self.max_velocity,  # delta pitch
                    yaw_axis * self.max_velocity     # delta yaw
                ])
            else:
                # Position-only control [dx, dy, dz, 0, 0, 0]
                action = np.array([
                    x_axis * self.max_velocity,      # delta X
                    y_axis * self.max_velocity,      # delta Y
                    z_axis * self.max_velocity,      # delta Z
                    0, 0, 0                          # No orientation control
                ])
            
            # Handle grasp control
            grasp_button = self.joystick.get_button(self.button_mappings['grasp'])
            
            # Use trigger for continuous grasp control, button for toggle
            if abs(grasp_trigger) > 0.1:
                # Map trigger directly: -1 = open, +1 = closed
                grasp_action = grasp_trigger
            elif grasp_button and not self.last_grasp_state:
                self.grasp_toggle = not self.grasp_toggle
                grasp_action = 1.0 if self.grasp_toggle else -1.0
            else:
                grasp_action = 1.0 if self.grasp_toggle else -1.0
                
            self.last_grasp_state = grasp_button
            
            # Ensure grasp_action is a valid number
            if np.isnan(grasp_action) or np.isinf(grasp_action):
                grasp_action = -1.0  # Default to open
            
            # Add grasp action to the action vector (-1 = open, +1 = closed)
            action = np.append(action, grasp_action)
            
            # Add action to queue
            self.action_queue.append(action)
            
            # Debug: print axis values occasionally
            if hasattr(self, 'debug_counter'):
                self.debug_counter += 1
            else:
                self.debug_counter = 0
                
            if self.debug_counter % 100 == 0:  # Print every 100 frames
                print(f"Debug - X:{x_axis:.2f} Y:{y_axis:.2f} Z:{z_axis:.2f} Roll:{roll_axis:.2f} Pitch:{pitch_axis:.2f} Yaw:{yaw_axis:.2f} Grasp:{grasp_action:.2f}")
            
            # Handle other buttons
            if self.joystick.get_button(self.button_mappings['record']):
                if not self.recording:
                    self.recording = True
                    self.episode_data = []
                    print("Recording started!")
                    
            if self.joystick.get_button(self.button_mappings['stop']):
                self.recording = False
                print("Recording stopped")
            
            # Note: Reset button is handled in the main loop, not here
            
            # Limit frame rate
            clock.tick(60)
        
        self.running = False
        
    def get_action(self):
        """Get the next action from the queue."""
        if self.action_queue:
            return self.action_queue.popleft()
        return np.zeros(7)  # Default action (no movement)
        
    def is_reset_pressed(self):
        """Check if reset button is pressed."""
        return self.joystick.get_button(self.button_mappings['reset'])
        
    def is_recording(self):
        """Check if currently recording."""
        return self.recording
        
    def add_observation(self, obs, action):
        """Add observation and action to episode data."""
        if self.recording:
            self.episode_data.append({
                'observation': obs,
                'action': action,
                'timestamp': time.time()
            })
            
    def save_episode(self, episode_idx, save_dir):
        """Save the recorded episode data."""
        if not self.episode_data:
            return False
            
        os.makedirs(save_dir, exist_ok=True)
        episode_file = os.path.join(save_dir, f'episode_{episode_idx:03d}.npz')
        
        # Convert to numpy arrays
        observations = np.array([d['observation'] for d in self.episode_data])
        actions = np.array([d['action'] for d in self.episode_data])
        timestamps = np.array([d['timestamp'] for d in self.episode_data])
        
        np.savez(episode_file, 
                observations=observations,
                actions=actions, 
                timestamps=timestamps)
        
        print(f"Saved episode {episode_idx} with {len(self.episode_data)} steps to {episode_file}")
        return True


class WristCameraViewer:
    """Real-time wrist camera viewer for robot teleoperation."""
    
    def __init__(self, camera_name="robot0_eye_in_hand"):
        self.camera_name = camera_name
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 6))
        self.fig.suptitle(f'Wrist Camera: {camera_name}', fontsize=14)
        self.ax.set_title('Press Q to close camera view')
        self.ax.axis('off')
        
        # Initialize with a blank image
        self.img_display = self.ax.imshow(np.zeros((256, 256, 3), dtype=np.uint8))
        plt.ion()  # Turn on interactive mode
        plt.show(block=False)
        
    def update_frame(self, obs):
        """Update the wrist camera display with new observation."""
        try:
            # Handle different observation formats
            if isinstance(obs, dict):
                # Dictionary format - extract wrist camera image
                if 'robot0_eye_in_hand_image' in obs:
                    wrist_img = obs['robot0_eye_in_hand_image']
                elif 'robot0_eye_in_hand' in obs:
                    wrist_img = obs['robot0_eye_in_hand']
                else:
                    # Try to find any wrist camera key
                    wrist_keys = [k for k in obs.keys() if 'eye_in_hand' in k or 'wrist' in k]
                    if wrist_keys:
                        wrist_img = obs[wrist_keys[0]]
                    else:
                        return  # No wrist camera found
            elif isinstance(obs, np.ndarray):
                # Numpy array format - need to check if this is the right format
                # For now, skip if it's not a dictionary
                return
            else:
                return  # Unknown format
            
            # Convert to RGB if needed
            if len(wrist_img.shape) == 3 and wrist_img.shape[2] == 3:
                # Already RGB
                rgb_img = wrist_img
            elif len(wrist_img.shape) == 3 and wrist_img.shape[2] == 4:
                # RGBA, convert to RGB
                rgb_img = wrist_img[:, :, :3]
            else:
                # Grayscale, convert to RGB
                rgb_img = np.stack([wrist_img] * 3, axis=-1)
            
            # Normalize to 0-255 range if needed
            if rgb_img.max() <= 1.0:
                rgb_img = (rgb_img * 255).astype(np.uint8)
            else:
                rgb_img = rgb_img.astype(np.uint8)
            
            # Update the display
            self.img_display.set_array(rgb_img)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            # Only print error occasionally to avoid spam
            if not hasattr(self, '_last_error_time') or time.time() - self._last_error_time > 5.0:
                print(f"Error updating wrist camera: {e}")
                self._last_error_time = time.time()
    
    def close(self):
        """Close the camera viewer."""
        plt.close(self.fig)
        
    def is_closed(self):
        """Check if the viewer window is closed."""
        return not plt.fignum_exists(self.fig.number)


def keyboard_input_monitor(input_queue):
    """Monitor keyboard input in a separate thread."""
    while True:
        try:
            user_input = input()
            if user_input.strip().lower() == 'r':
                input_queue.put('reset')
        except:
            pass

def make_env(args):
    """Create and return a wrapped robosuite environment."""
    cube_noise = getattr(args, 'cube_init_pos_noise_std', 0.01)
    ctrl_cfg = suite.load_controller_config(default_controller='OSC_POSE')
    
    # Create environment with neutral robot orientation using initialization_noise
    try:
        # Set initialization noise to 0 to ensure deterministic starting position
        # This should prevent the random wrist orientation
        if args.env == 'Hanoi':
            # Create Hanoi environment directly to ensure custom parameters are passed
            from robosuite.environments.manipulation.hanoi import Hanoi
            from robosuite.models.robots.manipulators.panda_robot import Panda
            
            raw_env = Hanoi(
                robots="Panda",
                controller_configs=ctrl_cfg,
                has_renderer=args.render,
                has_offscreen_renderer=True,
                horizon=1e8,
                use_camera_obs=True,
                use_object_obs=True,
                camera_names=["agentview", "robot0_eye_in_hand"],
                camera_heights=256,
                camera_widths=256,
                random_reset=args.random_reset,
                randomize_block_config=True,
                cube_init_pos_noise_std=cube_noise
            )
        else:
            # Use robosuite.make for other environments
            raw_env = suite.make(
                args.env,
                robots="Panda",
                controller_configs=ctrl_cfg,
                has_renderer=args.render,
                has_offscreen_renderer=True,
                horizon=1e8,
                use_camera_obs=True,
                use_object_obs=True,
                camera_names=["agentview", "robot0_eye_in_hand"],
                camera_heights=256,
                camera_widths=256,
                random_reset=args.random_reset,
                cube_init_pos_noise_std=cube_noise
            )
        
        if args.random_reset:
            print("Environment created with RANDOM RESET enabled - blocks will be placed randomly on pegs")
        else:
            print("Environment created with deterministic robot initialization")
        
    except Exception as e:
        print(f"Error creating environment: {e}")
        # Fallback to basic environment creation
        raw_env = suite.make(
            args.env,
            robots="Panda",
            controller_configs=ctrl_cfg,
            has_renderer=args.render,
            has_offscreen_renderer=True,
            horizon=1e8,
            use_camera_obs=True,
            use_object_obs=True,
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=256,
            camera_widths=256,
            random_reset=args.random_reset,
            randomize_block_config=True,
            cube_init_pos_noise_std=cube_noise
        )
    
    # Store reference to raw environment for camera access
    raw_env.raw_env = raw_env
    # Gym-compatible wrapper, proprio_obs toggles vision-based vs proprio inputs
    env = GymWrapper(raw_env, proprio_obs=not args.vision)
    return env


def get_detector(env, env_name):
    """Instantiate the correct detector for the chosen environment."""
    if env_name == 'Hanoi':
        return HanoiDetector(env)
    if env_name == 'KitchenEnv':
        return KitchenDetector(env)
    if env_name == 'NutAssembly':
        return NutAssemblyDetector(env)
    raise ValueError(f"Unknown env {env_name}")


def main():
    parser = argparse.ArgumentParser(description="Control robot with gamepad for teleoperation.")
    parser.add_argument('--env', type=str, choices=['Hanoi', 'KitchenEnv', 'NutAssembly'], default='Hanoi')
    parser.add_argument('--dir', type=str, default='./gamepad_datasets', help='Base directory for recorded episodes')
    parser.add_argument('--render', action='store_true', help='Render during execution')
    parser.add_argument('--vision', action='store_true', help='Use vision-based observations')
    parser.add_argument('--max-velocity', type=float, default=0.3, help='Maximum velocity for robot movement')
    parser.add_argument('--deadzone', type=float, default=0.1, help='Joystick deadzone')
    parser.add_argument('--name', type=str, default=None, help='Optional name override for experiment ID')
    parser.add_argument('--wrist-camera', action='store_true', help='Display wrist camera view in separate window')
    parser.add_argument('--no-orientation', action='store_true', help='Disable orientation controls (pitch, yaw)')
    parser.add_argument('--random-reset', action='store_true', help='Enable random reset for Hanoi environment (randomizes block placement)')

    args = parser.parse_args()

    # Build experiment directory structure
    exp_name = f"{args.env.lower()}_gamepad"
    timestamp = to_datestring(time.time())
    exp_id = args.name if args.name else timestamp
    args.env_dir = os.path.join(args.dir, exp_name, exp_id)
    os.makedirs(args.env_dir, exist_ok=True)
    traces_dir = os.path.join(args.env_dir, 'traces')
    os.makedirs(traces_dir, exist_ok=True)

    print(f"Starting gamepad control experiment at {args.env_dir}")
    if args.random_reset:
        print("Random reset: ENABLED - blocks will be placed randomly on pegs")
    else:
        print("Random reset: DISABLED - blocks will be placed in default configuration")
    print("Controls:")
    print("  Left stick: delta X, Y movement")
    print("  Right stick horizontal: delta roll")
    print("  Right stick vertical: delta Z movement")
    if not args.no_orientation:
        print("  D-pad up/down: delta pitch")
        print("  D-pad left/right: delta yaw")
        print("  Note: Full 6-DOF control")
    else:
        print("  Note: Position control only (no orientation control)")
    print("  Right trigger (R2): grasp control (-1 = open, +1 = closed)")
    print("  A button: Toggle grasp")
    print("  X button: Start/stop recording")
    print("  Y button: Stop recording")
    print("  B button: Reset environment")
    if args.wrist_camera:
        print("  Wrist camera view enabled - separate window will show robot's perspective")

    # Create environment
    env = make_env(args)
    
    # Reset environment to ensure proper starting state
    env.reset()
    print("Environment reset to initial state")
    
    # Try to set robot to neutral orientation if possible
    try:
        raw_env = env.env.raw_env
        if hasattr(raw_env, 'robots') and len(raw_env.robots) > 0:
            robot = raw_env.robots[0]
            if hasattr(robot, 'init_qpos'):
                # Just print the initial joint positions for debugging
                print(f"Robot initial joint positions: {robot.init_qpos}")
                print("Note: Robot orientation is set by environment configuration")
    except Exception as e:
        print(f"Could not get robot info: {e}")
    
    # Get detector for the environment
    detector = get_detector(env, args.env)

    # Initialize gamepad controller
    try:
        gamepad = GamepadController(
            deadzone=args.deadzone,
            max_velocity=args.max_velocity,
            orientation_enabled=not args.no_orientation
        )
        gamepad.start()
    except Exception as e:
        print(f"Failed to initialize gamepad: {e}")
        return

    # Initialize wrist camera viewer if enabled
    wrist_camera_viewer = None
    if args.wrist_camera:
        wrist_camera_viewer = WristCameraViewer()

    # Create a queue for keyboard input
    input_queue = queue.Queue()
    
    # Start keyboard monitor thread
    keyboard_thread = threading.Thread(target=keyboard_input_monitor, args=(input_queue,), daemon=True)
    keyboard_thread.start()
    
    # Main control loop
    episode_idx = 0
    reset_pressed = False  # Flag to prevent multiple rapid resets
    try:
        while True:
            print(f"\n=== Episode {episode_idx + 1} ===")
            print("Press B button to reset environment, or type 'r' and press Enter to reset, or close window to exit")
            
            # Reset environment
            try:
                obs, info = env.reset()
            except Exception as e:
                obs = env.reset()
                info = None
            
            step_count = 0
            
            while True:
                # Check for keyboard reset (non-blocking)
                try:
                    if not input_queue.empty():
                        command = input_queue.get_nowait()
                        if command == 'reset':
                            print("R key pressed - resetting environment!")
                            
                            # Close wrist camera viewer if it exists
                            if wrist_camera_viewer:
                                try:
                                    wrist_camera_viewer.close()
                                    wrist_camera_viewer = None
                                except:
                                    pass
                            
                            # Reset environment
                            env.reset()
                            print("Environment reset - continuing control")
                            
                            # Recreate wrist camera viewer if it was enabled
                            if args.wrist_camera and wrist_camera_viewer is None:
                                wrist_camera_viewer = WristCameraViewer()
                                print("Wrist camera viewer recreated")
                            
                            # Reset step counter for new episode
                            step_count = 0
                            break  # Exit the inner loop to restart episode
                except queue.Empty:
                    pass  # No keyboard input
                
                # Get action from gamepad
                action = gamepad.get_action()
                
                # Check for gamepad reset (B button)
                if gamepad.is_reset_pressed() and not reset_pressed:
                    print("Reset button pressed!")
                    reset_pressed = True
                    
                    # Close wrist camera viewer if it exists
                    if wrist_camera_viewer:
                        try:
                            wrist_camera_viewer.close()
                            wrist_camera_viewer = None
                        except:
                            pass
                    
                    # Reset environment to fix orientation and continue
                    env.reset()
                    print("Environment reset - continuing control")
                    
                    # Recreate wrist camera viewer if it was enabled
                    if args.wrist_camera and wrist_camera_viewer is None:
                        wrist_camera_viewer = WristCameraViewer()
                        print("Wrist camera viewer recreated")
                    
                    # Reset step counter for new episode
                    step_count = 0
                    
                    # Longer delay to prevent button sticking and double-press
                    time.sleep(1.0)
                    continue
                elif not gamepad.is_reset_pressed():
                    # Only reset flag after button is fully released and some time has passed
                    if reset_pressed:
                        time.sleep(0.2)  # Small additional delay
                    reset_pressed = False  # Reset flag when button is released
                

                
                # Fill out the rest of the action space if necessary
                rem_action_dim = env.action_dim - action.size
                if rem_action_dim > 0:
                    # Initialize remaining action space
                    rem_action = np.zeros(rem_action_dim)
                    # This is a multi-arm setting, choose which arm to control and fill the rest with zeros
                    action = np.concatenate([action, rem_action])
                elif rem_action_dim < 0:
                    # We're in an environment with no gripper action space, so trim the action space to be the action dim
                    action = action[: env.action_dim]

                # Step through the simulation
                try:
                    obs, reward, terminated, truncated, info = env.step(action)
                except ValueError:
                    # Handle older robosuite versions that return 4 values
                    try:
                        obs, reward, done, info = env.step(action)
                        terminated = done
                        truncated = False
                    except:
                        # Final fallback
                        result = env.step(action)
                        if len(result) == 4:
                            obs, reward, done, info = result
                            terminated = done
                            truncated = False
                        else:
                            obs, reward, terminated, truncated, info = result
                
                # Add to episode data if recording
                gamepad.add_observation(obs, action)
                
                # Render
                if args.render:
                    env.render()
                
                # Update wrist camera viewer if enabled
                if wrist_camera_viewer:
                    # Try to get raw observations for wrist camera
                    try:
                        # Access the raw environment to get camera observations
                        raw_env = env.env.raw_env
                        
                        # Get raw observations which contain camera data
                        if hasattr(raw_env, '_get_observations'):
                            raw_obs = raw_env._get_observations()
                            wrist_camera_viewer.update_frame(raw_obs)
                        else:
                            # Fallback to wrapped observations
                            wrist_camera_viewer.update_frame(obs)
                            
                    except Exception as e:
                        # Final fallback to wrapped observations
                        wrist_camera_viewer.update_frame(obs)
                    
                    # Check if user closed the camera window
                    if wrist_camera_viewer.is_closed():
                        print("Wrist camera window closed by user")
                        wrist_camera_viewer = None
                    # Small delay to prevent overwhelming the display
                    time.sleep(0.01)
                
                step_count += 1
                
                # Print status every 50 steps with action info
                if step_count % 50 == 0:
                    recording_status = "RECORDING" if gamepad.is_recording() else "not recording"
                    action_mag = np.linalg.norm(action[:3])  # Magnitude of position action
                    grasp_val = action[-1] if len(action) > 6 else 0
                    print(f"Step {step_count}, Action mag: {action_mag:.3f}, Grasp: {grasp_val:.2f}, Reward: {reward:.3f}, {recording_status}")
                
                # Check if episode is done
                if terminated or truncated:
                    print(f"Episode {episode_idx + 1} completed in {step_count} steps")
                    break
            
            # Save episode if recording was active
            if gamepad.episode_data:
                gamepad.save_episode(episode_idx, traces_dir)
                episode_idx += 1
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Cleanup
        gamepad.stop()
        if wrist_camera_viewer:
            wrist_camera_viewer.close()
        print("Gamepad controller stopped")


if __name__ == '__main__':
    main()
