import dataclasses

@dataclasses.dataclass
class Args:
    """Arguments for running the dataset making script
    
    Environments: AssemblyLineSorting, CubeSorting, Hanoi, Hanoi4x3, 
        HeightStacking, KitchenEnv, NutAssembly, PatternReplication
    """

    # --- Environment ---
    env: str = "AssemblyLineSorting"
    robot: str = "Kinova3"
    noise_std: float = 0.0       # Std for Gaussian action noise (scaled by remaining distance)
    noisy_fraction: float = 0.0  # Fraction of episodes that should use action noise (deterministic scheduling of the last fraction)
    random_block_placement: bool = False  # Place block on pegs randomly according to the rules of Towers of Hanoi
    random_block_selection: bool = False  # Randomly select 3 out of 4 blocks
    peg_xy_jitter: float = 0.025  # Std dev (meters) for XY jitter of the peg centers
    planner: str = "pddl"      # Planner type to use    

    # --- Assembly Line Specific ---
    cube_placement_noise: float = 0.025  # Uniform noise in meters to add to cube x and y positions during spawn
    bin_placement_noise: float = 0.025  # Uniform noise in meters to add to bin x and y positions during spawn

    # --- Observations ---
    relative_obs: bool = False  # Use relative gripper-goal features
    vision: bool = False        # Use vision-based observations

    # --- Episodes ---
    episodes: int = 5                # Number of episodes to record
    save_hd_agent_video: bool = True  # Save full resolution videos
    save_hd_wrist_video: bool = False  # Save full resolution videos

    # --- Random Seed ---
    seed: int = 0

    # --- Name ---
    name: str = None            # Optional name override for experiment ID

    # --- Directory ---
    dir: str = "./dataset_making/datasets"  # Directory for experiment outputs

    # --- Debugging ---
    verbose: bool = False      # Enable verbose debug output
    render: bool = False      # Render during execution (Not Working?)
