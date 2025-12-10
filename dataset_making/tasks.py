"""
Tasks
"""
import os
import numpy as np
from dataset_making.utils import cap, to_osc_pose

np.set_printoptions(linewidth=200, precision=4, suppress=True)

# ===== DEBUG PLOTTING FLAG =====
DEBUG_PLOTS = False  # Set to True to enable control debugging plots
DEBUG_PLOTS_DIR = "./debug_plots"
# ===============================

# ======== PATH VARIABLES ========
PARABOLIC_PATHS = True  # Use parabolic paths for XY movement
MAX_HEIGHT = 1.2        # Maximum height for ascend operations
# ================================

if DEBUG_PLOTS:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    os.makedirs(DEBUG_PLOTS_DIR, exist_ok=True)

class TaskOperation:
    """Base class for PDDL-driven operations, with common motion loops."""
    def __init__(self, args, env, detector, randomize=False, noise_std=0.03):
        self.args = args
        self.env = env
        self.detector = detector
        self.randomize = randomize
        self.noise_std = noise_std
        # Align our gripper reference with the detector's mapping so XY centering
        # uses the SAME frame the detector uses for `over(gripper,·)`.
        grip_name = detector.object_id.get('gripper', 'gripper0_eef')
        self._grip_name = grip_name
        self._gripper_is_site = False
        self._gripper_site = None
        self._gripper_body = None
        # Try site first
        try:
            self._gripper_site = env.sim.model.site_name2id(grip_name)
            self._gripper_is_site = True
        except Exception:
            # Fallback to body
            self._gripper_body = env.sim.model.body_name2id(grip_name)
            self._gripper_is_site = False
        self._open_sign = -1.0
        # -1 means "holding open," +1 means "holding closed," 0 means "no command"
        self._grip_cmd = 0
        
        # PID controller state for XY movement
        self._xy_integral = np.zeros(2, dtype=float)
        self._xy_prev_error = np.zeros(2, dtype=float)
        
        # PID controller state for Z movement
        self._z_integral = 0.0
        self._z_prev_error = 0.0
        
        # Debug plotting state
        self._plot_counter = 0
        self._debug_data = None
    
    def _gripper_pos(self):
        if self._gripper_is_site:
            return np.array(self.env.sim.data.site_xpos[self._gripper_site])
        else:
            return np.array(self.env.sim.data.body_xpos[self._gripper_body])
    
    def _measure_aperture(self):
        j1 = float(self.env.sim.data.get_joint_qpos('gripper0_finger_joint1'))
        j2 = float(self.env.sim.data.get_joint_qpos('gripper0_finger_joint2'))
        return abs(j1 - j2)

    def _calibrate_open_sign(self, steps: int = 20):
        """Try +1 and -1 for `steps` frames and see which increases aperture more."""
        base = self._measure_aperture()
        deltas = {}
        for sign in (+1, -1):
            # apply `sign` for a few steps
            for _ in range(steps):
                self.env.step(to_osc_pose(np.array([0, 0, 0, sign])))
            aper = self._measure_aperture()
            deltas[sign] = aper - base
            for _ in range(steps):
                self.env.step(to_osc_pose(np.array([0, 0, 0, -sign])))
        best = max(deltas, key=deltas.get)
        return best

    def record(self, obs, action):
        if hasattr(self.env, 'record_step'):
            self.env.record_step(obs, action)
    
    def _reset_xy_pid(self):
        """Reset XY PID controller state."""
        self._xy_integral = np.zeros(2, dtype=float)
        self._xy_prev_error = np.zeros(2, dtype=float)
    
    def _reset_z_pid(self):
        """Reset Z PID controller state."""
        self._z_integral = 0.0
        self._z_prev_error = 0.0
    
    def _init_debug_data(self, operation_name, has_xy=True, has_z=True):
        """Initialize debug data collection for plotting."""
        if not DEBUG_PLOTS:
            return
        
        self._debug_data = {
            'operation': operation_name,
            'has_xy': has_xy,
            'has_z': has_z,
            'time': [],
            'pos_x': [], 'pos_y': [], 'pos_z': [],
            'goal_x': [], 'goal_y': [], 'goal_z': [],
            'error_x': [], 'error_y': [], 'error_z': [],
            'p_term_x': [], 'p_term_y': [], 'p_term_z': [],
            'i_term_x': [], 'i_term_y': [], 'i_term_z': [],
            'd_term_x': [], 'd_term_y': [], 'd_term_z': [],
            'action_x': [], 'action_y': [], 'action_z': [],
        }
    
    def _record_debug_step(self, timestep, pos, goal, error, p_term, i_term, d_term, action):
        """Record one step of debug data."""
        if not DEBUG_PLOTS or self._debug_data is None:
            return
        
        self._debug_data['time'].append(timestep)
        self._debug_data['pos_x'].append(pos[0])
        self._debug_data['pos_y'].append(pos[1])
        self._debug_data['pos_z'].append(pos[2])
        self._debug_data['goal_x'].append(goal[0])
        self._debug_data['goal_y'].append(goal[1])
        self._debug_data['goal_z'].append(goal[2])
        self._debug_data['error_x'].append(error[0])
        self._debug_data['error_y'].append(error[1])
        self._debug_data['error_z'].append(error[2])
        self._debug_data['p_term_x'].append(p_term[0])
        self._debug_data['p_term_y'].append(p_term[1])
        self._debug_data['p_term_z'].append(p_term[2])
        self._debug_data['i_term_x'].append(i_term[0])
        self._debug_data['i_term_y'].append(i_term[1])
        self._debug_data['i_term_z'].append(i_term[2])
        self._debug_data['d_term_x'].append(d_term[0])
        self._debug_data['d_term_y'].append(d_term[1])
        self._debug_data['d_term_z'].append(d_term[2])
        self._debug_data['action_x'].append(action[0])
        self._debug_data['action_y'].append(action[1])
        self._debug_data['action_z'].append(action[2])
        
    def _plot_debug_data(self):
        """Generate and save control plots."""
        if not DEBUG_PLOTS or self._debug_data is None or len(self._debug_data['time']) == 0:
            return
        
        data = self._debug_data
        time = np.array(data['time'])
        
        # Create figure with subplots
        if data['has_xy'] and data['has_z']:
            fig, axes = plt.subplots(3, 2, figsize=(14, 12))
            fig.suptitle(f"{data['operation']} - Control Analysis", fontsize=14, fontweight='bold')
        elif data['has_xy']:
            fig, axes = plt.subplots(2, 2, figsize=(14, 8))
            fig.suptitle(f"{data['operation']} - XY Control Analysis", fontsize=14, fontweight='bold')
        else:  # Z only
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
            fig.suptitle(f"{data['operation']} - Z Control Analysis", fontsize=14, fontweight='bold')
            axes = axes.reshape(-1, 1)  # Make it 2D for consistent indexing
        
        # Plot XY tracking
        if data['has_xy']:
            # XY Position tracking
            ax = axes[0, 0] if data['has_z'] or data['has_xy'] else axes[0]
            ax.plot(time, data['pos_x'], 'b-', label='X Position', linewidth=2)
            ax.plot(time, data['goal_x'], 'b--', label='X Goal', linewidth=1.5, alpha=0.7)
            ax.plot(time, data['pos_y'], 'r-', label='Y Position', linewidth=2)
            ax.plot(time, data['goal_y'], 'r--', label='Y Goal', linewidth=1.5, alpha=0.7)
            ax.set_xlabel('Time (steps)')
            ax.set_ylabel('Position (m)')
            ax.set_title('XY Position Tracking')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # XY Error
            ax = axes[1, 0] if data['has_z'] or data['has_xy'] else axes[1]
            ax.plot(time, data['error_x'], 'b-', label='X Error', linewidth=2)
            ax.plot(time, data['error_y'], 'r-', label='Y Error', linewidth=2)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.set_xlabel('Time (steps)')
            ax.set_ylabel('Error (m)')
            ax.set_title('XY Tracking Error')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        # Plot Z tracking
        if data['has_z']:
            col_idx = 1 if data['has_xy'] else 0
            
            # Z Position tracking
            ax = axes[0, col_idx]
            ax.plot(time, data['pos_z'], 'g-', label='Z Position', linewidth=2)
            ax.plot(time, data['goal_z'], 'g--', label='Z Goal', linewidth=1.5, alpha=0.7)
            ax.set_xlabel('Time (steps)')
            ax.set_ylabel('Height (m)')
            ax.set_title('Z Position Tracking')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # Z Error
            ax = axes[1, col_idx]
            ax.plot(time, data['error_z'], 'g-', label='Z Error', linewidth=2)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.set_xlabel('Time (steps)')
            ax.set_ylabel('Error (m)')
            ax.set_title('Z Tracking Error')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        # Plot PID terms
        if data['has_xy'] and data['has_z']:
            # XY PID terms
            ax = axes[2, 0]
            ax.plot(time, data['p_term_x'], label='P (X)', linewidth=1.5)
            ax.plot(time, data['i_term_x'], label='I (X)', linewidth=1.5)
            ax.plot(time, data['d_term_x'], label='D (X)', linewidth=1.5)
            ax.set_xlabel('Time (steps)')
            ax.set_ylabel('Control Signal')
            ax.set_title('XY PID Terms')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # Z PID terms
            ax = axes[2, 1]
            ax.plot(time, data['p_term_z'], label='P (Z)', linewidth=1.5)
            ax.plot(time, data['i_term_z'], label='I (Z)', linewidth=1.5)
            ax.plot(time, data['d_term_z'], label='D (Z)', linewidth=1.5)
            ax.set_xlabel('Time (steps)')
            ax.set_ylabel('Control Signal')
            ax.set_title('Z PID Terms')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"{self._plot_counter:03d}_{data['operation']}.png"
        filepath = os.path.join(DEBUG_PLOTS_DIR, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[DEBUG] Saved plot: {filepath}")
        self._plot_counter += 1
        self._debug_data = None

    def _loop(self, predicate, action_fn, max_steps, render=False):
        """
        Generic loop: until predicate(obs) is True or max_steps reached.
        action_fn(obs) -> raw 4D action
        predicate() checks detector grounding for success.
        """
        for step in range(max_steps):
            obs = self.env.env._get_observations()
            if predicate():
                return True, obs
            raw_action = action_fn(obs)
            raw_action[3] = self._grip_cmd
            action = to_osc_pose(raw_action)
            self.record(obs, action)
            obs, *_ = self.env.step(action)
            if render:
                self.env.render()
        return False, obs
    
    def _compute_xy_action(self, track_body_id=None, track_obj_name=None, 
                          target_x=None, target_y=None,
                          add_noise=False, kp=1.5, ki=0.1, kd=0.15, 
                          dt=0.02, integral_limit=0.05, return_debug=False):
        """
        PID controller for XY action toward target object/peg or direct coordinates.
        
        Conservative tuning: PD with small integral to handle steady-state error
        without overshoot. Suitable for both free-space and near-contact motions.
        
        Args:
            track_body_id: Body ID to track (if not using peg position or direct coords)
            track_obj_name: Object name (used for peg lookup or general tracking)
            target_x: Direct X target coordinate (if provided, overrides object tracking)
            target_y: Direct Y target coordinate (if provided, overrides object tracking)
            add_noise: Whether to add randomization noise
            kp: Proportional gain (default: 1.0)
            ki: Integral gain (default: 0.05 - small to prevent overshoot)
            kd: Derivative gain (default: 0.15 - damps oscillations)
            dt: Time step for derivative/integral (default: 0.02s ~50Hz)
            integral_limit: Maximum magnitude for integral term (anti-windup)
            return_debug: Return debug info (target, error, PID terms)
            
        Returns:
            xy_action: [dx, dy] numpy array (PID output, already capped)
            If return_debug=True: (xy_action, target_xy, error, p_term, i_term, d_term)
        """
        pos = self._gripper_pos()
        
        # Compute error (target - current)
        if target_x is not None and target_y is not None:
            # Use direct target coordinates (e.g., for waypoint tracking)
            tgt_xy = np.array([target_x, target_y])
            error = tgt_xy - pos[:2]
        elif track_body_id is not None and track_obj_name is not None:
            # Use peg center map if available, otherwise use body position
            # Check if it's a peg by checking if it's in object_areas
            is_peg = hasattr(self.detector, 'object_areas') and track_obj_name in self.detector.object_areas
            if is_peg and hasattr(self.detector, '_get_area_pos'):
                # Base HanoiDetector uses _get_area_pos()
                peg_center = self.detector._get_area_pos(track_obj_name)
                tgt_xy = np.array(peg_center[:2])
            elif is_peg and hasattr(self.detector, '_get_peg_target_position'):
                # PandaHanoiDetector uses _get_peg_target_position()
                peg_center = self.detector._get_peg_target_position(track_obj_name)
                tgt_xy = np.array(peg_center[:2])
            else:
                tgt = np.array(self.env.sim.data.body_xpos[track_body_id])
                tgt_xy = tgt[:2]
            
            error = tgt_xy - pos[:2]
        else:
            tgt_xy = pos[:2]  # No target
            error = np.zeros(2, dtype=float)
        
        # Proportional term
        p_term = kp * error
        
        # Integral term (with anti-windup)
        self._xy_integral += error * dt
        # Clamp integral to prevent windup
        self._xy_integral = np.clip(self._xy_integral, -integral_limit, integral_limit)
        i_term = ki * self._xy_integral
        
        # Derivative term
        d_term = kd * (error - self._xy_prev_error) / dt
        self._xy_prev_error = error.copy()
        
        # PID output
        pid_output = p_term + i_term + d_term
        
        # Add noise if enabled
        if add_noise and self.randomize:
            sigma = self.noise_std * float(np.linalg.norm(pid_output))
            if sigma > 0.0:
                noise_xy = np.random.normal(0.0, sigma, size=2)
                pid_output += noise_xy
        
        if return_debug:
            return pid_output, tgt_xy, error, p_term, i_term, d_term
        return pid_output
    
    def _compute_z_action(self, target_z, direction='up', add_noise=False,
                         kp=2.0, ki=0.2, kd=0.1, dt=0.05, integral_limit=0.05,
                         return_debug=False):
        """
        PID controller for Z action toward target height.
        
        Conservative tuning: PD controller (no integral) for vertical motion.
        Good damping to prevent bouncing during contact.
        
        Args:
            target_z: Target Z coordinate
            direction: 'up' for ascending, 'down' for descending
            add_noise: Whether to add randomization noise
            kp: Proportional gain (default: 1.0)
            ki: Integral gain (default: 0.0 - no integral for Z)
            kd: Derivative gain (default: 0.2 - good damping)
            dt: Time step for derivative/integral (default: 0.02s ~50Hz)
            integral_limit: Maximum magnitude for integral term (anti-windup)
            return_debug: Return debug info (error, PID terms)
            
        Returns:
            dz: Scalar Z action (PID output, already capped and sign-constrained)
            If return_debug=True: (dz, error, p_term, i_term, d_term)
        """
        current_z = float(self._gripper_pos()[2])
        
        # Compute error (target - current)
        error = float(target_z - current_z)
        
        # Proportional term
        p_term = kp * error
        
        # Integral term (with anti-windup)
        self._z_integral += error * dt
        # Clamp integral to prevent windup
        self._z_integral = np.clip(self._z_integral, -integral_limit, integral_limit)
        i_term = ki * self._z_integral
        
        # Derivative term
        d_term = kd * (error - self._z_prev_error) / dt
        self._z_prev_error = error
        
        # PID output
        pid_output = p_term + i_term + d_term
        
        dz = np.array([pid_output])[0]
        
        # Add noise if enabled and moving
        if add_noise and self.randomize and abs(dz) > 0.0:
            sigma = self.noise_std * abs(dz)
            z_noise = float(np.random.normal(0.0, sigma))
            dz += z_noise
        
        # Enforce direction constraint
        if direction == 'up' and dz < 0.0:
            dz = 0.0
        elif direction == 'down' and dz > 0.0:
            dz = 0.0
        
        if return_debug:
            return dz, error, p_term, i_term, d_term
        return dz
    
    def _ascend(self, target_z, speed=10.0, max_steps=500):
        """Move gripper vertically upward. Fast in free space."""
        self._reset_z_pid()
        self._init_debug_data("ascend", has_xy=False, has_z=True)
        
        step_counter = {'i': 0}
        
        def predicate():
            return float(self._gripper_pos()[2]) >= target_z
        
        def action_fn(obs):
            if DEBUG_PLOTS and self._debug_data is not None:
                dz, error_z, p_z, i_z, d_z = self._compute_z_action(
                    target_z, direction='up', add_noise=False, return_debug=True
                )
                
                # Record debug data
                pos = self._gripper_pos()
                self._record_debug_step(
                    timestep=step_counter['i'],
                    pos=pos,
                    goal=np.array([pos[0], pos[1], target_z]),
                    error=np.array( [0.0, 0.0, error_z]),
                    p_term=np.array([0.0, 0.0, p_z]),
                    i_term=np.array([0.0, 0.0, i_z]),
                    d_term=np.array([0.0, 0.0, d_z]),
                    action=np.array([0.0, 0.0, dz])
                )
            else:
                dz = self._compute_z_action(target_z, direction='up', add_noise=False)
            
            move = np.array([0.0, 0.0, dz, self._grip_cmd])
            step_counter['i'] += 1
            return move * speed
        
        result = self._loop(predicate, action_fn, max_steps)
        self._plot_debug_data()
        return result

    def _descend(self, target_z, speed=10.0, max_steps=500,
                 track_body_id=None, track_obj_name=None):
        """
        Descend while tracking target in XY. Slower speed for stability near contact.
        XY tracking compensates for drift; noise disabled for stability.
        """
        self._reset_xy_pid()
        self._reset_z_pid()
        obj_name = track_obj_name if track_obj_name else "target"
        self._init_debug_data(f"descend_{obj_name}", has_xy=True, has_z=True)
        
        step_counter = {'i': 0}

        def predicate():
            return float(self._gripper_pos()[2]) <= target_z

        def action_fn(obs):
            if DEBUG_PLOTS and self._debug_data is not None:
                # Z: descend toward target
                dz, error_z, p_z, i_z, d_z = self._compute_z_action(
                    target_z, direction='down', add_noise=False, return_debug=True
                )
                
                # XY: track target (no noise for stability during descent)
                dist_xy, tgt_xy, error_xy, p_xy, i_xy, d_xy = self._compute_xy_action(
                    track_body_id=track_body_id,
                    track_obj_name=track_obj_name,
                    add_noise=False,
                    return_debug=True
                )
                
                # Record debug data
                pos = self._gripper_pos()
                self._record_debug_step(
                    timestep=step_counter['i'],
                    pos=pos,
                    goal=np.array([tgt_xy[0], tgt_xy[1], target_z]),
                    error=np.array([error_xy[0], error_xy[1], error_z]),
                    p_term=np.array([p_xy[0], p_xy[1], p_z]),
                    i_term=np.array([i_xy[0], i_xy[1], i_z]),
                    d_term=np.array([d_xy[0], d_xy[1], d_z]),
                    action=np.array([dist_xy[0], dist_xy[1], dz])
                )
            else:
                # Z: descend toward target
                dz = self._compute_z_action(target_z, direction='down', add_noise=False)
                
                # XY: track target (no noise for stability during descent)
                dist_xy = self._compute_xy_action(
                    track_body_id=track_body_id,
                    track_obj_name=track_obj_name,
                    add_noise=False
                )
            
            move = np.array([dist_xy[0], dist_xy[1], dz, self._grip_cmd])
            step_counter['i'] += 1
            return move * speed

        result = self._loop(predicate, action_fn, max_steps)
        self._plot_debug_data()
        return result

    def _descend_xy_until_on(self, pick_str, goal_str, target_z, speed=10.0, max_steps=500,
                              track_body_id=None, track_obj_name=None):
        """
        Descend while tracking XY until on(pick, goal) detected.
        Very slow and stable for precise placement.
        """
        self._reset_xy_pid()
        self._reset_z_pid()
        obj_name = track_obj_name if track_obj_name else goal_str
        self._init_debug_data(f"place_{pick_str}_on_{goal_str}", has_xy=True, has_z=True)
        
        step_counter = {'i': 0}

        def predicate():
            st = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            return bool(st.get(f"on({pick_str},{goal_str})", False))

        def action_fn(obs):
            if DEBUG_PLOTS and self._debug_data is not None:
                # Z: descend toward target
                dz, error_z, p_z, i_z, d_z = self._compute_z_action(
                    target_z, direction='down', add_noise=False, return_debug=True
                )
                
                # XY: track target (no noise for stability during placement)
                dist_xy, tgt_xy, error_xy, p_xy, i_xy, d_xy = self._compute_xy_action(
                    track_body_id=track_body_id,
                    track_obj_name=track_obj_name,
                    add_noise=False,
                    return_debug=True
                )
                
                # Record debug data
                pos = self._gripper_pos()
                self._record_debug_step(
                    timestep=step_counter['i'],
                    pos=pos,
                    goal=np.array([tgt_xy[0], tgt_xy[1], target_z]),
                    error=np.array([error_xy[0], error_xy[1], error_z]),
                    p_term=np.array([p_xy[0], p_xy[1], p_z]),
                    i_term=np.array([i_xy[0], i_xy[1], i_z]),
                    d_term=np.array([d_xy[0], d_xy[1], d_z]),
                    action=np.array([dist_xy[0], dist_xy[1], dz])
                )
            else:
                # Z: descend toward target
                dz = self._compute_z_action(target_z, direction='down', add_noise=False)
                
                # XY: track target (no noise for stability during placement)
                dist_xy = self._compute_xy_action(
                    track_body_id=track_body_id,
                    track_obj_name=track_obj_name,
                    add_noise=False
                )
            
            move = np.array([dist_xy[0], dist_xy[1], dz, self._grip_cmd])
            step_counter['i'] += 1
            return move * speed

        result = self._loop(predicate, action_fn, max_steps)
        self._plot_debug_data()
        return result

    def _move_xy(self, body_id, obj_name, speed=10.0, max_steps=500):
        """
        Center gripper over target in XY plane. Moderate speed for accuracy.
        """
        self._reset_xy_pid()
        self._init_debug_data(f"move_xy_{obj_name}", has_xy=True, has_z=False)
        
        step_counter = {'i': 0}
        
        def predicate():
            state = self.detector.get_groundings(
                as_dict=True, binary_to_float=False, return_distance=False
            )
            return state.get(f"over(gripper,{obj_name})", False)

        def action_fn(obs):
            # XY: move toward target (with noise for exploration)
            if DEBUG_PLOTS and self._debug_data is not None:
                dist_xy, tgt_xy, error_xy, p_xy, i_xy, d_xy = self._compute_xy_action(
                    track_body_id=body_id,
                    track_obj_name=obj_name,
                    add_noise=False,
                    return_debug=True
                )
                # Record debug data
                pos = self._gripper_pos()
                self._record_debug_step(
                    timestep=step_counter['i'],
                    pos=pos,
                    goal=np.array([tgt_xy[0], tgt_xy[1], pos[2]]),
                    error=np.array([error_xy[0], error_xy[1], 0.0]),
                    p_term=np.array([p_xy[0], p_xy[1], 0.0]),
                    i_term=np.array([i_xy[0], i_xy[1], 0.0]),
                    d_term=np.array([d_xy[0], d_xy[1], 0.0]),
                    action=np.array([dist_xy[0], dist_xy[1], 0.0])
                )
            else:
                dist_xy = self._compute_xy_action(
                    track_body_id=body_id,
                    track_obj_name=obj_name,
                    add_noise=False
                )
            
            move = np.array([dist_xy[0], dist_xy[1], 0.0, self._grip_cmd])
            step_counter['i'] += 1
            return move * speed

        result = self._loop(predicate, action_fn, max_steps)
        self._plot_debug_data()
        return result

    def _move_xy_object(self, body_id, obj_name, speed=7.5, max_steps=500):
        """Center the gripper over an OBJECT in XY using the object's live body pose.
        Ignores any peg target map to guarantee we're not using a peg center.
        Prints detailed debug every ~20 steps and near convergence.
        """
        step = {"i": 0}

        def predicate():
            # Use detector.over distance if available for consistent measurement
            dist = None
            over_flag = False
            try:
                dist = float(self.detector.over('gripper', obj_name, return_distance=True))
                over_flag = bool(self.detector.over('gripper', obj_name))
            except Exception:
                pos = self._gripper_pos()[:2]
                tgt = np.array(self.env.sim.data.body_xpos[body_id])[:2]
                dist = float(np.linalg.norm(tgt - pos))
                # Fallback to state dict boolean
                state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
                over_flag = bool(state.get(f"over(gripper,{obj_name})", False))
            return over_flag

        def action_fn(obs):
            pos3 = self._gripper_pos()
            tgt3 = np.array(self.env.sim.data.body_xpos[body_id])  # force body pos
            dist_xy = tgt3[:2] - pos3[:2]
            # dist_xy = cap(dist_xy)
            move = np.array([dist_xy[0], dist_xy[1], 0.0, 0.0])
            if self.randomize:
                sigma = self.noise_std * float(np.linalg.norm(dist_xy))
                if sigma > 0.0:
                    move[:2] += np.random.normal(0.0, sigma, size=2)
            move[3] = self._grip_cmd
            step["i"] += 1
            return move * speed

        return self._loop(predicate, action_fn, max_steps)
    
    def _gripper_actuate(self, open_grip=True, max_steps=50):
        """Open or close the gripper."""
        pred = 'open_gripper' if open_grip else 'grasped'

        def predicate():
            st = self.detector.get_groundings(as_dict=True,
                                              binary_to_float=False,
                                              return_distance=False)
            if open_grip:
                return bool(st.get(f"{pred}(gripper)", False))
            else:
                return any(v for k, v in st.items() if k.startswith("grasped(") and v)

        sign = self._open_sign if open_grip else -self._open_sign
        
        def action_fn(obs):
            return np.array([0, 0, 0, sign])

        self._grip_cmd = sign
        return self._loop(predicate, action_fn, max_steps)
    
    def _move_xy_parabolic(
        self, target_id, target_name, place=False, speed=10.0, max_steps=500
    ):
        """ Move the gripper over the object or placement in a parabolic path.
        """
        step = {"i": 0}
        # Store the noisy goal position for bins (computed once, used consistently)
        noisy_goal_xy = None
        if target_name.startswith(('bin', 'platform')) and place and self.args.bin_placement_noise > 0:
            base_goal = np.array(self.env.sim.data.body_xpos[target_id])[:2]
            noise = self.args.bin_placement_noise
            if noise > 0:
                noise_x = np.random.uniform(-noise, noise)
                noise_y = np.random.uniform(-noise, noise)
                noisy_goal_xy = base_goal + np.array([noise_x, noise_y])
            else:
                noisy_goal_xy = base_goal.copy()

        def predicate():
            # Use detector.over distance if available for consistent measurement
            dist = None
            over_flag = False
            try:
                if target_name.startswith(('bin', 'platform')):
                    # For bins during placement, check distance to noisy goal position
                    pos = self._gripper_pos()[:2]
                    dist_xy = np.linalg.norm(pos - noisy_goal_xy)
                    threshold = 0.005
                    over_flag = bool(dist_xy < threshold)
                else:
                    over_flag = bool(self.detector.over('gripper', target_name))
            except Exception:
                pos = self._gripper_pos()[:2]
                tgt = np.array(self.env.sim.data.body_xpos[target_id])[:2]
                dist = float(np.linalg.norm(tgt - pos))
                # Fallback to state dict boolean
                state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
                over_flag = bool(state.get(f"over(gripper,{target_name})", False))
            return over_flag

        def generate_parabolic_path(target_id, num_waypts=50, place=False):
            """Generate a parabolic path for the gripper to move over the objec
            or placement.
            """
            start_pos = self._gripper_pos()
            # Use peg center if available, otherwise use body position
            # Check if it's a peg by checking if it's in object_areas
            is_peg = hasattr(self.detector, 'object_areas') and target_name in self.detector.object_areas
            if is_peg and hasattr(self.detector, '_get_area_pos'):
                # Base HanoiDetector uses _get_area_pos()
                peg_center = self.detector._get_area_pos(target_name)
                peg_z = self.env.sim.data.body_xpos[target_id][2] + 0.0375
                goal_pos = np.array([peg_center[0], peg_center[1], peg_z])
            elif is_peg and hasattr(self.detector, '_get_peg_target_position'):
                # PandaHanoiDetector uses _get_peg_target_position()
                peg_center = self.detector._get_peg_target_position(target_name)
                peg_z = self.env.sim.data.body_xpos[target_id][2] + 0.0375
                goal_pos = np.array([peg_center[0], peg_center[1], peg_z])
            else:
                goal_pos = np.array(self.env.sim.data.body_xpos[target_id])
                if place:
                    goal_pos[2] = goal_pos[2] + 0.05
                    if target_name.startswith(('bin', 'platform')):
                        noise = self.args.bin_placement_noise
                        if noise > 0:
                            goal_pos[0] = noisy_goal_xy[0]
                            goal_pos[1] = noisy_goal_xy[1]
                        goal_pos[2] += 0.07
                    
            start_x, start_y, start_z = start_pos
            goal_x, goal_y, goal_z = goal_pos

            waypoints = []
            for i in range(num_waypts):
                t = i / (num_waypts - 1)

                # Linear XY
                wp_x = start_x + (goal_x - start_x) * t
                wp_y = start_y + (goal_y - start_y) * t

                # Quadratic Bezier curve Z
                start_bias = (1 - t)**2 * start_z
                mid_bias = 2*(1 - t)*t * MAX_HEIGHT
                end_bias = t**2 * goal_z

                wp_z = start_bias + mid_bias + end_bias
                
                waypoints.append(np.array([wp_x, wp_y, wp_z]))
            
            return waypoints

        # Precompute waypoints once, before the loop
        waypoints = generate_parabolic_path(target_id, place=place)
        num_waypts = len(waypoints)

        def action_fn(obs):
            pos3 = self._gripper_pos()

            # Clamp index so once we hit the last waypoint we just keep tracking it
            idx = min(step["i"] // 2, num_waypts - 1)
            target = waypoints[idx]

            # Move toward this waypoint
            delta = target - pos3  # [dx, dy, dz]
            delta_z = np.array([delta[2]])[0]
            if delta_z > 0.0:
                dir = 'up'
            else:
                dir = 'down'
            # move = np.array([delta_xy[0], delta_xy[1], delta_z, 0.0])
            # Use waypoint coordinates (which now use peg center for pegs)
            move_xy = self._compute_xy_action(
                target_x=target[0],
                target_y=target[1],
                # add_noise=True,
                # return_debug=True
            )
            move_z = self._compute_z_action(
                target_z=target[2],
                direction=dir,
                # add_noise=True,
                # return_debug=True
            )

            move = np.array([move_xy[0], move_xy[1], move_z, 0.0])
            move = move * speed

            if self.randomize:
                sigma = self.noise_std * float(np.linalg.norm(move[:3]))
                if sigma > 0.0:
                    move[:3] += np.random.normal(0.0, sigma, size=3)

            move[3] = self._grip_cmd

            step["i"] += 1
            return move

        return self._loop(predicate, action_fn, max_steps)

class PickOperation(TaskOperation):
    """Pick up the specified object."""
    def __init__(self, args, env, detector, randomize, noise_std, object_id):
        super().__init__(args, env, detector, randomize, noise_std)
        self.object_id = object_id
        self.body_id = env.sim.model.body_name2id(detector.object_id[object_id])

    def execute(self, obs):
        # Compute a dynamic hover height: 10 cm above the object
        offset = 0.0075
        object_z = float(self.env.sim.data.body_xpos[self.body_id][2]) + offset
        ref_z = MAX_HEIGHT #object_z + 0.10

        # 1) Open the gripper
        ok, obs = self._gripper_actuate(open_grip=True)
        if not ok:
            print("[PickOperation] ❌ failed to open gripper")
            return False, obs

        if PARABOLIC_PATHS:
            # 2) Move over the object
            ok, obs = self._move_xy_parabolic(self.body_id, self.object_id)
            if not ok:
                print(f"[Pick] ❌ failed parabolic move to {self.object_id}")
                return False, obs

        else:
            # 2) Move up to hover
            ok, obs = self._ascend(target_z=ref_z)
            if not ok:
                print(f"[PickOperation] ❌ failed to move above object at z={ref_z}")
                return False, obs

            # 2.5) Center in XY over the object
            ok, obs = self._move_xy_object(self.body_id, self.object_id)
            if not ok:
                print(f"[PickOperation] ❌ failed to XY-center over {self.object_id}")
                return False, obs

            # 3) Descend straight down onto the object
            ok, obs = self._descend(
                target_z=object_z,
                track_body_id=self.body_id,
                track_obj_name=self.object_id,
            )
            if not ok:
                print(f"[PickOperation] ❌ failed to descend to z={object_z}")
                return False, obs

        # 4) Close the gripper
        ok, obs = self._gripper_actuate(open_grip=False)
        if not ok:
            print("[PickOperation] ❌ failed to close on object")
            return False, obs

        # 5) Keep closing for a few more steps to ensure full closure
        EXTRA_CLOSE_STEPS = 10
        for _ in range(EXTRA_CLOSE_STEPS):
            obs = self.env.env._get_observations()
            action = to_osc_pose(np.array([0, 0, 0, -self._open_sign]))
            self.record(obs, action)
            obs, *_ = self.env.step(action)

        print(f"[PickOperation] ✅ picked {self.object_id}")
        return True, obs
    

class PlaceOperation(TaskOperation):
    """Place a held object at the target location."""
    def __init__(self, args, env, detector, randomize, noise_std, object_id, placement_id):
        super().__init__(args, env, detector, randomize, noise_std)
        self._grip_cmd = -self._open_sign
        self.object_id = object_id
        self.placement_id = placement_id 
        self.body_id = env.sim.model.body_name2id(detector.object_id[placement_id])

    def execute(self, obs):

        if PARABOLIC_PATHS:
            # 1) Move over placement location
            ok, obs = self._move_xy_parabolic(self.body_id, self.placement_id, place=True)
            if not ok:
                print(f"[Place] ❌ failed parabolic move to {self.placement_id}")
                return False, obs
                
        else:
            # Ascend to transport height
            ok, obs = self._ascend(target_z=MAX_HEIGHT)
            if not ok: 
                print(f"[PlaceOperation] ❌ failed to ascend to {MAX_HEIGHT}")
                return False, obs
            
            # Move over placement
            ok, obs = self._move_xy(self.body_id, self.placement_id, max_steps=1000)
            if not ok: 
                print(f"[PlaceOperation] ❌ failed to move over {self.placement_id}")
                return False, obs

            # Descend until on(pick, goal) detected
            place_z = float(self.env.sim.data.body_xpos[self.body_id][2])
            ok, obs = self._descend_xy_until_on(
                pick_str=self.object_id,
                goal_str=self.placement_id,
                target_z=place_z,
                track_body_id=self.body_id,
                track_obj_name=self.placement_id,
            )
            if not ok:
                print(f"[PlaceOperation] ❌ failed to descend until on({self.object_id},{self.placement_id})")
                return False, obs
        
        # Open to release
        ok, obs = self._gripper_actuate(open_grip=True)
        if not ok: 
            print(f"[PlaceOperation] ❌ failed to open gripper")
            return False, obs

        print(f"[PlaceOperation] ✅ placed {self.object_id} on {self.placement_id}")
        return True, obs
    

class TurnOnOperation(TaskOperation):
    """Switch a binary button on (e.g., stove)."""
    def __init__(self, args, env, detector, randomize, noise_std, object_id):
        super().__init__(args, env, detector, randomize, noise_std)
        self.object_id = object_id
        self.body_id = env.sim.model.body_name2id(detector.object_id[object_id])

    def execute(self, obs):
        # Move above
        ok, obs = self._ascend(target_z=MAX_HEIGHT)
        if not ok: return False, obs
        ok, obs = self._move_xy(self.body_id, self.object_id)
        if not ok: return False, obs
        # Descend to switch
        switch_z = float(self.env.sim.data.body_xpos[self.body_id][2])
        ok, obs = self._ascend(target_z=switch_z)
        if not ok: return False, obs
        # Actuate on
        ok, obs = self._gripper_actuate(open_grip=False)
        if not ok: return False, obs
        # Retract
        retract_height = self._gripper_pos()[2] + 0.05  # Lift 5cm
        ok, obs = self._ascend(target_z=retract_height)
        if not ok:
            print(f"[TurnOnOperation] ❌ failed to retract after actuation")
            return False, obs
        return True, obs
    

class TurnOffOperation(TurnOnOperation):
    """Switch a binary button off."""
    def execute(self, obs):
        # Same as TurnOn but open_grip inverted for actuation
        return super().execute(obs)
