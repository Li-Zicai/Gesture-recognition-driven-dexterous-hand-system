import os
import time
import numpy as np
import yaml
from typing import Dict, List, Union, Optional
from collections import deque
from threading import RLock

from .client3 import BMotorClient

class OrcaHand:
    """ORCA Hand controller for SCServo motors"""
    
    def __init__(self, model_path: str = None):
        """Initialize the OrcaHand controller"""
        
        # Get model path
        self.model_path = model_path or os.path.join(
            os.path.dirname(__file__), ".."
        )
        
        # Load configuration files
        self.config_path = os.path.join(self.model_path, "config.yaml")
        self.calib_path = os.path.join(self.model_path, "calibration.yaml")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f) or {}
        
        # Initialize calibration file if it doesn't exist
        if not os.path.exists(self.calib_path):
            self._init_calibration_file()
        
        with open(self.calib_path, 'r') as f:
            self.calib = yaml.safe_load(f) or {}
        
        # Extract configuration parameters
        self.motor_ids = self.cfg.get("motor_ids", [])
        self.joint_ids = self.cfg.get("joint_ids", [])
        self.baudrate = self.cfg.get("baudrate", 1000000)
        self.port = self.cfg.get("port", "/dev/ttyUSB0")
        self.max_current = self.cfg.get("max_current", 400)
        self.control_mode = self.cfg.get("control_mode", "current_based_position")
        self.type = self.cfg.get("type", None)
        
        # Calibration parameters
        self.calib_current = self.cfg.get("calib_current", 350)
        self.wrist_calib_current = self.cfg.get("wrist_calib_current", 100)
        self.calib_step_size = self.cfg.get("calib_step_size", 0.1)
        self.calib_step_period = self.cfg.get("calib_step_period", 0.001)
        self.calib_threshold = self.cfg.get("calib_threshold", 0.01)
        self.calib_num_stable = self.cfg.get("calib_num_stable", 10)
        self.calib_sequence = self.cfg.get("calib_sequence", [])
        
        # Joint configuration
        self.joint_to_motor_map = self.cfg.get("joint_to_motor_map", {})
        self.joint_roms = self.cfg.get("joint_roms", {})
        self.neutral_position = self.cfg.get("neutral_position", {})
        
        # Handle inverted motors (negative IDs in config)
        self.joint_inversion = {}
        for joint, motor_id in self.joint_to_motor_map.items():
            if motor_id < 0:
                self.joint_inversion[joint] = True
                self.joint_to_motor_map[joint] = abs(motor_id)
            else:
                self.joint_inversion[joint] = False
        
        self.motor_to_joint_map = {v: k for k, v in self.joint_to_motor_map.items()}
        
        # Load calibration data
        self.calibrated = self.calib.get("calibrated", False)
        self.motor_limits = self.calib.get("motor_limits", {})
        self.joint_to_motor_ratios = self.calib.get("joint_to_motor_ratios", {})
        
        # Initialize motor limits and ratios if not present
        if not self.motor_limits:
            self.motor_limits = {mid: [None, None] for mid in self.motor_ids}
        if not self.joint_to_motor_ratios:
            self.joint_to_motor_ratios = {mid: 0.0 for mid in self.motor_ids}
        
        # Create motor client
        self._motor_client = BMotorClient(
            motor_ids=self.motor_ids,
            port=self.port,
            baudrate=self.baudrate
        )
        
        self._motor_lock = RLock()
        self._sanity_check()

    def _init_calibration_file(self):
        """Initialize calibration file with default values"""
        default_calib = {
            "calibrated": False,
            "motor_limits": {mid: [None, None] for mid in self.motor_ids},
            "joint_to_motor_ratios": {mid: 0.0 for mid in self.motor_ids}
        }
        with open(self.calib_path, 'w') as f:
            yaml.dump(default_calib, f, default_flow_style=False, sort_keys=False)

    def _update_calibration(self, key: str, value):
        """Update calibration file"""
        try:
            with open(self.calib_path, 'r') as f:
                calib = yaml.safe_load(f) or {}
            calib[key] = value
            with open(self.calib_path, 'w') as f:
                yaml.dump(calib, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            print(f"[Calibration Update Error] {e}")

    def connect(self) -> tuple[bool, str]:
        """Connect to the hand"""
        try:
            with self._motor_lock:
                success, msg = self._motor_client.connect()
                if success:
                    # Load calibration if available
                    if self.calibrated:
                        self.load_calibration()
                return success, msg
        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    def disconnect(self) -> tuple[bool, str]:
        """Disconnect from the hand"""
        try:
            with self._motor_lock:
                self.disable_torque()
                time.sleep(0.1)
                self._motor_client.disconnect()
            return True, "Disconnected successfully"
        except Exception as e:
            return False, f"Disconnection failed: {str(e)}"

    def is_connected(self) -> bool:
        """Check if connected"""
        return self._motor_client.is_connected() if self._motor_client else False

    def is_calibrated(self) -> bool:
        """Check if calibrated"""
        for limits in self.motor_limits.values():
            if any(limit is None or limit == 0 for limit in limits):
                return False
        return True

    def enable_torque(self, motor_ids: List[int] = None):
        """Enable torque for motors"""
        if motor_ids is None:
            motor_ids = self.motor_ids
        with self._motor_lock:
            self._motor_client.set_torque_enabled(motor_ids, True)

    def disable_torque(self, motor_ids: List[int] = None):
        """Disable torque for motors"""
        if motor_ids is None:
            motor_ids = self.motor_ids
        with self._motor_lock:
            self._motor_client.set_torque_enabled(motor_ids, False)

    def set_max_current(self, current: Union[float, List[float]]):
        """Set maximum current for motors"""
        if isinstance(current, list):
            if len(current) != len(self.motor_ids):
                raise ValueError("Number of currents must match number of motors")
            currents = np.array(current)
        else:
            currents = np.ones(len(self.motor_ids)) * current
        
        with self._motor_lock:
            self._motor_client.write_desired_current(self.motor_ids, currents)

    def set_control_mode(self, mode: str, motor_ids: List[int] = None):
        """Set control mode for motors"""
        mode_map = {
            'current': 0,
            'velocity': 1,
            'position': 3,
            'multi_turn_position': 4,
            'current_based_position': 5
        }
        
        mode_value = mode_map.get(mode)
        if mode_value is None:
            raise ValueError(f"Invalid control mode: {mode}")
        
        if motor_ids is None:
            motor_ids = self.motor_ids
        
        with self._motor_lock:
            self._motor_client.set_operating_mode(motor_ids, mode_value)

    def get_motor_pos(self, as_dict: bool = False) -> Union[np.ndarray, dict]:
        """Get motor positions in radians"""
        with self._motor_lock:
            positions, _, _ = self._motor_client.read_pos_vel_cur()
            if as_dict:
                return {mid: pos for mid, pos in zip(self.motor_ids, positions)}
            return positions

    def get_motor_current(self, as_dict: bool = False) -> Union[np.ndarray, dict]:
        """Get motor currents in mA"""
        with self._motor_lock:
            _, _, currents = self._motor_client.read_pos_vel_cur()
            if as_dict:
                return {mid: cur for mid, cur in zip(self.motor_ids, currents)}
            return currents

    def get_motor_temp(self, as_dict: bool = False) -> Union[np.ndarray, dict]:
        """Get motor temperatures in Celsius"""
        with self._motor_lock:
            temps = self._motor_client.read_temperature()
            if as_dict:
                return {mid: temp for mid, temp in zip(self.motor_ids, temps)}
            return temps

    def get_joint_pos(self, as_list: bool = True) -> Union[dict, list]:
        """Get joint positions in degrees"""
        motor_pos = self.get_motor_pos()
        joint_pos = self._motor_to_joint_pos(motor_pos)
        
        if as_list:
            return [joint_pos.get(joint, 0.0) for joint in self.joint_ids]
        return joint_pos

    def set_joint_pos(self, joint_pos: Union[dict, list], num_steps: int = 1, step_size: float = 1.0,
                 speed: int = 60, torque: int = 600, acceleration: int = 50):
        """Set joint positions with configurable motion parameters"""
        if num_steps > 1:
            current_positions = self.get_joint_pos(as_list=False)
            
            if isinstance(joint_pos, list):
                if len(joint_pos) != len(self.joint_ids):
                    raise ValueError("Length of joint_pos must match number of joints")
                target_positions = {joint: pos for joint, pos in zip(self.joint_ids, joint_pos)}
            else:
                target_positions = joint_pos.copy()
            
            # Interpolate positions
            for step in range(num_steps + 1):
                t = step / num_steps
                interpolated = {}
                for joint in self.joint_ids:
                    if joint in target_positions:
                        current = current_positions.get(joint, 0)
                        target = target_positions[joint]
                        interpolated[joint] = current * (1 - t) + target * t
                    else:
                        interpolated[joint] = current_positions.get(joint, 0)
                
                motor_pos = self._joint_to_motor_pos(interpolated)
                self._set_motor_pos(motor_pos, speed=speed, torque=torque, acceleration=acceleration)
                
                if step < num_steps:
                    time.sleep(step_size)
        else:
            if isinstance(joint_pos, list):
                if len(joint_pos) != len(self.joint_ids):
                    raise ValueError("Length of joint_pos must match number of joints")
                joint_pos_dict = {joint: pos for joint, pos in zip(self.joint_ids, joint_pos)}
            elif isinstance(joint_pos, dict):
                joint_pos_dict = joint_pos
            else:
                raise ValueError("joint_pos must be a dict or list")
            
            motor_pos = self._joint_to_motor_pos(joint_pos_dict)
            self._set_motor_pos(motor_pos, speed=speed, torque=torque, acceleration=acceleration)

    def set_zero_position(self, num_steps: int = 25, step_size: float = 0.001,
                        speed: int = 60, torque: int = 500, acceleration: int = 50):
        """Move to zero position with configurable motion parameters"""
        self.set_joint_pos({joint: 0 for joint in self.joint_ids}, 
                        num_steps=num_steps, step_size=step_size,
                        speed=speed, torque=torque, acceleration=acceleration)

    def set_neutral_position(self, num_steps: int = 25, step_size: float = 0.001,
                            speed: int = 40, torque: int = 400, acceleration: int = 50):
        """Move to neutral position with configurable motion parameters"""
        if not self.neutral_position:
            raise ValueError("Neutral position not defined in config")
        self.set_joint_pos(self.neutral_position, num_steps=num_steps, step_size=step_size,
                        speed=speed, torque=torque, acceleration=acceleration)

    def init_joints(self, calibrate: bool = False):
        """Initialize joints"""
        self.enable_torque()
        self.set_control_mode(self.control_mode)
        self.set_max_current(self.max_current)
        
        if not self.calibrated or calibrate:
            self.calibrate()
        
        # self.set_neutral_position() # please uncomment this line after setting the correct neutral points 
        return 0  # Return success

    def calibrate_manual(self):
        """Manual calibration by physically moving joints"""
        if not self.is_connected():
            raise RuntimeError("Hand not connected")
        
        print("[Manual Calibration] Starting manual calibration...")
        print("Move each joint to its physical limits.")
        print("Press Enter when done.")
        
        # Disable all torques for manual movement
        self.disable_torque()
        time.sleep(0.5)
        
        # Initialize tracking dictionaries
        motor_min_positions = {mid: float('inf') for mid in self.motor_ids}
        motor_max_positions = {mid: float('-inf') for mid in self.motor_ids}
        
        # Start tracking positions
        try:
            while True:
                # Read current positions
                current_positions = self.get_motor_pos(as_dict=True)
                
                # Update min/max for each motor
                for motor_id, pos in current_positions.items():
                    if pos < motor_min_positions[motor_id]:
                        motor_min_positions[motor_id] = pos
                    if pos > motor_max_positions[motor_id]:
                        motor_max_positions[motor_id] = pos
                
                # Display current ranges
                print("\rCurrent ranges: ", end="")
                for motor_id in self.motor_ids:
                    joint = self.motor_to_joint_map.get(motor_id, f"Motor{motor_id}")
                    min_pos = motor_min_positions[motor_id]
                    max_pos = motor_max_positions[motor_id]
                    if min_pos != float('inf') and max_pos != float('-inf'):
                        print(f"{joint}: [{min_pos:.3f}, {max_pos:.3f}] ", end="")
                
                # Check for Enter key (non-blocking)
                import select
                import sys
                if select.select([sys.stdin], [], [], 0)[0]:
                    line = input()
                    break
                
                time.sleep(0.05)  # Update rate
                
        except KeyboardInterrupt:
            print("\n[Manual Calibration] Interrupted by user")
        
        print("\n[Manual Calibration] Recording limits...")
        
        # Store the limits
        motor_limits = {}
        for motor_id in self.motor_ids:
            if motor_min_positions[motor_id] != float('inf') and motor_max_positions[motor_id] != float('-inf'):
                motor_limits[motor_id] = [
                    float(motor_min_positions[motor_id]),
                    float(motor_max_positions[motor_id])
                ]
            else:
                motor_limits[motor_id] = [None, None]
                print(f"Warning: Motor {motor_id} was not moved during calibration")
        
        # Calculate joint-to-motor ratios
        joint_to_motor_ratios = {}
        for motor_id, limits in motor_limits.items():
            if None not in limits and limits[0] != limits[1]:
                delta_motor = limits[1] - limits[0]
                joint = self.motor_to_joint_map.get(motor_id)
                if joint and joint in self.joint_roms:
                    delta_joint = self.joint_roms[joint][1] - self.joint_roms[joint][0]
                    joint_to_motor_ratios[motor_id] = float(delta_motor / delta_joint)
                else:
                    joint_to_motor_ratios[motor_id] = 0.0
            else:
                joint_to_motor_ratios[motor_id] = 0.0
        
        # Update instance variables
        self.motor_limits = motor_limits
        self.joint_to_motor_ratios = joint_to_motor_ratios
        
        # Save to calibration file
        self._update_calibration('motor_limits', motor_limits)
        self._update_calibration('joint_to_motor_ratios', joint_to_motor_ratios)
        
        # Check if fully calibrated
        self.calibrated = self.is_calibrated()
        self._update_calibration('calibrated', self.calibrated)
        
        print(f"[Manual Calibration] Complete. Calibrated: {self.calibrated}")
        
        # Re-enable torque and move to neutral position
        self.enable_torque()
        if self.calibrated:
            print("[Manual Calibration] Moving to neutral position...")
            self.set_neutral_position()

    def calibrate(self):
        """
        Automatic calibration by detecting current limits in position control mode
        Tests each motor individually by moving in small steps until current threshold is reached
        Uses extended range position control for proper limit detection
        """
        if not self.is_connected():
            raise RuntimeError("Hand not connected")
        
        print("[Auto Calibration] Starting automatic calibration...")
        
        # Default calibration parameters
        default_params = {
            'step_size': 0.02,  # rad - step size for incremental movement
            'step_interval': 0.05,  # seconds - time between steps
            'calib_speed': 30,  # speed for calibration movement
            'calib_acceleration': 20,  # acceleration for calibration movement
            'calib_torque': 300,  # torque limit for calibration movement
            'current_threshold': 150,  # mA - current threshold to detect limit
            'stable_readings': 80,  # number of consecutive high current readings to confirm limit
            'max_range': 6.28,  # maximum range to test (2π radians) for safety
        }
        
        # ==================== SPECIAL MOTOR PARAMETERS SECTION ====================
        # Configure special parameters for specific motors here
        # Format: motor_id: {parameter_name: value, ...}
        special_motor_params = {
            17: {
                'stable_readings': 150,  # Motor 17 needs more stable readings
                'current_threshold': 400,  # Motor 17 has higher current threshold
                'calib_speed': 30,  # Slower speed for motor 17
                'calib_torque': 400,  # Higher torque for motor 17
            },
            # Add more special motors here as needed:
            # 5: {
            #     'current_threshold': 200,
            #     'stable_readings': 60,
            # },
            # 12: {
            #     'step_size': 0.005,  # Smaller steps for more precision
            #     'current_threshold': 300,
            # },
        }
        # ============================================================================
        
        # Return motion parameters (same for all motors)
        return_speed = 80  # faster speed for returning to start
        return_acceleration = 30
        return_torque = 400
        
        def get_motor_param(motor_id, param_name):
            """Get parameter value for specific motor, falling back to default"""
            if motor_id in special_motor_params and param_name in special_motor_params[motor_id]:
                return special_motor_params[motor_id][param_name]
            return default_params[param_name]
        
        # Initialize motor limits storage
        motor_limits = {mid: [None, None] for mid in self.motor_ids}
        
        # Set position control mode (mode 0 = servo/position mode)
        with self._motor_lock:
            for motor_id in self.motor_ids:
                result, error = self._motor_client.packetHandler.write1ByteTxRx(
                    motor_id, 33, 0)  # ADDR_MODE = 33, mode 0 = position mode
                if result != 0 or error != 0:
                    print(f"Warning: Failed to set position mode for motor {motor_id}")
        
        # Disable all motors initially
        print("[Auto Calibration] Disabling all motors...")
        self.disable_torque()
        time.sleep(0.5)
        
        # Read initial positions of all motors using extended range
        print("[Auto Calibration] Reading initial positions...")
        initial_positions = {}
        for motor_id in self.motor_ids:
            pos = self._motor_client.read_single_motor_position(motor_id)
            initial_positions[motor_id] = pos
        print(f"Initial positions: {initial_positions}")
        
        # Get calibration sequence from config
        calib_sequence = self.cfg.get("calib_sequence", [])
        if not calib_sequence:
            print("[Auto Calibration] No calibration sequence found in config")
            return
        
        # Process each step in calibration sequence
        for step_idx, step_info in enumerate(calib_sequence):
            step_joints = step_info.get("joints", {})
            print(f"\n[Auto Calibration] Processing step {step_idx + 1}: {step_joints}")
            
            # Process each joint in this step
            for joint_name, direction in step_joints.items():
                if joint_name not in self.joint_to_motor_map:
                    print(f"Warning: Joint {joint_name} not found in joint_to_motor_map")
                    continue
                    
                motor_id = self.joint_to_motor_map[joint_name]
                print(f"\n[Auto Calibration] Testing {joint_name} (Motor {motor_id}) - {direction}")
                
                # Get motor-specific parameters
                step_size = get_motor_param(motor_id, 'step_size')
                step_interval = get_motor_param(motor_id, 'step_interval')
                calib_speed = get_motor_param(motor_id, 'calib_speed')
                calib_acceleration = get_motor_param(motor_id, 'calib_acceleration')
                calib_torque = get_motor_param(motor_id, 'calib_torque')
                current_threshold = get_motor_param(motor_id, 'current_threshold')
                stable_readings = get_motor_param(motor_id, 'stable_readings')
                max_range = get_motor_param(motor_id, 'max_range')
                
                # Display motor-specific parameters if they differ from defaults
                if motor_id in special_motor_params:
                    print(f"  Using SPECIAL parameters for Motor {motor_id}:")
                    for param, value in special_motor_params[motor_id].items():
                        print(f"    {param}: {value}")
                else:
                    print(f"  Using default parameters")
                
                # Enable only this motor
                self.disable_torque()  # Disable all first
                time.sleep(0.1)
                
                # Enable torque for this motor
                with self._motor_lock:
                    result, error = self._motor_client.packetHandler.write1ByteTxRx(
                        motor_id, 40, 1)  # ADDR_TORQUE_ENABLE = 40
                    if result != 0 or error != 0:
                        print(f"Failed to enable torque for motor {motor_id}")
                        continue
                
                time.sleep(0.1)
                
                # Get starting position
                start_position = initial_positions[motor_id]
                current_position = start_position
                
                # Determine movement direction
                # flex = decrease position, extend = increase position
                move_direction = -1 if direction == 'flex' else 1
                
                # Apply joint inversion if needed
                if self.joint_inversion.get(joint_name, False):
                    move_direction *= -1
                
                print(f"  Starting position: {start_position:.3f} rad")
                print(f"  Movement direction: {move_direction} ({'flex' if move_direction < 0 else 'extend'})")
                print(f"  Active parameters: speed={calib_speed}, torque={calib_torque}, acc={calib_acceleration}")
                print(f"  Detection parameters: threshold={current_threshold}mA, stable_readings={stable_readings}")
                
                # Track current readings
                high_current_count = 0
                current_history = []
                limit_found = False
                
                # Move in steps until limit is reached
                step_count = 0
                max_steps = int(max_range / step_size)
                
                while not limit_found and step_count < max_steps:
                    # Calculate next position
                    next_position = current_position + (move_direction * step_size)
                    
                    # Move to next position using extended range write with motion parameters
                    success = self._motor_client.write_single_motor_position(
                        motor_id, next_position, 
                        speed=calib_speed, 
                        torque=calib_torque, 
                        acceleration=calib_acceleration
                    )
                    if not success:
                        print(f"    Failed to write position to motor {motor_id}")
                        break
                    
                    time.sleep(step_interval)
                    
                    # Read current position to verify movement
                    actual_position = self._motor_client.read_single_motor_position(motor_id)
                    
                    # Read current
                    try:
                        motor_current = self._motor_client.read_single_motor_current(motor_id)
                        current_abs = abs(motor_current)
                        current_history.append(current_abs)
                        
                        # Keep only recent history
                        if len(current_history) > 20:
                            current_history.pop(0)
                        
                        # Check if current exceeds threshold
                        if current_abs > current_threshold:
                            high_current_count += 1
                            print(f"    Step {step_count}: target={next_position:.3f}, actual={actual_position:.3f}, current={current_abs:.1f}mA (high #{high_current_count}/{stable_readings})")
                            
                            if high_current_count >= stable_readings:
                                limit_found = True
                                print(f"    *** LIMIT DETECTED *** at position {actual_position:.3f} rad after {stable_readings} stable readings")
                                current_position = actual_position  # Use actual position as limit
                        else:
                            if high_current_count > 0:
                                print(f"    Step {step_count}: target={next_position:.3f}, actual={actual_position:.3f}, current={current_abs:.1f}mA (reset)")
                            high_current_count = 0  # Reset counter if current drops
                        
                        # Update current position
                        current_position = next_position
                        step_count += 1
                        
                    except Exception as e:
                        print(f"    Error reading current: {e}")
                        time.sleep(0.01)
                        continue
                
                # Record the limit
                if limit_found:
                    if direction == 'flex':
                        motor_limits[motor_id][0] = float(current_position)  # min limit
                        print(f"  ✓ Recorded flex limit: {current_position:.3f} rad")
                    else:  # extend
                        motor_limits[motor_id][1] = float(current_position)  # max limit
                        print(f"  ✓ Recorded extend limit: {current_position:.3f} rad")
                else:
                    print(f"  ⚠ Warning: No limit found for {joint_name} in {direction} direction (reached max steps: {max_steps})")
                
                # Return to starting position with faster motion parameters
                print(f"  Returning to start position (speed={return_speed}, torque={return_torque})...")
                steps_back = 20
                for i in range(steps_back):
                    t = (i + 1) / steps_back
                    return_pos = current_position * (1 - t) + start_position * t
                    self._motor_client.write_single_motor_position(
                        motor_id, return_pos,
                        speed=return_speed,
                        torque=return_torque,
                        acceleration=return_acceleration
                    )
                    time.sleep(0.02)
                
                # Disable this motor
                with self._motor_lock:
                    self._motor_client.packetHandler.write1ByteTxRx(motor_id, 40, 0)  # Disable torque
                time.sleep(0.2)
        
        # Calculate joint-to-motor ratios
        print("\n[Auto Calibration] Calculating joint-to-motor ratios...")
        joint_to_motor_ratios = {}
        
        for motor_id, limits in motor_limits.items():
            joint_name = self.motor_to_joint_map.get(motor_id)
            
            if None not in limits and limits[0] != limits[1]:
                delta_motor = limits[1] - limits[0]
                
                if joint_name and joint_name in self.joint_roms:
                    delta_joint = self.joint_roms[joint_name][1] - self.joint_roms[joint_name][0]
                    if delta_joint != 0:
                        joint_to_motor_ratios[motor_id] = float(delta_motor / delta_joint)
                        print(f"  Motor {motor_id} ({joint_name}): range={delta_motor:.3f} rad, ratio={joint_to_motor_ratios[motor_id]:.3f}")
                    else:
                        joint_to_motor_ratios[motor_id] = 0.0
                        print(f"  Warning: Joint {joint_name} has zero range in config")
                else:
                    joint_to_motor_ratios[motor_id] = 0.0
                    print(f"  Warning: Joint ROM not found for motor {motor_id}")
            else:
                joint_to_motor_ratios[motor_id] = 0.0
                print(f"  Warning: Motor {motor_id} calibration incomplete")
        
        # Update instance variables
        self.motor_limits = motor_limits
        self.joint_to_motor_ratios = joint_to_motor_ratios
        
        # Save calibration data to file
        print("\n[Auto Calibration] Saving calibration data...")
        self._update_calibration('motor_limits', motor_limits)
        self._update_calibration('joint_to_motor_ratios', joint_to_motor_ratios)
        
        # Check if fully calibrated
        self.calibrated = self.is_calibrated()
        self._update_calibration('calibrated', self.calibrated)
        
        print(f"\n[Auto Calibration] Complete. Calibrated: {self.calibrated}")
        print(f"Motor limits: {motor_limits}")
        
        # Display summary of special motor configurations used
        if special_motor_params:
            print("\n[Auto Calibration] Special motor configurations used:")
            for motor_id, params in special_motor_params.items():
                if motor_id in self.motor_ids:  # Only show if motor was actually processed
                    joint_name = self.motor_to_joint_map.get(motor_id, f"Motor{motor_id}")
                    print(f"  Motor {motor_id} ({joint_name}): {params}")
        
        # Re-enable all motors and restore normal operation
        self.enable_torque()
        self.set_max_current(self.max_current)
        
        if self.calibrated:
            print("[Auto Calibration] Moving to neutral position...")
            try:
                # Use gentle motion parameters for final neutral position
                self.set_neutral_position(speed=40, torque=300, acceleration=25)
            except Exception as e:
                print(f"Warning: Could not move to neutral position: {e}")

    def load_calibration(self):
        """Load calibration data from file"""
        if not os.path.exists(self.calib_path):
            print("[Load Calibration] Calibration file not found")
            return
        
        with open(self.calib_path, 'r') as f:
            calib = yaml.safe_load(f) or {}
        
        self.calibrated = calib.get('calibrated', False)
        self.motor_limits = calib.get('motor_limits', {})
        self.joint_to_motor_ratios = calib.get('joint_to_motor_ratios', {})
        
        print(f"[Load Calibration] Loaded. Calibrated: {self.calibrated}")

    def _set_motor_pos(self, desired_pos: Union[dict, np.ndarray, list], 
                rel_to_current: bool = False, speed: int = 60, 
                torque: int = 500, acceleration: int = 50):
        """Set motor positions (internal) with motion parameters using synchronized write"""
        with self._motor_lock:
            current_pos = self.get_motor_pos()
            
            # Prepare position dictionary for all motors
            position_dict = {}
            
            if isinstance(desired_pos, dict):
                # If input is dictionary, fill positions for all motors
                for i, mid in enumerate(self.motor_ids):
                    if mid in desired_pos:
                        position_dict[mid] = desired_pos[mid]
                    else:
                        # Keep current position
                        position_dict[mid] = current_pos[i] if not rel_to_current else 0
            elif isinstance(desired_pos, (np.ndarray, list)):
                # If input is array/list, map in order to motor IDs
                motor_pos_array = np.array(desired_pos)
                if len(motor_pos_array) != len(self.motor_ids):
                    raise ValueError(f"Position array length {len(motor_pos_array)} doesn't match motor count {len(self.motor_ids)}")
                
                for i, mid in enumerate(self.motor_ids):
                    position_dict[mid] = motor_pos_array[i]
            else:
                raise ValueError("desired_pos must be dict, ndarray or list")
            
            # Handle relative position
            if rel_to_current:
                for i, mid in enumerate(self.motor_ids):
                    if mid in position_dict:
                        position_dict[mid] += current_pos[i]
            
            # Use synchronized write method to set all motors at once
            self._motor_client.write_positions_with_speed_acc_torque(
                position_dict, speed=speed, acc=acceleration, torque=torque)

    def _motor_to_joint_pos(self, motor_pos: np.ndarray) -> dict:
        """Convert motor positions to joint positions"""
        joint_pos = {}
        
        for idx, pos in enumerate(motor_pos):
            motor_id = self.motor_ids[idx]
            joint_name = self.motor_to_joint_map.get(motor_id)
            
            if not joint_name:
                continue
            
            # Check if calibrated
            if (motor_id not in self.motor_limits or 
                any(limit is None for limit in self.motor_limits[motor_id]) or
                self.joint_to_motor_ratios.get(motor_id, 0) == 0):
                joint_pos[joint_name] = None
                continue
            
            # Convert position
            if self.joint_inversion.get(joint_name, False):
                joint_pos[joint_name] = (self.joint_roms[joint_name][1] - 
                                        (pos - self.motor_limits[motor_id][0]) / 
                                        self.joint_to_motor_ratios[motor_id])
            else:
                joint_pos[joint_name] = (self.joint_roms[joint_name][0] + 
                                        (pos - self.motor_limits[motor_id][0]) / 
                                        self.joint_to_motor_ratios[motor_id])
        
        return joint_pos

    def _joint_to_motor_pos(self, joint_pos: dict) -> np.ndarray:
        """Convert joint positions to motor positions"""
        motor_pos = self.get_motor_pos().copy()  # Ensure getting complete motor position array
        
        for joint_name, pos in joint_pos.items():
            if pos is None:
                continue
            
            motor_id = self.joint_to_motor_map.get(joint_name)
            if motor_id is None:
                continue
            
            # Check if motor ID is in motor list
            if motor_id not in self.motor_ids:
                print(f"Warning: Motor {motor_id} for joint {joint_name} not in motor_ids")
                continue
            
            # Check if calibrated
            if (motor_id not in self.motor_limits or
                any(limit is None for limit in self.motor_limits[motor_id])):
                raise ValueError(f"Motor {motor_id} ({joint_name}) not calibrated")
            
            motor_idx = self.motor_ids.index(motor_id)
            
            # Ensure index is within valid range
            if motor_idx >= len(motor_pos):
                print(f"Error: motor_idx {motor_idx} >= len(motor_pos) {len(motor_pos)}")
                print(f"motor_ids: {self.motor_ids}")
                print(f"motor_pos shape: {motor_pos.shape}")
                raise IndexError(f"Motor index {motor_idx} out of range for motor_pos array")
            
            # Convert position
            if self.joint_inversion.get(joint_name, False):
                motor_pos[motor_idx] = (self.motor_limits[motor_id][0] + 
                                    (self.joint_roms[joint_name][1] - pos) * 
                                    self.joint_to_motor_ratios[motor_id])
            else:
                motor_pos[motor_idx] = (self.motor_limits[motor_id][0] + 
                                    (pos - self.joint_roms[joint_name][0]) * 
                                    self.joint_to_motor_ratios[motor_id])
        return motor_pos

    def _sanity_check(self):
        """Validate configuration"""
        if len(self.motor_ids) != len(self.joint_ids):
            raise ValueError("Number of motor IDs and joint IDs don't match")
        
        if self.control_mode not in ['current', 'velocity', 'position', 
                                     'multi_turn_position', 'current_based_position']:
            raise ValueError(f"Invalid control mode: {self.control_mode}")
        
        for joint, motor_id in self.joint_to_motor_map.items():
            if joint not in self.joint_ids:
                raise ValueError(f"Joint {joint} not in joint_ids")
            if motor_id not in self.motor_ids:
                raise ValueError(f"Motor {motor_id} not in motor_ids")

    # Compatibility methods for scripts
    def read_pos_vel_cur(self):
        """Read position, velocity, current (for compatibility)"""
        return self._motor_client.read_pos_vel_cur()

    def write_desired_pos(self, joint_ids, positions):
        """Write desired positions (for compatibility)"""
        # Convert joint IDs to motor IDs if needed
        if all(isinstance(jid, str) for jid in joint_ids):
            motor_ids = [self.joint_to_motor_map[jid] for jid in joint_ids]
        else:
            motor_ids = joint_ids
        
        self._motor_client.write_desired_pos(motor_ids, positions)

    def set_torque_enabled(self, motor_ids, enabled):
        """Set torque enabled (for compatibility)"""
        self._motor_client.set_torque_enabled(motor_ids, enabled)

    def set_joint_current(self, current_dict):
        """Set joint currents (for compatibility)"""
        motor_currents = {}
        for joint, current in current_dict.items():
            motor_id = self.joint_to_motor_map.get(joint, joint)
            motor_currents[motor_id] = current
        
        currents = [motor_currents.get(mid, self.max_current) for mid in self.motor_ids]
        self._motor_client.write_desired_current(self.motor_ids, np.array(currents))

    def read_joint_currents(self):
        """Read joint currents (for compatibility)"""
        return self.get_motor_current(as_dict=True)

    def joint_ids(self):
        """Get joint IDs (for compatibility)"""
        return self.motor_ids