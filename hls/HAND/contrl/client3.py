import atexit
import logging
import time
from typing import Optional, Sequence, Union, Tuple, Dict
import numpy as np
import sys

sys.path.append("..")
from scservo_sdk import *
from scservo_sdk.protocol_packet_handler import protocol_packet_handler
from scservo_sdk.group_sync_write import GroupSyncWrite
from scservo_sdk.group_sync_read import GroupSyncRead

# SCServo position scale: 0-4095 raw units = 0-360 degrees = 0-2π radians
DEFAULT_POS_SCALE = 2.0 * np.pi / 4095  # Convert raw to radians
DEFAULT_VEL_SCALE = 0.732  # rpm per unit (for SCServo)
DEFAULT_CUR_SCALE = 6.5  # mA per unit

# Register addresses for SCServo
ADDR_TORQUE_ENABLE = 40
ADDR_GOAL_ACCELERATION = 41
ADDR_GOAL_POSITION_L = 42
ADDR_GOAL_TORQUE_L = 44
ADDR_GOAL_SPEED_L = 46
ADDR_PRESENT_POSITION_L = 56
ADDR_PRESENT_SPEED_L = 58
ADDR_PRESENT_LOAD_L = 60
ADDR_PRESENT_CURRENT_L = 69
ADDR_PRESENT_TEMPERATURE = 63
ADDR_MOVING = 66
ADDR_MODE = 33
ADDR_LOCK = 55

class BMotorClient:
    """Client for SCServo motors, mimicking DynamixelClient interface"""
    
    def __init__(self,
                 motor_ids: Sequence[int],
                 port: str = "/dev/ttyUSB0",
                 baudrate: int = 1000000,
                 pos_scale: Optional[float] = None,
                 vel_scale: Optional[float] = None,
                 cur_scale: Optional[float] = None,
                 lazy_connect: bool = False):
        
        self.motor_ids = list(motor_ids)
        self.port = port
        self.baudrate = baudrate
        self.lazy_connect = lazy_connect
        
        self.pos_scale = pos_scale if pos_scale is not None else DEFAULT_POS_SCALE
        self.vel_scale = vel_scale if vel_scale is not None else DEFAULT_VEL_SCALE
        self.cur_scale = cur_scale if cur_scale is not None else DEFAULT_CUR_SCALE
        
        self.ph = PortHandler(port)
        self.packetHandler = protocol_packet_handler(self.ph, 0)
        self.groupSyncWrite = GroupSyncWrite(self.packetHandler, ADDR_GOAL_ACCELERATION, 7)
        self.groupSyncRead = GroupSyncRead(self.packetHandler, ADDR_PRESENT_POSITION_L, 10)
        
        self._is_connected = False

    def connect(self):
        """Connect to SCServo motors"""
        try:
            if self.ph.openPort():
                logging.info(f"Opened port {self.port}")
                if self.ph.setBaudRate(self.baudrate):
                    logging.info(f"Set baudrate to {self.baudrate}")
                    self._is_connected = True
                    # Enable torque by default
                    self.set_torque_enabled(self.motor_ids, True)
                    return True, "Connection successful"
                else:
                    return False, f"Failed to set baudrate {self.baudrate}"
            else:
                return False, f"Failed to open port {self.port}"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"

    def disconnect(self):
        """Disconnect from motors"""
        if self._is_connected:
            self.set_torque_enabled(self.motor_ids, False, retries=0)
            self.ph.closePort()
            self._is_connected = False

    def is_connected(self) -> bool:
        """Check connection status"""
        return self._is_connected

    def check_connected(self):
        """Ensure connected, with lazy connect support"""
        if self.lazy_connect and not self._is_connected:
            self.connect()
        if not self._is_connected:
            raise RuntimeError("Not connected to motors")

    def set_operating_mode(self, motor_ids: Sequence[int], mode: int):
        """
        Set operating mode (SCServo has limited modes)
        0: current control mode -> Not directly supported, use current-based position
        1: velocity control mode -> Wheel mode
        3: position control mode -> Servo mode (default)
        4: multi-turn position control mode -> Not supported
        5: current-based position control mode -> Servo mode with torque limit
        """
        self.check_connected()
        
        # Disable torque to change mode
        self.set_torque_enabled(motor_ids, False)
        
        for mid in motor_ids:
            if mode == 1:  # Velocity mode -> Wheel mode
                self.packetHandler.write1ByteTxRx(mid, ADDR_MODE, 1)
            else:  # Position modes -> Servo mode
                self.packetHandler.write1ByteTxRx(mid, ADDR_MODE, 0)
        
        # Re-enable torque
        self.set_torque_enabled(motor_ids, True)

    def set_torque_enabled(self, motor_ids: Sequence[int], enabled: bool, 
                          retries: int = -1, retry_interval: float = 0.25):
        """Enable/disable torque for motors"""
        self.check_connected()
        remaining_ids = list(motor_ids)
        
        while remaining_ids:
            failed_ids = []
            for mid in remaining_ids:
                result, error = self.packetHandler.write1ByteTxRx(
                    mid, ADDR_TORQUE_ENABLE, int(enabled))
                if result != COMM_SUCCESS or error != 0:
                    failed_ids.append(mid)
            
            remaining_ids = failed_ids
            if remaining_ids:
                logging.error(f"Failed to set torque for IDs: {remaining_ids}")
                if retries == 0:
                    break
                time.sleep(retry_interval)
                retries -= 1

    def read_temperature(self) -> np.ndarray:
        """Read temperature for all motors"""
        self.check_connected()
        temps = []
        
        for mid in self.motor_ids:
            temp, result, error = self.packetHandler.read1ByteTxRx(
                mid, ADDR_PRESENT_TEMPERATURE)
            temps.append(float(temp))
        
        return np.array(temps, dtype=np.float32)

    def read_status_is_done_moving(self) -> np.ndarray:
        """Check if motors have finished moving"""
        self.check_connected()
        status = []
        
        for mid in self.motor_ids:
            moving, result, error = self.packetHandler.read1ByteTxRx(mid, ADDR_MOVING)
            status.append(moving == 0)  # 0 means stopped, 1 means moving
        
        return np.array(status, dtype=bool)

    def write_desired_current(self, motor_ids: Sequence[int], currents: np.ndarray):
        """Set current/torque limits for motors"""
        self.check_connected()
        assert len(motor_ids) == len(currents)
        
        # Prepare parameters for sync write
        param = []
        
        for mid, current in zip(motor_ids, currents):
            torque_raw = int(current / self.cur_scale)
            torque_raw = np.clip(torque_raw, 0, 1000)
            
            # Add to parameter list: ID, Torque_L, Torque_H
            param.append(mid)
            param.append(self.packetHandler.scs_lobyte(torque_raw))
            param.append(self.packetHandler.scs_hibyte(torque_raw))
        
        # Use syncWriteTxOnly to write torque limits
        result = self.packetHandler.syncWriteTxOnly(
            start_address=ADDR_GOAL_TORQUE_L,
            data_length=2,
            param=param,
            param_length=len(param)
        )
        
        if result != COMM_SUCCESS:
            logging.error(f"Failed to sync write torque limits: {self.packetHandler.getTxRxResult(result)}")

    def write_profile_velocity(self, motor_ids: Sequence[int], velocities: np.ndarray):
        """Set profile velocities (speed limits) for position moves"""
        self.check_connected()
        assert len(motor_ids) == len(velocities)
        
        # Prepare parameters for sync write
        param = []
        
        for mid, vel in zip(motor_ids, velocities):
            vel_raw = int(vel / self.vel_scale)
            vel_raw = np.clip(vel_raw, 0, 1000)
            
            # Add to parameter list: ID, Speed_L, Speed_H
            param.append(mid)
            param.append(self.packetHandler.scs_lobyte(vel_raw))
            param.append(self.packetHandler.scs_hibyte(vel_raw))
        
        # Use syncWriteTxOnly to write speeds
        result = self.packetHandler.syncWriteTxOnly(
            start_address=ADDR_GOAL_SPEED_L,
            data_length=2,
            param=param,
            param_length=len(param)
        )
        
        if result != COMM_SUCCESS:
            logging.error(f"Failed to sync write speeds: {self.packetHandler.getTxRxResult(result)}")

    def sync_write_pos_with_params(self, position_dict: Dict[int, float],
                                   speed: int = 60, acc: int = 50, torque: int = 500):
        """Synchronized write with position, speed, acceleration and torque"""
        self.check_connected()
        
        # Prepare parameters for sync write
        # Each motor needs: ID + acc + pos_L + pos_H + torque_L + torque_H + speed_L + speed_H
        param = []
        
        for mid, pos_rad in position_dict.items():
            pos_raw = int(pos_rad / self.pos_scale)
            pos_raw = np.clip(pos_raw, 0, 4095)
            
            # Add motor ID
            param.append(mid)
            # Add parameters: acc, pos_L, pos_H, torque_L, torque_H, speed_L, speed_H
            param.append(acc)
            param.append(self.packetHandler.scs_lobyte(pos_raw))
            param.append(self.packetHandler.scs_hibyte(pos_raw))
            param.append(self.packetHandler.scs_lobyte(torque))
            param.append(self.packetHandler.scs_hibyte(torque))
            param.append(self.packetHandler.scs_lobyte(speed))
            param.append(self.packetHandler.scs_hibyte(speed))
        
        # Use syncWriteTxOnly
        result = self.packetHandler.syncWriteTxOnly(
            start_address=ADDR_GOAL_ACCELERATION,
            data_length=7,  # acc + pos(2) + torque(2) + speed(2)
            param=param,
            param_length=len(param)
        )
        
        if result != COMM_SUCCESS:
            logging.error(f"Sync write failed: {self.packetHandler.getTxRxResult(result)}")

    def __enter__(self):
        """Context manager support"""
        if not self._is_connected:
            self.connect()
        return self

    def __exit__(self, *args):
        """Context manager support"""
        self.disconnect()

    def __del__(self):
        """Cleanup on deletion"""
        self.disconnect()

    def read_current(self) -> np.ndarray:
        """Read current for all motors"""
        self.check_connected()
        currents = []
        
        for mid in self.motor_ids:
            # Read current (2 bytes)
            cur_raw, result, error = self.packetHandler.read2ByteTxRx(
                mid, ADDR_PRESENT_CURRENT_L)
            if result == COMM_SUCCESS and error == 0:
                cur = self.packetHandler.scs_tohost(cur_raw, 15) * self.cur_scale
            else:
                cur = 0.0  # Default value if read fails
            currents.append(cur)
        
        return np.array(currents, dtype=np.float32)

    def read_pos_vel_cur(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read position, velocity and current for all motors (with extended range support)"""
        self.check_connected()
        
        positions = []
        velocities = []
        currents = []
        
        for mid in self.motor_ids:
            # Read position (2 bytes) - handle signed values for extended range
            pos_raw, result, error = self.packetHandler.read2ByteTxRx(
                mid, ADDR_PRESENT_POSITION_L)
            if result == COMM_SUCCESS and error == 0:
                # Use scs_tohost to handle signed conversion for extended range
                pos_signed = self.packetHandler.scs_tohost(pos_raw, 15)
                pos = pos_signed * self.pos_scale
            else:
                pos = 0.0
            
            # Read velocity (2 bytes)
            vel_raw, result, error = self.packetHandler.read2ByteTxRx(
                mid, ADDR_PRESENT_SPEED_L)
            if result == COMM_SUCCESS and error == 0:
                vel_signed = self.packetHandler.scs_tohost(vel_raw, 15)
                vel = vel_signed * self.vel_scale
            else:
                vel = 0.0
            
            # Read current (2 bytes)
            cur_raw, result, error = self.packetHandler.read2ByteTxRx(
                mid, ADDR_PRESENT_CURRENT_L)
            if result == COMM_SUCCESS and error == 0:
                cur_signed = self.packetHandler.scs_tohost(cur_raw, 15)
                cur = cur_signed * self.cur_scale
            else:
                cur = 0.0
            
            positions.append(pos)
            velocities.append(vel)
            currents.append(cur)
        
        return np.array(positions), np.array(velocities), np.array(currents)

    def read_single_motor_current(self, motor_id: int) -> float:
        """Read current for a single motor with proper signed handling"""
        self.check_connected()
        
        cur_raw, result, error = self.packetHandler.read2ByteTxRx(
            motor_id, ADDR_PRESENT_CURRENT_L)
        if result == COMM_SUCCESS and error == 0:
            # Use scs_tohost for proper signed conversion
            cur_signed = self.packetHandler.scs_tohost(cur_raw, 15)
            return cur_signed * self.cur_scale
        else:
            return 0.0

    def read_single_motor_position(self, motor_id: int) -> float:
        """Read position for a single motor with extended range support"""
        self.check_connected()
        
        pos_raw, result, error = self.packetHandler.read2ByteTxRx(
            motor_id, ADDR_PRESENT_POSITION_L)
        if result == COMM_SUCCESS and error == 0:
            # Use scs_tohost for proper signed conversion
            pos_signed = self.packetHandler.scs_tohost(pos_raw, 15)
            return pos_signed * self.pos_scale
        else:
            return 0.0

    def write_desired_pos(self, motor_ids: Sequence[int], positions: np.ndarray, 
                     speed: Optional[int] = None, torque: Optional[int] = None):
        """Write desired positions to motors (in radians) with extended range support"""
        self.check_connected()
        assert len(motor_ids) == len(positions)
        
        # Default values if not specified
        default_speed = 30
        default_torque = 300
        
        for mid, pos_rad in zip(motor_ids, positions):
            # Convert radians to raw position values
            pos_raw = int(pos_rad / self.pos_scale)
            
            # Use scs_toscs to handle signed conversion for extended range
            pos_scs = self.packetHandler.scs_toscs(pos_raw, 15)
            
            # Set speed if specified
            if speed is not None:
                speed_result, speed_error = self.packetHandler.write2ByteTxRx(
                    mid, ADDR_GOAL_SPEED_L, speed)
                if speed_result != COMM_SUCCESS or speed_error != 0:
                    logging.warning(f"Failed to set speed for motor {mid}")
            
            # Set torque if specified
            if torque is not None:
                torque_result, torque_error = self.packetHandler.write2ByteTxRx(
                    mid, ADDR_GOAL_TORQUE_L, torque)
                if torque_result != COMM_SUCCESS or torque_error != 0:
                    logging.warning(f"Failed to set torque for motor {mid}")
            
            # Write position using 2-byte write
            result, error = self.packetHandler.write2ByteTxRx(mid, ADDR_GOAL_POSITION_L, pos_scs)
            
            if result != COMM_SUCCESS or error != 0:
                logging.error(f"Failed to write position to motor {mid}: result={result}, error={error}")

    def write_single_motor_position(self, motor_id: int, position_rad: float, 
                                speed: Optional[int] = None, torque: Optional[int] = None, 
                                acceleration: Optional[int] = None):
        """Write position to a single motor with extended range support and motion parameters"""
        self.check_connected()
        
        # Convert radians to raw position
        pos_raw = int(position_rad / self.pos_scale)
        
        # Use scs_toscs for proper signed conversion
        pos_scs = self.packetHandler.scs_toscs(pos_raw, 15)
        
        # Set acceleration if specified
        if acceleration is not None:
            acc_result, acc_error = self.packetHandler.write1ByteTxRx(
                motor_id, ADDR_GOAL_ACCELERATION, acceleration)
            if acc_result != COMM_SUCCESS or acc_error != 0:
                logging.warning(f"Failed to set acceleration for motor {motor_id}")
        
        # Set speed if specified
        if speed is not None:
            speed_result, speed_error = self.packetHandler.write2ByteTxRx(
                motor_id, ADDR_GOAL_SPEED_L, speed)
            if speed_result != COMM_SUCCESS or speed_error != 0:
                logging.warning(f"Failed to set speed for motor {motor_id}")
        
        # Set torque if specified
        if torque is not None:
            torque_result, torque_error = self.packetHandler.write2ByteTxRx(
                motor_id, ADDR_GOAL_TORQUE_L, torque)
            if torque_result != COMM_SUCCESS or torque_error != 0:
                logging.warning(f"Failed to set torque for motor {motor_id}")
        
        # Write position
        result, error = self.packetHandler.write2ByteTxRx(motor_id, ADDR_GOAL_POSITION_L, pos_scs)
        
        if result != COMM_SUCCESS or error != 0:
            logging.error(f"Failed to write position to motor {motor_id}: result={result}, error={error}")
            return False
        return True

    def write_positions_with_speed_acc_torque(self, position_dict: Dict[int, float],
                                            speed: int = 60, acc: int = 50, torque: int = 500):
        """Synchronized write with position, speed, acceleration and torque using extended range"""
        self.check_connected()
        
        # Prepare parameters for synchronized write
        # Format: [ID1, acc, pos_L, pos_H, torque_L, torque_H, speed_L, speed_H, ID2, ...]
        param = []
        
        for motor_id in self.motor_ids:  # Use fixed motor_ids order to ensure consistency
            if motor_id in position_dict:
                pos_rad = position_dict[motor_id]
            else:
                # If position not specified for this motor, read its current position
                pos_rad = self.read_single_motor_position(motor_id)
            
            # Convert radians to raw position value
            pos_raw = int(pos_rad / self.pos_scale)
            pos_scs = self.packetHandler.scs_toscs(pos_raw, 15)
            
            # Add to parameter list
            param.append(motor_id)
            param.append(acc)
            param.append(self.packetHandler.scs_lobyte(pos_scs))
            param.append(self.packetHandler.scs_hibyte(pos_scs))
            param.append(self.packetHandler.scs_lobyte(torque))
            param.append(self.packetHandler.scs_hibyte(torque))
            param.append(self.packetHandler.scs_lobyte(speed))
            param.append(self.packetHandler.scs_hibyte(speed))
        
        # Use synchronized write to send all commands at once
        result = self.packetHandler.syncWriteTxOnly(
            start_address=ADDR_GOAL_ACCELERATION,
            data_length=7,  # acc(1) + pos(2) + torque(2) + speed(2)
            param=param,
            param_length=len(param)
        )
        
        if result != COMM_SUCCESS:
            logging.error(f"Sync write failed: {self.packetHandler.getTxRxResult(result)}")
            # If synchronized write fails, fall back to individual writes (ensure compatibility)
            for motor_id, pos_rad in position_dict.items():
                self.write_single_motor_position(motor_id, pos_rad, speed, torque, acc)