from dynamixel_sdk import *  # Uses Dynamixel SDK library
import time
import threading

# -------- User Parameters --------
CURRENT_RESOLUTION = 2.69  # mA/unit (check your motor's datasheet)
POSITION_RESOLUTION = 4096  # ticks per revolution (for MX-28, X-series, etc.)
ROTATIONS = 1.5*POSITION_RESOLUTION  # 2 full turns
WAIT_TIME = 2.0      # seconds for ramp up
RAMP_DOWN_TIME = 10.0 # seconds for ramp down
STEPS = 100           # Number of steps in the ramp
RAMP_END_CURRENT = 0.3  # End current in Amperes

# -------- Control Table Addresses --------
ADDR_TORQUE_ENABLE      = 64
ADDR_GOAL_CURRENT       = 102
ADDR_PRESENT_CURRENT    = 126
ADDR_OPERATING_MODE     = 11
ADDR_GOAL_POSITION      = 116
ADDR_PRESENT_POSITION   = 132
ADDR_PROFILE_VELOCITY = 112
ADDR_VELOCITY_LIMIT = 44
ADDR_CURRENT_LIMIT = 38
ADDR_PROFILE_ACCELERATION = 108
ADDR_HOMING_OFFSET = 20



# -------- Protocol Settings --------
PROTOCOL_VERSION        = 2.0
DXL_ID                  = 2
BAUDRATE                = 1000000
DEVICENAME              = '/dev/ttyUSB0'

# -------- Motor Control Values --------
TORQUE_ENABLE           = 1
TORQUE_DISABLE          = 0
CURRENT_CONTROL_MODE    = 0
POSITION_CONTROL_MODE   = 4
present_current_a = 0
present_pos = 0
reached_goal = None
begin_pos = 0

# -------- Initialize SDK --------
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Add this near your other addresses

def to_unsigned_32bit(val):
    return val & 0xFFFFFFFF

def set_profile_acceleration(accel_rpm2):
    # 1 unit = 214.577 [rev/min^2]
    accel_unit = int(accel_rpm2 / 214.577)
    packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_PROFILE_ACCELERATION, accel_unit)

def set_current_limit(current_a):
    current_unit = int(current_a * 1000 / 2.69)
    packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_CURRENT_LIMIT, current_unit)

def set_profile_velocity(velocity_rps):
    velocity_rpm = velocity_rps * 60
    velocity_unit = int(velocity_rpm / 0.229)
    packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_PROFILE_VELOCITY, velocity_unit)

def set_velocity_limit(velocity_rps):
    velocity_rpm = velocity_rps * 60
    velocity_unit = int(velocity_rpm / 0.229)
    packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_VELOCITY_LIMIT, velocity_unit)
    
def set_operating_mode(mode):
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_OPERATING_MODE, mode)
    packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

def move_to_position(goal_pos):
    packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_POSITION, goal_pos)

def wait_until_position_reached(goal_pos, tolerance=20, timeout=30):
    """Wait until the motor reaches the goal position or timeout (seconds) or stop_flag is set."""
    global reached_goal
    start_time = time.time()
    while True:
        if stop_flag:
            print("Motion stopped by user.")
            break

        present_pos, _, _ = packetHandler.read4ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_POSITION)
        if goal_pos > 2**31 and present_pos < 2**31:
            error = abs((goal_pos - 2**32) - present_pos)
        else:
            error = abs(goal_pos - present_pos)
        print('position error' , error)
        if error <= tolerance:
            print(f"Goal reached: {present_pos} (error: {error})")
            reached_goal = present_pos
            break
        if time.time() - start_time > timeout:
            print("Timeout waiting for position.")
            break
        time.sleep(0.05)


def set_current(current_a):
    set_operating_mode(CURRENT_CONTROL_MODE)
    goal_current = int(current_a / CURRENT_RESOLUTION * 1000) & 0xFFFF
    packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_CURRENT, goal_current)

# -------- Open Port --------
if not portHandler.openPort():
    raise IOError("Failed to open port")
if not portHandler.setBaudRate(BAUDRATE):
    raise IOError("Failed to set baudrate")



stop_flag = False
def wait_for_enter():
    global stop_flag
    input("Press Enter at any time to stop the motion...\n")
    stop_flag = True

thread = threading.Thread(target=wait_for_enter)
thread.daemon = True
thread.start()

# -------- Move arm up (negative direction, 2 rotations) --------

set_operating_mode(POSITION_CONTROL_MODE)
print(f"Moving arm up {ROTATIONS/POSITION_RESOLUTION}")
packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_HOMING_OFFSET, 100000)
present_pos, _, _ = packetHandler.read4ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_POSITION)
print(present_pos, 'begin posss')
goal_position = int(present_pos - ROTATIONS)
if goal_position < 0:
    goal_position = to_unsigned_32bit(goal_position)
print("goal position", goal_position)
time.sleep(0.5)
set_profile_acceleration(3*214.577)  # 1 unit, very smooth
set_velocity_limit(0.3)      # Max 0.2 rotations/sec (12 rpm)
set_profile_velocity(0.1)    # Move at 0.1 rotations/sec (6 rpm)
set_current_limit(0.3)  # Limit to 0.2 A

move_to_position(int(goal_position))
wait_until_position_reached(goal_position, tolerance=20, timeout=30)
present_current_raw, _, _ = packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_PRESENT_CURRENT)
current_hold = present_current_raw
print('current:', present_current_raw)



if not stop_flag:
    # -------- Hold position --------
    print("Holding position for 2 seconds...")
    for _ in range(int(WAIT_TIME / 0.1)):
        move_to_position(reached_goal)
        if stop_flag:
            break
        time.sleep(0.1)

if not stop_flag and present_current_a == 0:
    # Read present current (in Dynamixel units)
    # Convert to signed value if necessary (Dynamixel returns unsigned, but current can be negative)
    if current_hold > 32767:
        current_hold -= 65536

    # Convert to Amperes
    present_current_a = current_hold * CURRENT_RESOLUTION / 1000.0  # mA to A
    print(f"Present current: {present_current_a:.2f} A")


if not stop_flag:
    # -------- Move arm down (positive direction) with current ramp down --------
    print("Switching to current control mode and ramping down current...")
    set_operating_mode(CURRENT_CONTROL_MODE)

    current_down_start = present_current_a  # [A], replace with actual value if you read it
    linspace_current = (RAMP_END_CURRENT - current_down_start) / STEPS

    for step in range(STEPS + 1):
        if stop_flag:
            print("Ramp stopped by user.")
            break
        current = current_down_start + linspace_current * step
        goal_current = int(current / CURRENT_RESOLUTION * 1000) & 0xFFFF
        packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_CURRENT, goal_current)
        time.sleep(RAMP_DOWN_TIME / STEPS)

print("Motion complete or stopped by user.")

# -------- Stop Torque --------
packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_CURRENT, 0)
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
portHandler.closePort()