from dynamixel_sdk import *  # Uses Dynamixel SDK library
import time
import threading
import sys

CURRENT_START = 0.0   # Start at 0 A
CURRENT_END = -0.1     # End at 0.1 A
RAMP_TIME = 10.0      # seconds
STEPS = 100           # Number of steps in the ramp

# -------- Control Table Addresses --------
ADDR_TORQUE_ENABLE      = 64
ADDR_GOAL_CURRENT       = 102
ADDR_PRESENT_CURRENT    = 126
ADDR_OPERATING_MODE     = 11

# -------- Protocol Settings --------
PROTOCOL_VERSION        = 2.0
DXL_ID                  = 2
BAUDRATE                = 1000000
DEVICENAME              = '/dev/ttyUSB0'

# -------- Motor Control Values --------
TORQUE_ENABLE           = 1
CURRENT_CONTROL_MODE    = 0

# Current resolution in mA/unit (example: 2.69 mA/unit for some Dynamixel motors)
CURRENT_RESOLUTION = 2.69  # Replace with the actual resolution from your motor's datasheet

# -------- Initialize SDK --------
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

# -------- Open Port --------
if not portHandler.openPort():
    raise IOError("Failed to open port")

if not portHandler.setBaudRate(BAUDRATE):
    raise IOError("Failed to set baudrate")

# -------- Disable Torque Before Mode Change --------
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, 0)

# -------- Set Operating Mode to Current Control --------
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_OPERATING_MODE, CURRENT_CONTROL_MODE)

# -------- Enable Torque --------
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)

# -------- Non-blocking Enter detection --------
stop_flag = False
def wait_for_enter():
    global stop_flag
    input("Press Enter to stop the ramp at any time...\n")
    stop_flag = True

thread = threading.Thread(target=wait_for_enter)
thread.daemon = True
thread.start()

# -------- Ramp Current --------
for step in range(STEPS + 1):
    if stop_flag:
        print("Ramp stopped by user.")
        break
    current = CURRENT_START + (CURRENT_END - CURRENT_START) * step / STEPS
    goal_current = int(current / CURRENT_RESOLUTION * 1000) & 0xFFFF  # Convert to units and unsigned
    packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_CURRENT, goal_current)
    time.sleep(RAMP_TIME / STEPS)

print("Current ramp complete or stopped.")

# -------- Stop Torque --------
packetHandler.write2ByteTxRx(portHandler, DXL_ID, ADDR_GOAL_CURRENT, 0)

# -------- Disable Torque & Close --------
packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_TORQUE_ENABLE, 0)
portHandler.closePort()