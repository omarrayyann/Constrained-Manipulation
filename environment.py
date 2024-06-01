import pybullet as p
import pybullet_data
import time
import pinocchio as pin

def setup_pybullet_env():
    p.connect(p.GUI)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    # Loading Assets
    p.loadURDF("plane.urdf", [0, 0, 0], [0, 0, 0, 1])
    targid = p.loadURDF("urdf/iiwa.urdf", [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)
    pin_robot = pin.RobotWrapper.BuildFromURDF("urdf/iiwa.urdf", "urdf")
    return targid, pin_robot

