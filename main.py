# main.py
import time
import pybullet as p
from environment import setup_pybullet_env
import numpy as np
from utils import Utils
import proxsuite
import pinocchio as pin

END_EFF_FRAME_ID = 16

# Computed Forward Kinematics
def forward_kinematics(pin_robot, q):
    pin_robot.forwardKinematics(q)
    T_S_F = pin_robot.framePlacement(q, END_EFF_FRAME_ID)
    transformation = np.eye(4)
    transformation[0:3, 3] = T_S_F.translation
    transformation[0:3, 0:3] = T_S_F.rotation
    return transformation

# Computed Jacobian (in a frame oriented like the spatial frame but located at the end-effector)
def get_velocity_jacobian(pin_robot, q):
    T_S_F = pin_robot.framePlacement(q, END_EFF_FRAME_ID)
    J = pin_robot.computeFrameJacobian(q, END_EFF_FRAME_ID)
    J = T_S_F.rotation @ J[:3, :]
    return J

# Computed Jacobian 
def get_body_jacobian(pin_robot, q):
    J = pin_robot.computeFrameJacobian(q, END_EFF_FRAME_ID)
    return J

# True EEF transformation
def current_eef(targid):
    link_state = p.getLinkState(targid, 6)
    transformation = np.eye(4)
    transformation[0:3, 3] = np.array(link_state[4])
    transformation[0:3, 0:3] = Utils.quaternion_to_rot(link_state[5])
    return transformation

def solve_dq(pin_robot, q, target, max_dq):
    fk = forward_kinematics(pin_robot, q)
    J = get_body_jacobian(pin_robot, q)
    
    twist_error = -Utils.transformation_to_twist(Utils.inverse_homogeneous_matrix(fk) @ target, v_top=True)

    H = J.T @ J
    g = J.T @ twist_error

    qp = proxsuite.proxqp.dense.QP(7, 0, 0, True)
    lower_dq_limit = -np.ones(7) * max_dq
    upper_dq_limit = np.ones(7) * max_dq
    qp.init(H, g, None, None, None, None, None, lower_dq_limit, upper_dq_limit)
    qp.solve()
    dq = qp.results.x

    return dq, twist_error

def main():
    control_frequency = 500  # Control frequency in Hz
    control_period = 1.0 / control_frequency  # Control period in seconds

    num_joints = 7
    targid, pin_robot = setup_pybullet_env()
    
    q = np.zeros(7)
    dq = np.zeros(7)
   
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)

    target_position = np.array([0.5, 0.23, 0.94])

    target_T = np.array([
        [1.0, 0.0, 0.0, 0.2],
        [0.0, 1.0, 0.0, 0.2],
        [0.0, 0.0, 1.0, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    ])


    try:
        start_time = time.time()
        for _ in range(10000):
            
            control_start_time = time.time()
            

            # Update Joints Positions and Velocities
            for joint_index in range(0, 7):
                joint_info = p.getJointState(targid, joint_index)
                q[joint_index] = joint_info[0]
                dq[joint_index] = joint_info[1]

            dq_new, error = solve_dq(pin_robot, q, target_T, 0.1)
            joint_angles = q + dq_new*1

            print("Before: ", q)
            print("After ", joint_angles)

            print("Error: {:.4f}".format(np.sqrt(np.sum(error * error))))
            print("Time: {:.4f}".format(time.time()-start_time))

            for joint_index in range(num_joints):
                p.setJointMotorControl2(bodyUniqueId=targid,
                                        jointIndex=joint_index,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=joint_angles[joint_index],
                                        force=500)
            

            elapsed_time = time.time() - control_start_time
            print(1/elapsed_time)
            if elapsed_time<control_period:
                time.sleep(control_period-elapsed_time)
            p.stepSimulation()

    except KeyboardInterrupt:
        print("Simulation interrupted!")

    finally:
        p.disconnect()

if __name__ == "__main__":
    main()
