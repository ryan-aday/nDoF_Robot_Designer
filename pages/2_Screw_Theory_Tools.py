import numpy as np
import streamlit as st

from utils.robot import rotation_matrix


st.title("Screw Theory & Kinematics Tools")
st.write(
    "Utilities to reason about twists, wrenches, and exponential coordinates based on screw theory."
)

st.markdown(
    """
    These calculators are inspired by the formulations in the references below. They are kept on a
    dedicated page to avoid interrupting the main robot design flow while still surfacing the 
    kinematic/dynamic equations that govern the visualization.
    """
)

with st.expander("Twist calculator"):
    v = st.text_input("Linear velocity vector (comma separated)", "0.1, 0.0, 0.0")
    w = st.text_input("Angular velocity vector (comma separated)", "0.0, 0.0, 1.0")
    try:
        v_vec = np.array([float(x.strip()) for x in v.split(",")])
        w_vec = np.array([float(x.strip()) for x in w.split(",")])
        twist = np.concatenate([w_vec, v_vec])
        st.write("Twist (ω, v):", twist)
        st.code(
            "Twist ξ = [ω, v]^T; SE(3) transform via exp([ω]^x θ) and screw axis", language="python"
        )
    except Exception as e:  # noqa: BLE001
        st.error(f"Invalid input: {e}")

with st.expander("Exponential coordinates"):
    theta = st.number_input("Rotation angle (rad)", -np.pi, np.pi, 0.5, 0.1)
    axis = st.text_input("Axis (unit vector)", "0,0,1")
    try:
        axis_vec = np.array([float(x.strip()) for x in axis.split(",")])
        rot = rotation_matrix(axis_vec, theta)
        st.write("Rotation matrix R = exp([ω]^x θ):")
        st.write(rot)
    except Exception as e:  # noqa: BLE001
        st.error(f"Invalid axis: {e}")

with st.expander("Wrench projection"):
    wrench = st.text_input("Wrench (force, torque) vector", "5,0,0,  0,0,1")
    try:
        wrench_vec = np.array([float(x.strip()) for x in wrench.split(",")])
        force = wrench_vec[:3]
        torque = wrench_vec[3:]
        st.write(f"Force: {force}, Torque: {torque}")
        st.write("Power = force · linear_velocity + torque · angular_velocity")
    except Exception as e:  # noqa: BLE001
        st.error(f"Invalid wrench: {e}")

st.info(
    "Key references: Wikipedia on screw theory and Herman Bruyninckx's 'Robot Kinematics and Dynamics'."
)
