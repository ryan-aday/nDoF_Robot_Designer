import numpy as np
import streamlit as st

from utils.robot import rotation_matrix

st.title("Fundamental Formulas")
st.write(
    "A quick reference for screw theory, forward kinematics, and related formulas used throughout the app."
)
st.markdown(
    "References: [Screw theory](https://en.wikipedia.org/wiki/Screw_theory) | "
    "[Robot Kinematics and Dynamics](https://u0011821.pages.gitlab.kuleuven.be/robotics/2009-HermanBruyninckx-robot-kinematics-and-dynamics.pdf) | "
    "[Numerical IK (Modern Robotics)](https://modernrobotics.northwestern.edu/nu-gm-book-resource/6-2-numerical-inverse-kinematics-part-1-of-2/) | "
    "[Inverse kinematics overview](https://www.mathworks.com/discovery/inverse-kinematics.html) | "
    "[Gradient descent in robotics](https://www.meegle.com/en_us/topics/gradient-descent/gradient-descent-in-robotics) | "
    "[Screw-theory improvements](https://journals.sagepub.com/doi/10.5772/60834) | "
    "[Trajectory smoothing](https://www.witpress.com/Secure/elibrary/papers/HPSM25/HPSM25011FU1.pdf) | "
    "[Manipulator optimization](https://www.sciencedirect.com/science/article/abs/pii/S0094114X05001424) | "
    "[Forward kinematics via DH](https://automaticaddison.com/homogeneous-transformation-matrices-using-denavit-hartenberg/)"
)

st.markdown(
    """
    These calculators and snippets keep the core equations close at hand without interrupting the main robot design flow.
    """
)

with st.expander("Forward kinematics (DH-style)"):
    st.latex(r"{}^{i-1}T_i = R_{z}(\theta_i) ")
    st.latex(r"\quad R_{x}(\alpha_i) ")
    st.latex(r"\quad T_{z}(d_i) ")
    st.latex(r"\quad T_{x}(a_i)")
    st.markdown(
        "Where $a_i$ is the link length, $\alpha_i$ the twist, $d_i$ the offset, and $\theta_i$ the joint variable. "
        "A chain's end-effector transform multiplies successive ${}^{i-1}T_i$ matrices."
    )
    st.markdown(
        "For the finite-difference solver used here, the positional Jacobian is sampled directly from repeated forward-kinematics "
        "evaluations based on these homogeneous transforms (see Addison's DH refresher)."
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
            "Twist ξ = [ω, v]^T; SE(3) transform via exp([ω]^x θ) and screw axis",
            language="python",
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
    "Key references: screw theory (Wikipedia), Herman Bruyninckx's 'Robot Kinematics and Dynamics', "
    "Modern Robotics numerical IK, MathWorks inverse-kinematics overview, gradient-descent applications to robotics, "
    "screw-theoretic refinements (He et al.), trajectory smoothing, redundant manipulator optimization, and DH-style homogeneous transforms (Addison)."
)
