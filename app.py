import streamlit as st

st.set_page_config(
    page_title="n-DoF Robot Designer",
    page_icon="🤖",
    layout="wide",
)

st.title("n-DoF Robot Designer")
st.write(
    "Design, analyze, and animate custom serial robots using revolute or prismatic joints."
)

st.markdown(
    """
    Use the sidebar to navigate between pages:
    - **Robot Designer & Dynamics**: Build the robot, compute inertia/torque budgets, and visualize motion.
    - **Screw Theory & Kinematics Tools**: Interactive calculators for twists, wrenches, and exponential coordinates.
    
    Built with reference to [screw theory](https://en.wikipedia.org/wiki/Screw_theory) and 
    Herman Bruyninckx's kinematics/dynamics notes.
    """
)

st.info(
    "Robots with more than 6 DoF will highlight redundant DoF both in the summary and the dynamic viewer."
)
