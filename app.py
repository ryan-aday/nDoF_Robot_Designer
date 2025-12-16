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
    - **C Code Generator**: Turn your JSON report into scaffolded C to drive hardware start→target→start motion.

    Built with reference to [screw theory](https://en.wikipedia.org/wiki/Screw_theory),
    Herman Bruyninckx's [Robot Kinematics and Dynamics](https://u0011821.pages.gitlab.kuleuven.be/robotics/2009-HermanBruyninckx-robot-kinematics-and-dynamics.pdf),
    [Modern Robotics numerical IK](https://modernrobotics.northwestern.edu/nu-gm-book-resource/6-2-numerical-inverse-kinematics-part-1-of-2/),
    notes on [gradient descent in robotics](https://www.meegle.com/en_us/topics/gradient-descent/gradient-descent-in-robotics),
    and screw-theory refinements such as [He et al.](https://journals.sagepub.com/doi/10.5772/60834),
    [trajectory smoothing for manipulators](https://www.witpress.com/Secure/elibrary/papers/HPSM25/HPSM25011FU1.pdf),
    [redundant manipulator optimization](https://www.sciencedirect.com/science/article/abs/pii/S0094114X05001424),
    and [DH-style forward kinematics](https://automaticaddison.com/homogeneous-transformation-matrices-using-denavit-hartenberg/).
    """
)

st.info(
    "Robots with more than 6 DoF will highlight redundant DoF both in the summary and the dynamic viewer."
)
