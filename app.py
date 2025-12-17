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
    - **Fundamental Formulas**: Interactive calculators for twists, wrenches, forward-kinematics snippets, and exponential coordinates.
    - **C Code Generator**: Turn your JSON report into scaffolded C to drive hardware start→target→start motion.
    - **Axial Flux Motor Designer**: Quick geometry and pole-count sizing for coreless axial flux machines.

    Built with reference to [screw theory](https://en.wikipedia.org/wiki/Screw_theory),
    Herman Bruyninckx's [Robot Kinematics and Dynamics](https://u0011821.pages.gitlab.kuleuven.be/robotics/2009-HermanBruyninckx-robot-kinematics-and-dynamics.pdf),
    [Modern Robotics numerical IK](https://modernrobotics.northwestern.edu/nu-gm-book-resource/6-2-numerical-inverse-kinematics-part-1-of-2/),
    notes on [gradient descent in robotics](https://www.meegle.com/en_us/topics/gradient-descent/gradient-descent-in-robotics),
    and screw-theory refinements such as [He et al.](https://journals.sagepub.com/doi/10.5772/60834),
    [trajectory smoothing for manipulators](https://www.witpress.com/Secure/elibrary/papers/HPSM25/HPSM25011FU1.pdf),
    [redundant manipulator optimization](https://www.sciencedirect.com/science/article/abs/pii/S0094114X05001424),
    [DH-style forward kinematics](https://automaticaddison.com/homogeneous-transformation-matrices-using-denavit-hartenberg/),
    axial-flux motor sizing notes from [Batzel et al.](https://cd14.ijme.us/papers/088__Todd%20D.%20Batzel,%20Andrew%20M.%20Skraba,%20Ray%20D.%20Massi.pdf),
    and Caden Kraft's [coreless axial flux walkthrough](https://cadenkraft.com/designing-a-coreless-axial-flux-motor-part-1/).
    """
)

st.info(
    "Robots with more than 6 DoF will highlight redundant DoF both in the summary and the dynamic viewer."
)
