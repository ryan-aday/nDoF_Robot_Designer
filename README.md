# n-DoF Robot Designer

A Streamlit app for designing serial n-DoF robots, solving kinematics with forward-kinematics-driven tools, and animating the resulting motion.

## Features
- Build robots with user-defined DoF, automatically flagging redundancy beyond 6 DoF.
- Choose revolute-only or mixed prismatic/revolute joint strategies to achieve the requested DoF with the minimum joint count.
- Configure homogeneous or per-joint beam lengths, cross-sectional areas, masses, and motor torque/force budgets.
- Tune homogeneous or per-joint joint body offsets, masses, and inertia tensors that shape the visualization and feed the torqu
e budget.
- Compute inertia tensors from length and cross-sectional area (square cross-section assumption), torque estimates, and a basic torque budget including payloads and configurable gravity (default Earth 9.81 m/s²).
- Forward finite-difference solver (DH-inspired) is the authoritative default, with legacy IK solvers (damped least squares, Newton-Raphson, gradient descent, matrix projection, screw-adaptive) retained for diagnostics; convergence feedback, adjustable iteration budgets, monotonic step acceptance, manipulability checks, and residual reporting are surfaced.
- Inspect the pure matrix transform from the current end-effector pose to the target (translation-only target frame) to verify the required motion before solving.
- Configure homogeneous or per-joint motion limits (singular poses are allowed) and replay an evenly interpolated start→target 60 FPS motion (blue legs, purple end-effector, red target, green origin) from user-defined home poses relative to the robot origin (camera locked during playback) while inspecting an end-effector motion profile plot.
- Display the solved joint states (radians/meters and degrees for revolute joints) alongside the forward solution for traceability against inverse-kinematics references.
- Visualize the reachable workspace via a sampled point cloud surfaced into an opaque convex envelope to understand why targets may be unreachable under the current limits and axes.
- Screw theory and forward-kinematics formulas page for twists, exponential coordinates, DH snippets, and wrench inspection.
- Axial Flux Motor Designer for early geometry/pole sizing (shear-stress based torque sizing, pole heuristics, magnet band estimates) with references to coreless motor design literature.
- Downloadable JSON report capturing the solved kinematics and dynamics summary, plus a multi-target code generator (C/C++/FANUC/KUKA pseudo) to drive start→target→start motion on hardware.

## Running the app
```bash
pip install -r requirements.txt  # or ensure streamlit, numpy, plotly, and scipy are available
streamlit run app.py
```

## Notes on kinematics and dynamics
- Joint axes alternate across x, y, z to improve workspace coverage and expose redundancy visually.
- Torque estimates use a simple gravity loading model across the serial chain; treat them as sizing guidance, not detailed FEA.
- Kinematics defaults to the forward finite-difference solver that leans on repeated DH-style transforms; legacy inverse-kinematics variants remain for diagnostics only.

## References
- Wikipedia contributors. [Screw theory](https://en.wikipedia.org/wiki/Screw_theory).
- Herman Bruyninckx. [Robot Kinematics and Dynamics](https://u0011821.pages.gitlab.kuleuven.be/robotics/2009-HermanBruyninckx-robot-kinematics-and-dynamics.pdf).
- Modern Robotics. [Numerical inverse kinematics](https://modernrobotics.northwestern.edu/nu-gm-book-resource/6-2-numerical-inverse-kinematics-part-1-of-2/).
- MathWorks. [Inverse kinematics overview](https://www.mathworks.com/discovery/inverse-kinematics.html).
- Meegle. [Gradient descent in robotics](https://www.meegle.com/en_us/topics/gradient-descent/gradient-descent-in-robotics).
- He et al. [Improved screw theory for manipulator motion](https://journals.sagepub.com/doi/10.5772/60834).
- Fu et al. [Trajectory smoothing for manipulators](https://www.witpress.com/Secure/elibrary/papers/HPSM25/HPSM25011FU1.pdf).
- Tabak & Moosavian. [Redundant manipulator optimization](https://www.sciencedirect.com/science/article/abs/pii/S0094114X05001424).
- Addison. [Homogeneous transforms with Denavit–Hartenberg](https://automaticaddison.com/homogeneous-transformation-matrices-using-denavit-hartenberg/).
- Batzel et al. [Coreless axial-flux motor sizing](https://cd14.ijme.us/papers/088__Todd%20D.%20Batzel,%20Andrew%20M.%20Skraba,%20Ray%20D.%20Massi.pdf).
- Caden Kraft. [Designing a coreless axial flux motor](https://cadenkraft.com/designing-a-coreless-axial-flux-motor-part-1/).
