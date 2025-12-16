# n-DoF Robot Designer

A Streamlit app for designing serial n-DoF robots, solving inverse kinematics with screw-theory inspired tools, and animating the resulting motion.

## Features
- Build robots with user-defined DoF, automatically flagging redundancy beyond 6 DoF.
- Choose revolute-only or mixed prismatic/revolute joint strategies to achieve the requested DoF with the minimum joint count.
- Configure homogeneous or per-joint beam lengths, cross-sectional areas, masses, and motor torque/force budgets.
- Compute inertia tensors from length and cross-sectional area (square cross-section assumption), torque estimates, and a basic torque budget including payloads and configurable gravity (default Earth 9.81 m/s²).
- Interactive IK solvers (damped least squares, Newton-Raphson, gradient descent, a basic matrix-projection solver, a screw-inspired adaptive solver, and a forward-kinematics finite-difference solver) with convergence feedback, adjustable iteration budgets, monotonic step acceptance to prevent divergence, manipulability checks, and residual reporting.
- Inspect the pure matrix transform from the current end-effector pose to the target (translation-only target frame) to verify the required motion before solving.
- Configure homogeneous or per-joint motion limits (singular poses are allowed) and replay an evenly interpolated start→target 60 FPS motion (blue legs, purple end-effector, red target, green origin) from user-defined home poses relative to the robot origin (camera locked during playback).
- Visualize the reachable workspace via a sampled point cloud to understand why targets may be unreachable under the current limits and axes.
- Screw theory calculators for twists, exponential coordinates, and wrench inspection.
- Downloadable JSON report capturing the solved kinematics and dynamics summary, plus a C code generator to drive start→target→start motion on hardware.

## Running the app
```bash
pip install -r requirements.txt  # or ensure streamlit, numpy, and plotly are available
streamlit run app.py
```

## Notes on kinematics and dynamics
- Joint axes alternate across x, y, z to improve workspace coverage and expose redundancy visually.
- Torque estimates use a simple gravity loading model across the serial chain; treat them as sizing guidance, not detailed FEA.
- Inverse kinematics defaults to damped least squares but also exposes Newton-Raphson and gradient-descent variants; adjust link lengths or targets if convergence warnings appear.

## References
- Wikipedia contributors. [Screw theory](https://en.wikipedia.org/wiki/Screw_theory).
- Herman Bruyninckx. [Robot Kinematics and Dynamics](https://u0011821.pages.gitlab.kuleuven.be/robotics/2009-HermanBruyninckx-robot-kinematics-and-dynamics.pdf).
- Modern Robotics. [Numerical inverse kinematics](https://modernrobotics.northwestern.edu/nu-gm-book-resource/6-2-numerical-inverse-kinematics-part-1-of-2/).
- Meegle. [Gradient descent in robotics](https://www.meegle.com/en_us/topics/gradient-descent/gradient-descent-in-robotics).
- He et al. [Improved screw theory for manipulator motion](https://journals.sagepub.com/doi/10.5772/60834).
- Fu et al. [Trajectory smoothing for manipulators](https://www.witpress.com/Secure/elibrary/papers/HPSM25/HPSM25011FU1.pdf).
- Tabak & Moosavian. [Redundant manipulator optimization](https://www.sciencedirect.com/science/article/abs/pii/S0094114X05001424).
- Addison. [Homogeneous transforms with Denavit–Hartenberg](https://automaticaddison.com/homogeneous-transformation-matrices-using-denavit-hartenberg/).
