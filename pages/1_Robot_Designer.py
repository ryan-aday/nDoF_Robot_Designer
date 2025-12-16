import json
import math
from typing import List

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.robot import (
    build_robot,
    damped_least_squares_ik,
    gradient_descent_ik,
    joint_summary,
    newton_raphson_ik,
    screw_enhanced_ik,
)


def build_frame_data(robot, positions, transforms, target):
    end_effector = positions[-1]

    traces = [
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[0.0],
            mode="markers",
            marker=dict(size=6, color="#2ca02c"),
            name="Origin",
        ),
        go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="lines+markers",
            marker=dict(size=5, color="#1f77b4"),
            line=dict(width=5, color="#1f77b4"),
            name="Robot legs",
        ),
        go.Scatter3d(
            x=[end_effector[0]],
            y=[end_effector[1]],
            z=[end_effector[2]],
            mode="markers",
            marker=dict(size=7, color="#6a0dad"),
            name="End effector",
        ),
        go.Scatter3d(
            x=[target[0]],
            y=[target[1]],
            z=[target[2]],
            mode="markers",
            marker=dict(size=6, color="#d62728"),
            name="Target",
        ),
    ]
    return traces


def render_robot_plot(robot, positions: np.ndarray, transforms, target: np.ndarray, redundant: int) -> go.Figure:
    fig = go.Figure(data=build_frame_data(robot, positions, transforms, target))
    all_points = np.vstack([positions, target.reshape(1, 3), np.zeros((1, 3))])
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
    pad = 0.1 * max(x_max - x_min, y_max - y_min, z_max - z_min, 1e-3)
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            xaxis=dict(range=[x_min - pad, x_max + pad]),
            yaxis=dict(range=[y_min - pad, y_max + pad]),
            zaxis=dict(range=[z_min - pad, z_max + pad]),
            aspectmode="cube",
        ),
        height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        title=f"Robot visualization{' — ' + str(redundant) + ' redundant DoF' if redundant else ''}",
        uirevision="robot-view",
    )
    return fig


def main():
    st.title("Robot Designer & Dynamics")
    st.markdown(
        "References: [Screw theory](https://en.wikipedia.org/wiki/Screw_theory) | "
        "[Robot Kinematics and Dynamics](https://u0011821.pages.gitlab.kuleuven.be/robotics/2009-HermanBruyninckx-robot-kinematics-and-dynamics.pdf) | "
        "[Numerical IK (Modern Robotics)](https://modernrobotics.northwestern.edu/nu-gm-book-resource/6-2-numerical-inverse-kinematics-part-1-of-2/) | "
        "[Gradient descent in robotics](https://www.meegle.com/en_us/topics/gradient-descent/gradient-descent-in-robotics) | "
        "[Screw-theory improvements](https://journals.sagepub.com/doi/10.5772/60834) | "
        "[Trajectory smoothing](https://www.witpress.com/Secure/elibrary/papers/HPSM25/HPSM25011FU1.pdf) | "
        "[Manipulator optimization](https://www.sciencedirect.com/science/article/abs/pii/S0094114X05001424)"
    )
    st.sidebar.header("Robot setup")

    desired_dof = st.sidebar.slider("Desired DoF", 1, 12, 6)
    allow_prismatic = st.sidebar.checkbox("Allow prismatic joints", value=True)
    prismatic_count = st.sidebar.slider(
        "Prismatic joints (if allowed)", 0, desired_dof, min(desired_dof, 1)
    )

    homogeneous = st.sidebar.radio("Link sizing", ["Homogeneous", "Vary by joint"], index=0)
    limit_mode = st.sidebar.radio(
        "Joint range mode",
        ["Homogeneous limits", "Vary limits by joint"],
        index=0,
        help="Choose whether all joints share the same motion range or each joint can be tuned individually.",
    )
    default_length = st.sidebar.number_input("Default link length (m)", 0.05, 2.0, 0.25, 0.05)
    default_area = st.sidebar.number_input(
        "Default cross-sectional area (m²)", 1e-4, 0.2, 0.125, 0.0001,
        help="Used to compute inertia assuming a square cross section; can vary per joint.",
    )
    default_mass = st.sidebar.number_input("Default link mass (kg)", 0.1, 20.0, 1.5, 0.1)
    gravity = st.sidebar.number_input(
        "Gravity (m/s²)", min_value=0.0, max_value=30.0, value=9.81, step=0.01, help="Set the gravitational acceleration used for torque estimates."
    )
    default_revolute_min = st.sidebar.number_input(
        "Homogeneous revolute min (rad)", value=-math.pi, step=0.1, format="%.2f"
    )
    default_revolute_max = st.sidebar.number_input(
        "Homogeneous revolute max (rad)", value=math.pi, step=0.1, format="%.2f"
    )
    default_prismatic_min = st.sidebar.number_input(
        "Homogeneous prismatic min (m)", value=-default_length, step=0.05
    )
    default_prismatic_max = st.sidebar.number_input(
        "Homogeneous prismatic max (m)", value=default_length, step=0.05
    )

    default_types = ["prismatic" if allow_prismatic and i < prismatic_count else "revolute" for i in range(desired_dof)]

    def _default_start(idx: int, joint_type: str) -> float:
        if joint_type == "prismatic":
            return 0.05
        direction = 1 if idx % 2 == 0 else -1
        return 0.2 * direction

    editor_data = [
        {
            "Joint": i + 1,
            "Type": default_types[i] if i < len(default_types) else "revolute",
            "Length (m)": default_length,
            "Cross-sectional area (m²)": default_area,
            "Mass (kg)": default_mass,
            "Start (rad/m)": _default_start(i, default_types[i] if i < len(default_types) else "revolute"),
            "Min (rad/m)": default_revolute_min if default_types[i] == "revolute" else -default_length,
            "Max (rad/m)": default_revolute_max if default_types[i] == "revolute" else default_length,
            "Note": "",
        }
        for i in range(max(1, desired_dof))
    ]

    column_config = {
        "Type": st.column_config.SelectboxColumn(
            "Joint type",
            options=["revolute", "prismatic"] if allow_prismatic else ["revolute"],
            help="Toggle joint actuation type; defaults favor the minimal efficient mix.",
        ),
        "Length (m)": st.column_config.NumberColumn(min_value=0.01, step=0.01),
        "Cross-sectional area (m²)": st.column_config.NumberColumn(min_value=1e-4, step=1e-4),
        "Mass (kg)": st.column_config.NumberColumn(min_value=0.01, step=0.05),
        "Start (rad/m)": st.column_config.NumberColumn(
            help="Initial joint state (radians for revolute, meters for prismatic) to avoid singular home poses.",
            step=0.01,
        ),
        "Min (rad/m)": st.column_config.NumberColumn(
            help="Lower bound for the joint's motion (radians for revolute, meters for prismatic).",
            step=0.01,
        ),
        "Max (rad/m)": st.column_config.NumberColumn(
            help="Upper bound for the joint's motion (radians for revolute, meters for prismatic).",
            step=0.01,
        ),
        "Note": st.column_config.TextColumn(help="Add per-joint notes; they flow into the summary and report."),
    }

    edited = st.data_editor(
        editor_data,
        num_rows="dynamic",
        use_container_width=True,
        column_config=column_config,
        key="links_editor",
    )

    if len(edited) == 0:
        st.warning("Add at least one joint to continue.")
        return

    solved_dof = len(edited)
    link_lengths = [row["Length (m)"] for row in edited]
    link_areas = [row["Cross-sectional area (m²)"] for row in edited]
    link_masses = [row["Mass (kg)"] for row in edited]
    joint_types = [row["Type"] for row in edited]
    joint_notes = [row["Note"] for row in edited]
    start_states = [row.get("Start (rad/m)", 0.0) for row in edited]
    joint_mins = [row.get("Min (rad/m)") for row in edited]
    joint_maxes = [row.get("Max (rad/m)") for row in edited]

    invalid_starts: List[str] = []
    for idx, (jt, start, length, jmin, jmax) in enumerate(
        zip(joint_types, start_states, link_lengths, joint_mins, joint_maxes), start=1
    ):
        lower_bound = jmin if jmin is not None else (-math.pi if jt == "revolute" else -length)
        upper_bound = jmax if jmax is not None else (math.pi if jt == "revolute" else length)
        if lower_bound >= upper_bound:
            invalid_starts.append(f"Joint {idx}: min range must be less than max range.")
        if start < lower_bound or start > upper_bound:
            invalid_starts.append(
                f"Joint {idx}: start ({start:.2f}) is outside its range [{lower_bound:.2f}, {upper_bound:.2f}] and may block convergence."
            )
        if jt == "prismatic" and abs(start) > 2 * length:
            invalid_starts.append(
                f"Joint {idx}: prismatic start ({start:.2f} m) exceeds twice its link length ({length:.2f} m) and may be unreachable."
            )
    if invalid_starts:
        st.warning("Start states outside typical bounds:\n" + "\n".join(invalid_starts))

    if homogeneous == "Homogeneous":
        link_lengths = [default_length] * solved_dof
        link_areas = [default_area] * solved_dof
        link_masses = [default_mass] * solved_dof

    if limit_mode == "Homogeneous limits":
        joint_mins = [default_revolute_min if jt == "revolute" else default_prismatic_min for jt in joint_types]
        joint_maxes = [default_revolute_max if jt == "revolute" else default_prismatic_max for jt in joint_types]

    robot = build_robot(
        dof=solved_dof,
        allow_prismatic=allow_prismatic,
        prismatic_count=prismatic_count,
        link_lengths=link_lengths,
        link_areas=link_areas,
        link_masses=link_masses,
        joint_types=joint_types,
        joint_notes=joint_notes,
        joint_mins=joint_mins,
        joint_maxes=joint_maxes,
        gravity=gravity,
    )

    st.subheader("Joint solution")
    st.write(f"Minimum joints required: {robot.dof}")
    if robot.redundant_dof:
        st.warning(
            f"The requested DoF exceeds 6. There are {robot.redundant_dof} redundant DoF that will be highlighted in the viewer."
        )

    st.write("Joint notes:")
    for note in robot.notes:
        st.info(note)

    summary_table = joint_summary(robot)
    st.dataframe(summary_table, hide_index=True, use_container_width=True)

    payload_mass = st.number_input("Payload mass at end-effector (kg)", 0.0, 20.0, 0.0, 0.1)
    torque_budget = robot.torque_budget(payload_mass=payload_mass)
    st.write("Torque/force budget per joint (outermost to base):")
    st.write({f"J{i+1}": round(t, 3) for i, t in enumerate(torque_budget)})
    st.caption(f"Gravity applied: {robot.gravity:.3f} m/s² (Earth ≈ 9.81 m/s² by default).")

    st.divider()
    st.subheader("Dynamics-aware visualization")
    st.write(
        "Drag inside the plot to rotate the view. Define a 3D target relative to the robot origin, tune solver type/steps, and set motion ranges before playing the 60 FPS start→target motion (no stepwise jog buttons)."
    )
    st.caption("Use the start (rad/m) column to set a non-singular home pose; the animation interpolates from that pose to the target solution.")

    if "joint_states" not in st.session_state or len(st.session_state.joint_states) != robot.dof:
        st.session_state.joint_states = start_states
    elif len(start_states) == robot.dof and st.session_state.get("synced_from_editor") is not True:
        st.session_state.joint_states = start_states
    st.session_state.synced_from_editor = False

    target_point = st.session_state.get("target_point", np.array([sum(link_lengths), 0.0, 0.0]))

    max_reach = sum(link_lengths) + sum(abs(s) for jt, s in zip(joint_types, start_states) if jt == "prismatic")
    if np.linalg.norm(target_point) > max_reach + 1e-6:
        st.warning(
            "Target lies beyond the nominal reach of the chain; convergence may fail. Reduce distance or extend link lengths/prismatic travel."
        )

    max_steps = st.slider(
        "Solver steps/iterations",
        min_value=50,
        max_value=1200,
        value=400,
        step=25,
        help="Control how many update steps the chosen solver can take to seek the target.",
    )

    animation_frames = st.slider(
        "Animation frames from start to end",
        min_value=60,
        max_value=1800,
        value=360,
        step=30,
        help="Frames for the 60 FPS playback; each is an even slice between the home pose and the solved pose (regardless of solver iterations).",
    )

    solver = st.selectbox(
        "Inverse-kinematics solver",
        [
            "Damped least squares",
            "Newton-Raphson",
            "Gradient descent",
            "Screw-enhanced adaptive",
        ],
        index=0,
        help="Choose between classic numerical IK or a screw-theory-inspired adaptive variant from recent literature.",
    )

    with st.form("target_form"):
        col_tx, col_ty, col_tz = st.columns(3)
        with col_tx:
            target_x = st.number_input(
                "Target X relative to origin (m)", value=float(target_point[0]), step=0.05
            )
        with col_ty:
            target_y = st.number_input(
                "Target Y relative to origin (m)", value=float(target_point[1]), step=0.05
            )
        with col_tz:
            target_z = st.number_input(
                "Target Z relative to origin (m)", value=float(target_point[2]), step=0.05
            )
        submitted = st.form_submit_button("Update target")
        if submitted:
            target_point = np.array([target_x, target_y, target_z])
    st.session_state.target_point = target_point

    # IK solve
    if solver == "Damped least squares":
        ik_states, converged, trajectory = damped_least_squares_ik(
            robot, target_point, st.session_state.joint_states, damping=0.02, max_iters=max_steps
        )
    elif solver == "Newton-Raphson":
        ik_states, converged, trajectory = newton_raphson_ik(
            robot, target_point, st.session_state.joint_states, max_iters=max_steps
        )
    elif solver == "Gradient descent":
        ik_states, converged, trajectory = gradient_descent_ik(
            robot, target_point, st.session_state.joint_states, step_size=0.05, max_iters=max_steps
        )
    else:
        ik_states, converged, trajectory = screw_enhanced_ik(
            robot,
            target_point,
            st.session_state.joint_states,
            step_size=0.06,
            damping_base=0.015,
            momentum=0.12,
            max_iters=max_steps,
        )
    st.session_state.joint_states = ik_states.tolist()

    if not converged:
        st.error(
            "Inverse kinematics did not converge. Adjust start states, link sizes, or target location to improve reach."
        )

    transforms = robot.homogenous_transforms(st.session_state.joint_states)
    positions = np.array([t[:3, 3] for t in transforms])
    end_effector = positions[-1]
    fig = render_robot_plot(robot, positions, transforms, target_point, robot.redundant_dof)

    # Animation evenly interpolates between the home pose and the solved pose (not solver iterations)
    home_state = np.array(start_states)
    final_state = np.array(trajectory[-1]) if len(trajectory) else ik_states
    alphas = np.linspace(0.0, 1.0, num=animation_frames + 1)
    path_states = [home_state + alpha * (final_state - home_state) for alpha in alphas]

    frames: List[go.Frame] = []
    for i, state in enumerate(path_states):
        frame_transforms = robot.homogenous_transforms(state)
        frame_positions = np.array([t[:3, 3] for t in frame_transforms])
        frames.append(go.Frame(data=build_frame_data(robot, frame_positions, frame_transforms, target_point), name=f"frame{i}"))

    fig.frames = frames
    frame_names = [f.name for f in frames]
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "showactive": True,
                "bgcolor": "#d2b48c",
                "font": {"color": "#000", "size": 12},
                "buttons": [
                    {
                        "label": "Play 60 FPS loop",
                        "method": "animate",
                        "args": [
                            frame_names,
                            {
                                "frame": {"duration": 1000 / 60, "redraw": True},
                                "fromcurrent": True,
                                "mode": "immediate",
                                "transition": {"duration": 0, "easing": "linear"},
                            },
                        ],
                    }
                ],
            }
        ],
        sliders=[],
    )

    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    st.subheader("Reach target evaluation")
    st.write(
        f"Current end-effector position: {end_effector.round(3)} | Target: {target_point.round(3)} | Residual: {(target_point - end_effector).round(4)}"
    )

    st.subheader("Download report")
    report = {
        "desired_dof": desired_dof,
        "redundant_dof": robot.redundant_dof,
        "solver": solver,
        "joints": summary_table,
        "torque_budget": torque_budget,
        "gravity": robot.gravity,
        "joint_limits": list(zip(joint_mins, joint_maxes)),
        "target": target_point.tolist(),
        "end_effector": end_effector.tolist(),
        "converged": converged,
        "home_state": home_state.tolist(),
        "start_states": start_states,
        "ik_solution": ik_states.tolist(),
        "solver_max_steps": max_steps,
        "solver_trajectory": [state.tolist() for state in trajectory],
        "animation_frames": animation_frames,
        "animation_path": [state.tolist() for state in path_states],
    }
    st.download_button("Download JSON report", data=json.dumps(report, indent=2), file_name="robot_report.json")


if __name__ == "__main__":
    main()
