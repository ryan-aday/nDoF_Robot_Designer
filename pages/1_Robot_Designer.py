import json
import math
from typing import List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.spatial import ConvexHull, QhullError

from utils.robot import (
    build_robot,
    damped_least_squares_ik,
    forward_finite_ik,
    gradient_descent_ik,
    joint_summary,
    matrix_projection_ik,
    newton_raphson_ik,
    manipulability_index,
    reachability_report,
    sample_workspace_points,
    screw_enhanced_ik,
    transform_from_target_to_start,
)


def build_frame_data(
    robot,
    positions,
    transforms,
    target,
    path_points: Optional[np.ndarray] = None,
):
    end_effector = positions[-1]

    legs_x: List[float] = []
    legs_y: List[float] = []
    legs_z: List[float] = []
    revolute_x: List[float] = []
    revolute_y: List[float] = []
    revolute_z: List[float] = []

    for idx in range(1, len(positions)):
        prev = positions[idx - 1]
        curr = positions[idx]
        joint = robot.joints[idx - 1]
        link = robot.links[idx - 1]

        if joint.joint_type == "revolute":
            axis_world = transforms[idx - 1][:3, :3] @ joint.axis
            axis_norm = np.linalg.norm(axis_world)
            if axis_norm < 1e-9:
                axis_world = joint.axis
                axis_norm = np.linalg.norm(axis_world) + 1e-9
            axis_world = axis_world / axis_norm

            cyl_length = max(0.02, robot.joints[idx - 1].body_length)
            body_tip = prev + axis_world * cyl_length

            # Keep legs connected at the joint midpoint and suppress the opposite extension
            legs_x.extend([prev[0], curr[0], None])
            legs_y.extend([prev[1], curr[1], None])
            legs_z.extend([prev[2], curr[2], None])

            revolute_x.extend([prev[0], body_tip[0], None])
            revolute_y.extend([prev[1], body_tip[1], None])
            revolute_z.extend([prev[2], body_tip[2], None])
        else:
            legs_x.extend([prev[0], curr[0], None])
            legs_y.extend([prev[1], curr[1], None])
            legs_z.extend([prev[2], curr[2], None])

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
            x=legs_x,
            y=legs_y,
            z=legs_z,
            mode="lines+markers",
            marker=dict(size=5, color="#1f77b4"),
            line=dict(width=5, color="#1f77b4"),
            name="Robot legs",
            connectgaps=False,
        ),
    ]

    if revolute_x:
        traces.append(
            go.Scatter3d(
                x=revolute_x,
                y=revolute_y,
                z=revolute_z,
                mode="lines",
                line=dict(width=6, color="#2ca02c"),
                name="Revolute joint body",
                connectgaps=False,
            )
        )

    traces.extend(
        [
            go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode="markers",
                marker=dict(size=4, color="#1f77b4"),
                name="Joint nodes",
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
    )

    if path_points is not None and len(path_points) > 0:
        traces.append(
            go.Scatter3d(
                x=path_points[:, 0],
                y=path_points[:, 1],
                z=path_points[:, 2],
                mode="lines",
                line=dict(color="#7eb6ff", width=4),
                name="End-effector path",
            )
        )
    return traces


def render_robot_plot(
    robot,
    positions: np.ndarray,
    transforms,
    target: np.ndarray,
    redundant: int,
    path_points: Optional[np.ndarray] = None,
) -> go.Figure:
    fig = go.Figure(data=build_frame_data(robot, positions, transforms, target, path_points))
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
        "[Inverse kinematics overview](https://www.mathworks.com/discovery/inverse-kinematics.html) | "
        "[Gradient descent in robotics](https://www.meegle.com/en_us/topics/gradient-descent/gradient-descent-in-robotics) | "
        "[Screw-theory improvements](https://journals.sagepub.com/doi/10.5772/60834) | "
        "[Trajectory smoothing](https://www.witpress.com/Secure/elibrary/papers/HPSM25/HPSM25011FU1.pdf) | "
        "[Manipulator optimization](https://www.sciencedirect.com/science/article/abs/pii/S0094114X05001424) | "
        "[Forward kinematics via DH](https://automaticaddison.com/homogeneous-transformation-matrices-using-denavit-hartenberg/)"
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
    default_joint_offset = st.sidebar.number_input(
        "Default joint body length (m)", min_value=0.0, max_value=0.5, value=0.1, step=0.01,
        help="Controls the green joint body length used for offsets and torque/inertia aggregation.",
    )
    default_joint_mass = st.sidebar.number_input(
        "Default joint mass (kg)", min_value=0.0, max_value=10.0, value=0.5, step=0.05,
        help="Mass attributed to the joint body; included in torque/force budgets.",
    )
    default_joint_inertia = st.sidebar.number_input(
        "Default joint inertia (kg·m²)", min_value=0.0, max_value=5.0, value=0.02, step=0.01,
        help="Diagonal inertia term applied about the joint body's principal axes.",
    )
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

    axis_options = ["X", "Y", "Z"]

    editor_data = [
        {
            "Joint": i + 1,
            "Type": default_types[i] if i < len(default_types) else "revolute",
            "Axis": axis_options[i % len(axis_options)],
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
        "Axis": st.column_config.SelectboxColumn(
            "Rotation/translation axis",
            options=axis_options,
            help="Set the revolute rotation axis or prismatic translation axis per joint (XYZ basis).",
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
    joint_axes = [row.get("Axis", "X") for row in edited]
    joint_notes = [row["Note"] for row in edited]
    start_states = [row.get("Start (rad/m)", 0.0) for row in edited]
    joint_mins = [row.get("Min (rad/m)") for row in edited]
    joint_maxes = [row.get("Max (rad/m)") for row in edited]

    st.markdown("**Joint body offsets, mass, and inertia**")
    joint_body_mode = st.radio(
        "Joint body configuration",
        ["Homogeneous", "Vary by joint"],
        index=0,
        help="Control the physical joint body length, mass, and inertia that feed the torque budget and visualization offsets.",
    )

    body_table_defaults = [
        {
            "Joint": i + 1,
            "Offset length (m)": default_joint_offset,
            "Joint mass (kg)": default_joint_mass,
            "Joint inertia Ix (kg·m²)": default_joint_inertia,
            "Joint inertia Iy (kg·m²)": default_joint_inertia,
            "Joint inertia Iz (kg·m²)": default_joint_inertia,
        }
        for i in range(solved_dof)
    ]

    joint_body_lengths = [default_joint_offset] * solved_dof
    joint_body_masses = [default_joint_mass] * solved_dof
    joint_body_inertias = [(default_joint_inertia, default_joint_inertia, default_joint_inertia)] * solved_dof

    joint_body_column_config = {
        "Offset length (m)": st.column_config.NumberColumn(min_value=0.0, step=0.01),
        "Joint mass (kg)": st.column_config.NumberColumn(min_value=0.0, step=0.05),
        "Joint inertia Ix (kg·m²)": st.column_config.NumberColumn(min_value=0.0, step=0.01),
        "Joint inertia Iy (kg·m²)": st.column_config.NumberColumn(min_value=0.0, step=0.01),
        "Joint inertia Iz (kg·m²)": st.column_config.NumberColumn(min_value=0.0, step=0.01),
    }

    if joint_body_mode == "Vary by joint":
        joint_body_rows = st.data_editor(
            body_table_defaults,
            num_rows="fixed",
            hide_index=True,
            use_container_width=True,
            column_config=joint_body_column_config,
            key="joint_body_editor",
        )

        joint_body_lengths = [row.get("Offset length (m)", default_joint_offset) for row in joint_body_rows]
        joint_body_masses = [row.get("Joint mass (kg)", default_joint_mass) for row in joint_body_rows]
        joint_body_inertias = [
            (
                row.get("Joint inertia Ix (kg·m²)", default_joint_inertia),
                row.get("Joint inertia Iy (kg·m²)", default_joint_inertia),
                row.get("Joint inertia Iz (kg·m²)", default_joint_inertia),
            )
            for row in joint_body_rows
        ]
    else:
        st.data_editor(
            body_table_defaults,
            num_rows="fixed",
            hide_index=True,
            use_container_width=True,
            column_config=joint_body_column_config,
            key="joint_body_editor_locked",
            disabled=True,
        )

    axis_lookup = {"X": np.array([1.0, 0.0, 0.0]), "Y": np.array([0.0, 1.0, 0.0]), "Z": np.array([0.0, 0.0, 1.0])}
    joint_axis_vectors = [axis_lookup.get(axis_key, np.array([1.0, 0.0, 0.0])) for axis_key in joint_axes]

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
        joint_axes=joint_axis_vectors,
        joint_notes=joint_notes,
        joint_mins=joint_mins,
        joint_maxes=joint_maxes,
        joint_body_lengths=joint_body_lengths,
        joint_body_masses=joint_body_masses,
        joint_body_inertias=joint_body_inertias,
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
        "Drag inside the plot to rotate the view. Define a 3D target relative to the robot origin, tune solver steps, and play the 60 FPS start→target motion (no stepwise jog buttons)."
    )
    st.caption(
        "The forward finite (DH-inspired) solver is the default and authoritative placement method. Legacy inverse-kinematics functions remain available for diagnostics but are not relied on for target placement."
    )
    st.caption("Use the start (rad/m) column to set your preferred home pose (singular poses are allowed); the animation interpolates from that pose to the target solution.")

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
        help="Control how many update steps the forward finite solver can take to seek the target.",
    )

    animation_frames = st.slider(
        "Animation frames from start to end",
        min_value=60,
        max_value=1800,
        value=60,
        step=30,
        help="Frames for the 60 FPS playback; each is an even slice between the home pose and the solved pose (regardless of solver iterations).",
    )

    diagnostic_solver = st.selectbox(
        "Optional legacy IK diagnostic (not used for placement)",
        [
            "None (use forward finite only)",
            "Damped least squares",
            "Newton-Raphson",
            "Gradient descent",
            "Screw-enhanced adaptive",
            "Matrix projection",
        ],
        index=0,
        help="Legacy inverse-kinematics routines are available for troubleshooting but are not relied upon for end-effector placement.",
    )

    solver = "Forward finite (DH-based default)"

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

    delta_tf = transform_from_target_to_start(robot, target_point, start_states)
    with st.expander("Target-to-start transform (for transparency)"):
        st.write(
            "Relative transform from the current end-effector pose to the target (pure translation target frame):"
        )
        st.write(delta_tf)

    manip, cond = manipulability_index(robot, start_states)
    if cond > 500:
        st.warning(
            f"Home-pose manipulability is low (geometric mean {manip:.4e}, condition number {cond:.1f}); solver damping will try to avoid singularities, but consider adjusting joint axes or start angles."
        )
    else:
        st.info(
            f"Home-pose manipulability metric: {manip:.4e} (condition number {cond:.1f})."
        )

    diag_residual_value = None
    # IK solve
    feasible, reachability_reasons = reachability_report(robot, target_point, start_states)
    if not feasible:
        st.error("Target is unreachable with the current geometry:\n" + "\n".join(reachability_reasons))
        ik_states = np.array(start_states)
        trajectory: List[np.ndarray] = [ik_states.copy()]
        converged = False
        diag_result = None
    else:
        ik_states, converged, trajectory = forward_finite_ik(
            robot, target_point, st.session_state.joint_states, max_iters=max_steps
        )

        diag_result = None
        if diagnostic_solver != "None (use forward finite only)":
            if diagnostic_solver == "Damped least squares":
                diag_result = damped_least_squares_ik(
                    robot, target_point, st.session_state.joint_states, damping=0.02, max_iters=max_steps
                )
            elif diagnostic_solver == "Newton-Raphson":
                diag_result = newton_raphson_ik(
                    robot, target_point, st.session_state.joint_states, max_iters=max_steps
                )
            elif diagnostic_solver == "Gradient descent":
                diag_result = gradient_descent_ik(
                    robot, target_point, st.session_state.joint_states, step_size=0.05, max_iters=max_steps
                )
            elif diagnostic_solver == "Matrix projection":
                diag_result = matrix_projection_ik(
                    robot, target_point, st.session_state.joint_states, max_iters=max_steps
                )
            else:
                diag_result = screw_enhanced_ik(
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
            final_error = float(np.linalg.norm(target_point - robot.homogenous_transforms(ik_states)[-1][:3, 3]))
            at_limits = [
                f"Joint {i+1} at limit ({state:.3f} within bounds {j.min_state:.3f}–{j.max_state:.3f})"
                for i, (state, j) in enumerate(zip(ik_states, robot.joints))
                if abs(state - j.min_state) < 1e-6 or abs(state - j.max_state) < 1e-6
            ]
            reason_lines = [f"Residual remained at {final_error:.4f} m despite monotonic steps."]
            if at_limits:
                reason_lines.append("Joint limits reached: " + ", ".join(at_limits))
            st.error("Forward finite solver did not converge:\n" + "\n".join(reason_lines))
        if diag_result:
            diag_states, diag_converged, _ = diag_result
            diag_residual = float(np.linalg.norm(target_point - robot.homogenous_transforms(diag_states)[-1][:3, 3]))
            diag_residual_value = diag_residual
            st.info(
                f"Diagnostic {diagnostic_solver} residual: {diag_residual:.4f} m (converged={diag_converged}). "
                "Legacy solvers are provided for analysis only and are not used for placement."
            )

    transforms = robot.homogenous_transforms(st.session_state.joint_states)
    positions = np.array([t[:3, 3] for t in transforms])
    end_effector = positions[-1]
    # Animation evenly interpolates between the home pose and the solved pose (not solver iterations)
    home_state = np.array(start_states)
    final_state = np.array(trajectory[-1]) if len(trajectory) else ik_states
    alphas = np.linspace(0.0, 1.0, num=animation_frames + 1)
    path_states = [home_state + alpha * (final_state - home_state) for alpha in alphas]
    path_end_positions = np.array(
        [robot.homogenous_transforms(state)[-1][:3, 3] for state in path_states]
    )

    st.markdown("### Final joint states (rad/m and deg)")
    st.caption(
        "Solved joint configuration reported for traceability (see MathWorks inverse kinematics overview for why joint solutions matter)."
    )
    final_rows = []
    for idx, (state, joint, axis_key) in enumerate(zip(final_state, robot.joints, joint_axes), start=1):
        lower, upper = joint.min_state, joint.max_state
        within = lower - 1e-9 <= state <= upper + 1e-9
        final_rows.append(
            {
                "Joint": idx,
                "Type": joint.joint_type,
                "Axis": axis_key,
                "State (rad/m)": round(float(state), 5),
                "State (deg)": round(math.degrees(state), 3) if joint.joint_type == "revolute" else "—",
                "Within limits": "Yes" if within else "No",
            }
        )
    st.dataframe(final_rows, hide_index=True, use_container_width=True)

    fig = render_robot_plot(
        robot, positions, transforms, target_point, robot.redundant_dof, path_end_positions
    )

    frames: List[go.Frame] = []
    for i, state in enumerate(path_states):
        frame_transforms = robot.homogenous_transforms(state)
        frame_positions = np.array([t[:3, 3] for t in frame_transforms])
        frames.append(
            go.Frame(
                data=build_frame_data(
                    robot, frame_positions, frame_transforms, target_point, path_end_positions
                ),
                name=f"frame{i}",
            )
        )

    fig.update(frames=frames)
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
                            None,
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

    st.subheader("End-effector motion profile")
    trajectory_positions = [robot.homogenous_transforms(state)[-1][:3, 3] for state in trajectory]
    residuals = [float(np.linalg.norm(target_point - pos)) for pos in trajectory_positions]
    profile_fig = go.Figure()
    profile_fig.add_trace(
        go.Scatter(
            x=list(range(len(residuals))),
            y=residuals,
            mode="lines+markers",
            name="Residual distance (m)",
        )
    )
    profile_fig.update_layout(
        xaxis_title="Iteration",
        yaxis_title="Residual to target (m)",
        title="Residual decay during forward finite solving",
        height=300,
    )
    st.plotly_chart(profile_fig, use_container_width=True)

    st.subheader("Workspace envelope")
    cloud_samples = st.slider(
        "Number of workspace samples",
        min_value=200,
        max_value=4000,
        value=1200,
        step=200,
        help="Random joint-state samples used to visualize the reachable workspace as a point cloud before surfacing.",
    )

    if "workspace_cloud" not in st.session_state:
        st.session_state.workspace_cloud = sample_workspace_points(robot, samples=cloud_samples)

    regenerate = st.button(
        "Generate workspace cloud", help="Refresh the workspace samples with the current limits and axes."
    )
    if regenerate:
        st.session_state.workspace_cloud = sample_workspace_points(robot, samples=cloud_samples)

    cloud_points = st.session_state.get("workspace_cloud", np.empty((0, 3)))
    if not regenerate and len(cloud_points) != cloud_samples:
        st.session_state.workspace_cloud = sample_workspace_points(robot, samples=cloud_samples)
        cloud_points = st.session_state.workspace_cloud
    cloud_fig = go.Figure()

    hull_added = False
    if len(cloud_points) >= 4:
        try:
            hull = ConvexHull(cloud_points)
            cloud_fig.add_trace(
                go.Mesh3d(
                    x=cloud_points[:, 0],
                    y=cloud_points[:, 1],
                    z=cloud_points[:, 2],
                    i=hull.simplices[:, 0],
                    j=hull.simplices[:, 1],
                    k=hull.simplices[:, 2],
                    color="#7f7f7f",
                    opacity=0.4,
                    name="Workspace envelope",
                    flatshading=True,
                )
            )
            hull_added = True
        except QhullError:
            st.warning("Workspace surface could not be generated (degenerate sample); showing markers instead.")

    if not hull_added and len(cloud_points) > 0:
        cloud_fig.add_trace(
            go.Scatter3d(
                x=cloud_points[:, 0],
                y=cloud_points[:, 1],
                z=cloud_points[:, 2],
                mode="markers",
                marker=dict(size=2, color="rgba(100,100,100,0.45)"),
                name="Workspace samples",
            )
        )

    cloud_fig.add_trace(
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[0.0],
            mode="markers",
            marker=dict(size=6, color="#2ca02c"),
            name="Origin",
        )
    )
    cloud_fig.add_trace(
        go.Scatter3d(
            x=[end_effector[0]],
            y=[end_effector[1]],
            z=[end_effector[2]],
            mode="markers",
            marker=dict(size=7, color="#6a0dad"),
            name="End effector",
        )
    )
    cloud_fig.add_trace(
        go.Scatter3d(
            x=[target_point[0]],
            y=[target_point[1]],
            z=[target_point[2]],
            mode="markers",
            marker=dict(size=6, color="#d62728"),
            name="Target",
        )
    )

    if len(cloud_points) > 0:
        combined = np.vstack([cloud_points, target_point.reshape(1, 3), end_effector.reshape(1, 3), np.zeros((1, 3))])
        x_min, x_max = combined[:, 0].min(), combined[:, 0].max()
        y_min, y_max = combined[:, 1].min(), combined[:, 1].max()
        z_min, z_max = combined[:, 2].min(), combined[:, 2].max()
        pad = 0.1 * max(x_max - x_min, y_max - y_min, z_max - z_min, 1e-3)
    else:
        pad = 0.1
        x_min = y_min = z_min = -0.5
        x_max = y_max = z_max = 0.5

    cloud_fig.update_layout(
        scene=dict(
            xaxis=dict(range=[x_min - pad, x_max + pad]),
            yaxis=dict(range=[y_min - pad, y_max + pad]),
            zaxis=dict(range=[z_min - pad, z_max + pad]),
            aspectmode="cube",
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        title="Reachable workspace envelope",
        uirevision="workspace-view",
    )

    st.plotly_chart(cloud_fig, use_container_width=True, config={"scrollZoom": True})

    st.subheader("Download report")
    report = {
        "desired_dof": desired_dof,
        "redundant_dof": robot.redundant_dof,
        "solver": solver,
        "diagnostic_solver": diagnostic_solver,
        "diagnostic_residual": diag_residual_value,
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
        "final_joint_states_rad": final_state.tolist(),
        "final_joint_states_deg": [math.degrees(s) if jt == "revolute" else None for s, jt in zip(final_state, joint_types)],
        "solver_max_steps": max_steps,
        "solver_trajectory": [state.tolist() for state in trajectory],
        "animation_frames": animation_frames,
        "animation_path": [state.tolist() for state in path_states],
        "end_effector_path": path_end_positions.tolist(),
        "joint_body_lengths": joint_body_lengths,
        "joint_body_masses": joint_body_masses,
        "joint_body_inertias": [list(inertia) for inertia in joint_body_inertias],
    }
    st.download_button("Download JSON report", data=json.dumps(report, indent=2), file_name="robot_report.json")


if __name__ == "__main__":
    main()
