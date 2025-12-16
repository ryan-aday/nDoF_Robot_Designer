import json
from typing import List

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.robot import (
    build_robot,
    damped_least_squares_ik,
    joint_summary,
    spherical_adjust,
)


def render_robot_plot(positions: np.ndarray, target: np.ndarray, redundant: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="lines+markers",
            marker=dict(size=5, color="#1f77b4"),
            line=dict(width=5, color="#1f77b4"),
            name="Robot Chain",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[target[0]],
            y=[target[1]],
            z=[target[2]],
            mode="markers",
            marker=dict(size=6, color="#d62728"),
            name="Target",
        )
    )
    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        title=f"Robot visualization{' — ' + str(redundant) + ' redundant DoF' if redundant else ''}",
    )
    return fig


def main():
    st.title("Robot Designer & Dynamics")
    st.sidebar.header("Robot setup")

    desired_dof = st.sidebar.slider("Desired DoF", 1, 12, 6)
    allow_prismatic = st.sidebar.checkbox("Allow prismatic joints", value=True)
    prismatic_count = st.sidebar.slider(
        "Prismatic joints (if allowed)", 0, desired_dof, min(desired_dof, 1)
    )

    homogeneous = st.sidebar.radio("Link sizing", ["Homogeneous", "Vary by joint"], index=0)
    default_length = st.sidebar.number_input("Default link length (m)", 0.05, 2.0, 0.25, 0.05)
    default_width = st.sidebar.number_input("Default link width (m)", 0.01, 0.5, 0.05, 0.01)
    default_depth = st.sidebar.number_input("Default link depth (m)", 0.01, 0.5, 0.05, 0.01)
    default_mass = st.sidebar.number_input("Default link mass (kg)", 0.1, 20.0, 1.5, 0.1)

    solved_dof = max(1, desired_dof)

    if homogeneous == "Homogeneous":
        link_lengths = [default_length] * solved_dof
        link_widths = [default_width] * solved_dof
        link_depths = [default_depth] * solved_dof
        link_masses = [default_mass] * solved_dof
    else:
        editor_data = [
            {
                "Link": i + 1,
                "Length (m)": default_length,
                "Width (m)": default_width,
                "Depth (m)": default_depth,
                "Mass (kg)": default_mass,
            }
            for i in range(solved_dof)
        ]
        edited = st.data_editor(editor_data, num_rows="dynamic", key="links_editor")
        link_lengths = [row["Length (m)"] for row in edited]
        link_widths = [row["Width (m)"] for row in edited]
        link_depths = [row["Depth (m)"] for row in edited]
        link_masses = [row["Mass (kg)"] for row in edited]

    robot = build_robot(
        dof=solved_dof,
        allow_prismatic=allow_prismatic,
        prismatic_count=prismatic_count,
        link_lengths=link_lengths,
        link_widths=link_widths,
        link_depths=link_depths,
        link_masses=link_masses,
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

    st.divider()
    st.subheader("Dynamics-aware visualization")
    st.write(
        "Drag inside the plot to rotate the view. Use the Cartesian arrow buttons to jog the target. Spherical buttons adjust the radius."
    )

    if "joint_states" not in st.session_state or len(st.session_state.joint_states) != robot.dof:
        st.session_state.joint_states = [0.0] * robot.dof

    target_point = st.session_state.get("target_point", np.array([sum(link_lengths), 0.0, 0.0]))

    # Jog controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("← X-0.05 m"):
            target_point = target_point + np.array([-0.05, 0, 0])
        if st.button("→ X+0.05 m"):
            target_point = target_point + np.array([0.05, 0, 0])
    with col2:
        if st.button("↓ Y-0.05 m"):
            target_point = target_point + np.array([0, -0.05, 0])
        if st.button("↑ Y+0.05 m"):
            target_point = target_point + np.array([0, 0.05, 0])
    with col3:
        if st.button("↧ Z-0.05 m"):
            target_point = target_point + np.array([0, 0, -0.05])
        if st.button("↥ Z+0.05 m"):
            target_point = target_point + np.array([0, 0, 0.05])

    col4, col5 = st.columns(2)
    with col4:
        if st.button("Spherical: contract 0.05 m"):
            target_point = spherical_adjust(target_point, -0.05)
    with col5:
        if st.button("Spherical: expand 0.05 m"):
            target_point = spherical_adjust(target_point, 0.05)

    st.session_state.target_point = target_point

    # IK solve
    ik_states, converged = damped_least_squares_ik(
        robot, target_point, st.session_state.joint_states, damping=0.02, max_iters=400
    )
    st.session_state.joint_states = ik_states.tolist()

    if not converged:
        st.error(
            "Inverse kinematics did not converge. Adjust link sizes or target location to improve reach."
        )

    positions = robot.joint_positions(st.session_state.joint_states)
    fig = render_robot_plot(positions, target_point, robot.redundant_dof)

    # Animation between home and target
    home_state = np.zeros(robot.dof)
    frames: List[go.Frame] = []
    n_frames = 20
    for i in range(n_frames + 1):
        alpha = i / n_frames
        interp_state = (1 - alpha) * home_state + alpha * ik_states
        frame_positions = robot.joint_positions(interp_state)
        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(
                        x=frame_positions[:, 0],
                        y=frame_positions[:, 1],
                        z=frame_positions[:, 2],
                        mode="lines+markers",
                    )
                ],
                name=f"frame{i}",
            )
        )

    fig.frames = frames
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "showactive": True,
                "buttons": [
                    {
                        "label": "Play loop",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 200, "redraw": True}, "fromcurrent": True, "mode": "immediate"}],
                    }
                ],
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "label": str(i),
                        "method": "animate",
                        "args": [[f"frame{i}"]],
                    }
                    for i in range(n_frames + 1)
                ]
            }
        ],
    )

    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    st.subheader("Reach target evaluation")
    end_effector = positions[-1]
    st.write(
        f"Current end-effector position: {end_effector.round(3)} | Target: {target_point.round(3)} | Residual: {(target_point - end_effector).round(4)}"
    )

    st.subheader("Download report")
    report = {
        "desired_dof": desired_dof,
        "redundant_dof": robot.redundant_dof,
        "joints": summary_table,
        "torque_budget": torque_budget,
        "target": target_point.tolist(),
        "end_effector": end_effector.tolist(),
        "converged": converged,
    }
    st.download_button("Download JSON report", data=json.dumps(report, indent=2), file_name="robot_report.json")


if __name__ == "__main__":
    main()
