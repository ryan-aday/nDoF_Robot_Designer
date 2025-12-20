"""
Live controller and field robot interface for UART/Arduino bridge.
"""

import json
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


@dataclass
class JointRow:
    joint_number: int
    joint_type: str
    axis: np.ndarray
    leg_length: float


def parse_axis(token: str) -> np.ndarray:
    token = token.strip().lower().replace("[", "").replace("]", "")
    shortcuts = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}
    if token in shortcuts:
        return shortcuts[token]

    try:
        parts = [float(v) for v in token.replace(",", " ").split() if v]
        if len(parts) == 3:
            vec = np.array(parts, dtype=float)
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                return vec / norm
    except ValueError:
        pass

    return np.array([0.0, 0.0, 1.0])


def parse_matrix(raw_text: str) -> List[JointRow]:
    rows: List[JointRow] = []
    for idx, line in enumerate(raw_text.strip().splitlines(), start=1):
        if not line.strip():
            continue

        cells = line.replace(",", " ").split()
        if len(cells) < 4:
            st.warning(f"Line {idx} is missing entries. Expected 4 columns; got {len(cells)}")
            continue

        try:
            joint_number = int(float(cells[0]))
            joint_type_token = cells[1].lower()
            joint_type = "revolute" if joint_type_token.startswith("r") or joint_type_token == "0" else "prismatic"
            axis = parse_axis(cells[2])
            leg_length = float(cells[3])
            rows.append(JointRow(joint_number, joint_type, axis, leg_length))
        except ValueError:
            st.warning(f"Line {idx} could not be parsed: '{line}'")
    return rows


def rows_to_dataframe(rows: List[JointRow]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "joint_number": row.joint_number,
                "joint_type": row.joint_type,
                "axis_x": row.axis[0],
                "axis_y": row.axis[1],
                "axis_z": row.axis[2],
                "leg_length": row.leg_length,
            }
            for row in rows
        ]
    )


def accumulate_positions(rows: List[JointRow]) -> np.ndarray:
    positions = [np.zeros(3)]
    for row in rows:
        step = row.axis * row.leg_length
        next_pos = positions[-1] + step
        positions.append(next_pos)
    return np.vstack(positions)


def render_chain(points: np.ndarray, name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="lines+markers",
            marker=dict(size=5, color="#1f77b4"),
            line=dict(width=6, color="#1f77b4"),
            name=name,
        )
    )
    fig.update_layout(
        title=name,
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=420,
    )
    return fig


def load_setup(uploaded) -> Optional[pd.DataFrame]:
    if not uploaded:
        return None
    try:
        raw = uploaded.read()
        try:
            parsed = json.loads(raw)
            return pd.DataFrame(parsed)
        except json.JSONDecodeError:
            uploaded.seek(0)
            return pd.read_csv(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load setup: {exc}")
        return None


st.title("Live Controller ↔ Field Robot Bridge")
st.caption(
    "Connect a UART/USB microcontroller feed, visualize the live n-DoF controller, "
    "and map its targets into a differently scaled field robot."
)

controller_col, robot_col = st.columns(2)

with controller_col:
    st.subheader("Manual Controller IDE")
    protocol = st.selectbox("Transport", ["UART/USB", "I²C", "SPI", "CAN"])
    port = st.text_input("Port or device path", value="/dev/ttyUSB0")
    baud = st.number_input("Baud rate", value=115200, step=1200)
    refresh_rate = st.slider("Expected update rate (Hz)", 1, 120, 30)

    if "controller_connected" not in st.session_state:
        st.session_state.controller_connected = False

    connect_label = "Disconnect" if st.session_state.controller_connected else "Connect"
    if st.button(connect_label, use_container_width=True):
        st.session_state.controller_connected = not st.session_state.controller_connected
        state = "connected" if st.session_state.controller_connected else "disconnected"
        st.toast(f"Controller {state} over {protocol} @ {baud} baud")

    st.markdown(
        "Paste the **n×4 joint table** streamed by the microcontroller. Columns: joint number, "
        "joint type (R/P), rotation/translation axis, and leg length / linearized travel."
    )
    sample = """1 R 0 45\n2 R 1 80\n3 P z 30\n4 R 0,1,0 50"""
    raw_matrix = st.text_area("Live joint matrix", value=sample, height=180)
    controller_rows = parse_matrix(raw_matrix) if raw_matrix.strip() else []

    upload_controller = st.file_uploader("Upload controller offsets & link lengths", type=["json", "csv"], key="controller_upload")
    controller_setup = load_setup(upload_controller)
    if controller_setup is not None:
        st.success("Controller setup loaded; values merged into the live table where columns match.")

    controller_df = rows_to_dataframe(controller_rows)
    if controller_setup is not None:
        controller_df = controller_df.combine_first(controller_setup)

    st.dataframe(controller_df, use_container_width=True)

    controller_positions = accumulate_positions(controller_rows) if controller_rows else np.zeros((1, 3))
    st.plotly_chart(render_chain(controller_positions, "Manual controller model"), use_container_width=True)

with robot_col:
    st.subheader("Field Robot IDE")
    robot_workspace = st.number_input("Field robot workspace span (mm)", value=1000.0, step=50.0)
    controller_workspace = st.number_input("Controller workspace span (mm)", value=100.0, step=10.0)
    scale_factor = robot_workspace / max(controller_workspace, 1e-6)
    st.info(f"Target coordinates are scaled by {scale_factor:.2f}× from controller space to field space.")

    target_col1, target_col2, target_col3 = st.columns(3)
    with target_col1:
        target_x = st.number_input("Controller target X", value=-75.0)
    with target_col2:
        target_y = st.number_input("Controller target Y", value=25.0)
    with target_col3:
        target_z = st.number_input("Controller target Z", value=5.0)

    scaled_target = np.array([target_x, target_y, target_z]) * scale_factor
    st.metric("Field robot target (scaled)", f"[{scaled_target[0]:.1f}, {scaled_target[1]:.1f}, {scaled_target[2]:.1f}] mm")

    upload_robot = st.file_uploader("Upload field robot offsets & link lengths", type=["json", "csv"], key="robot_upload")
    robot_setup = load_setup(upload_robot)

    robot_source = robot_setup if robot_setup is not None else controller_df
    st.dataframe(robot_source, use_container_width=True)

    robot_rows: List[JointRow] = []
    if not robot_source.empty:
        for _, row in robot_source.iterrows():
            axis = np.array([row.get("axis_x", 0.0), row.get("axis_y", 0.0), row.get("axis_z", 1.0)])
            robot_rows.append(
                JointRow(
                    int(row.get("joint_number", len(robot_rows))),
                    str(row.get("joint_type", "revolute")),
                    axis if np.linalg.norm(axis) > 1e-9 else np.array([0.0, 0.0, 1.0]),
                    float(row.get("leg_length", 0.0)),
                )
            )

    robot_positions = accumulate_positions(robot_rows) if robot_rows else np.zeros((1, 3))
    robot_plot = render_chain(robot_positions, "Field robot model")
    robot_plot.add_trace(
        go.Scatter3d(
            x=[scaled_target[0]],
            y=[scaled_target[1]],
            z=[scaled_target[2]],
            mode="markers",
            marker=dict(size=6, color="#d62728"),
            name="Scaled target",
        )
    )
    st.plotly_chart(robot_plot, use_container_width=True)

st.subheader("Forwarding live kinematics")
st.write(
    "Once connected, pipe the controller matrix to a secondary process (e.g., a motion server) "
    "that consumes the scaled target and the joint table. The field IDE reuses uploaded offsets "
    "and link lengths when translating to the physical robot's coordinate frame."
)
