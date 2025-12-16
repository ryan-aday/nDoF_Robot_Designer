from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Literal, Sequence, Tuple

import numpy as np

JointType = Literal["revolute", "prismatic"]


def rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = axis / norm
    ux, uy, uz = axis
    c, s = math.cos(theta), math.sin(theta)
    return np.array(
        [
            [c + ux * ux * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s],
            [uy * ux * (1 - c) + uz * s, c + uy * uy * (1 - c), uy * uz * (1 - c) - ux * s],
            [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz * uz * (1 - c)],
        ]
    )


@dataclass
class Link:
    length: float
    cross_section_area: float
    mass: float

    def inertia_tensor(self) -> Tuple[float, float, float]:
        """Estimate inertia assuming a square cross section from provided area."""

        side = math.sqrt(max(self.cross_section_area, 1e-9))
        ix = (1 / 6) * self.mass * (side**2)
        iy = (1 / 12) * self.mass * (self.length**2 + side**2)
        iz = (1 / 12) * self.mass * (self.length**2 + side**2)
        return ix, iy, iz


@dataclass
class Joint:
    joint_type: JointType
    axis: np.ndarray
    note: str = ""
    max_torque: float = 0.0
    max_force: float = 0.0
    min_state: float = -math.pi
    max_state: float = math.pi


@dataclass
class RobotModel:
    dof: int
    links: List[Link]
    joints: List[Joint]
    gravity: float = 9.81
    redundant_dof: int = 0
    notes: List[str] = field(default_factory=list)

    def homogenous_transforms(self, states: Sequence[float]) -> List[np.ndarray]:
        transforms: List[np.ndarray] = [np.eye(4)]
        for joint, link, state in zip(self.joints, self.links, states):
            t = np.eye(4)
            if joint.joint_type == "revolute":
                rot = rotation_matrix(joint.axis, state)
                t[:3, :3] = rot
                t[:3, 3] = rot @ np.array([link.length, 0, 0])
            else:
                t[:3, :3] = np.eye(3)
                t[:3, 3] = state * joint.axis + np.array([link.length, 0, 0])
            transforms.append(transforms[-1] @ t)
        return transforms

    def forward_kinematics(self, states: Sequence[float]) -> np.ndarray:
        return self.homogenous_transforms(states)[-1]

    def joint_positions(self, states: Sequence[float]) -> np.ndarray:
        transforms = self.homogenous_transforms(states)
        return np.array([t[:3, 3] for t in transforms])

    def torque_budget(self, payload_mass: float = 0.0) -> List[float]:
        torques = []
        cumulative_mass = payload_mass
        cumulative_length = 0.0
        for joint, link in reversed(list(zip(self.joints, self.links))):
            cumulative_mass += link.mass
            cumulative_length += link.length
            torque = cumulative_mass * self.gravity * (cumulative_length)
            torques.append(torque)
            joint.max_torque = torque
        torques.reverse()
        return torques


AXES = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]


def build_robot(
    dof: int,
    allow_prismatic: bool,
    prismatic_count: int,
    link_lengths: Sequence[float],
    link_areas: Sequence[float],
    link_masses: Sequence[float],
    joint_types: Sequence[JointType] | None = None,
    joint_notes: Sequence[str] | None = None,
    joint_mins: Sequence[float] | None = None,
    joint_maxes: Sequence[float] | None = None,
    joint_axes: Sequence[Sequence[float]] | None = None,
    gravity: float = 9.81,
) -> RobotModel:
    """Construct a RobotModel from user-edited link and joint tables.

    The minimum DoF is inferred from the provided link sizing and joint types to
    avoid mismatches that can destabilize downstream Jacobian operations.
    """

    redundant = max(0, dof - 6)
    solved_dof = min(
        max(1, dof),
        len(link_lengths),
        len(link_areas),
        len(link_masses),
        len(joint_types) if joint_types is not None else dof,
    )

    # Default to an efficient mix if the user did not supply explicit joint types
    # (prioritise revolute joints unless prismatic_count is requested).
    if joint_types is None:
        revolute_count = solved_dof - prismatic_count if allow_prismatic else solved_dof
        prismatic_count = prismatic_count if allow_prismatic else 0
        joint_types = ["prismatic" if allow_prismatic and i < prismatic_count else "revolute" for i in range(solved_dof)]

    if joint_notes is None:
        joint_notes = [""] * solved_dof
    if joint_mins is None:
        joint_mins = [None] * solved_dof
    if joint_maxes is None:
        joint_maxes = [None] * solved_dof

    joints: List[Joint] = []
    links: List[Link] = []

    for i in range(solved_dof):
        if joint_axes is not None and i < len(joint_axes):
            axis = np.array(joint_axes[i], dtype=float)
        else:
            axis = AXES[i % len(AXES)]
        if np.linalg.norm(axis) < 1e-8:
            axis = AXES[i % len(AXES)]
        default_min = -math.pi if joint_types[i] == "revolute" else -link_lengths[i]
        default_max = math.pi if joint_types[i] == "revolute" else link_lengths[i]
        joints.append(
            Joint(
                joint_type=joint_types[i],
                axis=axis,
                note=joint_notes[i],
                min_state=joint_mins[i] if joint_mins[i] is not None else default_min,
                max_state=joint_maxes[i] if joint_maxes[i] is not None else default_max,
            )
        )
        links.append(
            Link(
                length=link_lengths[i],
                cross_section_area=link_areas[i],
                mass=link_masses[i],
            )
        )

    notes = []
    if redundant:
        notes.append(f"Robot has {redundant} redundant DoF beyond 6.")
    if not allow_prismatic:
        notes.append("Joint solution restricted to revolute joints only.")
    user_notes = [note for note in joint_notes if note]
    notes.extend(user_notes)

    robot = RobotModel(
        dof=solved_dof,
        links=links,
        joints=joints,
        gravity=gravity,
        redundant_dof=redundant,
        notes=notes,
    )
    robot.torque_budget()
    return robot


def _jacobian(robot: RobotModel, transforms: List[np.ndarray]) -> np.ndarray:
    end_effector = transforms[-1][:3, 3]
    J_cols = []
    for i, joint in enumerate(robot.joints):
        origin = transforms[i][:3, 3]
        z_axis = transforms[i][:3, :3] @ joint.axis
        if joint.joint_type == "revolute":
            v = np.cross(z_axis, end_effector - origin)
            J_cols.append(np.concatenate([z_axis, v]))
        else:
            J_cols.append(np.concatenate([np.zeros(3), z_axis]))
    return np.array(J_cols).T  # 6 x n


def _accept_step(
    robot: RobotModel,
    states: np.ndarray,
    delta: np.ndarray,
    target: np.ndarray,
    transforms: List[np.ndarray],
    current_error_norm: float,
    scales: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001),
) -> Tuple[np.ndarray, float, List[np.ndarray], bool, bool]:
    """Try scaled updates and only accept steps that reduce the residual.

    Returns the updated state, the new residual, the updated transforms, a flag
    indicating whether the error decreased, and a flag indicating whether any
    step was accepted. The extra flag lets callers stop if every scale fails to
    improve the error, avoiding aimless motion when a configuration is blocked
    by limits or singular structure.
    """

    mins = np.array([j.min_state for j in robot.joints])
    maxs = np.array([j.max_state for j in robot.joints])

    for scale in scales:
        candidate = np.clip(states + scale * delta, mins, maxs)
        cand_transforms = robot.homogenous_transforms(candidate)
        cand_error_norm = np.linalg.norm(target - cand_transforms[-1][:3, 3])

        if cand_error_norm < current_error_norm - 1e-9:
            return candidate, cand_error_norm, cand_transforms, True, True

    return states, current_error_norm, transforms, False, False


def reachable_distance(robot: RobotModel) -> float:
    """Upper bound on straight-line reach based on link lengths and limits."""

    reach = 0.0
    for joint, link in zip(robot.joints, robot.links):
        reach += link.length
        if joint.joint_type == "prismatic":
            reach += max(abs(joint.min_state), abs(joint.max_state))
    return reach


def damped_least_squares_ik(
    robot: RobotModel,
    target: np.ndarray,
    initial_states: Sequence[float],
    max_iters: int = 200,
    damping: float = 0.01,
    tol: float = 1e-3,
) -> Tuple[np.ndarray, bool, List[np.ndarray], str]:
    states = np.array(initial_states, dtype=float)
    trajectory: List[np.ndarray] = [states.copy()]
    transforms = robot.homogenous_transforms(states)
    current = transforms[-1][:3, 3]
    error = target - current
    current_error_norm = np.linalg.norm(error)
    if current_error_norm < tol:
        return states, True, trajectory, ""

    reason = ""
    for _ in range(max_iters):
        J = _jacobian(robot, transforms)
        J_pos = J[:3, :]
        damping_matrix = (damping**2) * np.eye(J_pos.shape[0])
        delta = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + damping_matrix) @ error
        states, current_error_norm, transforms, improved, accepted = _accept_step(
            robot, states, delta, target, transforms, current_error_norm
        )
        error = target - transforms[-1][:3, 3]
        trajectory.append(states.copy())
        if current_error_norm < tol:
            return states, True, trajectory, ""
        if not accepted:
            reason = "All candidate steps violated limits or failed to reduce the residual (possible singularity or range block)."
            break
    if not reason:
        reason = "Max iterations reached before residual tolerance was met."
    return states, False, trajectory, reason


def newton_raphson_ik(
    robot: RobotModel,
    target: np.ndarray,
    initial_states: Sequence[float],
    max_iters: int = 200,
    tol: float = 1e-3,
) -> Tuple[np.ndarray, bool, List[np.ndarray], str]:
    states = np.array(initial_states, dtype=float)
    trajectory: List[np.ndarray] = [states.copy()]
    transforms = robot.homogenous_transforms(states)
    current = transforms[-1][:3, 3]
    error = target - current
    current_error_norm = np.linalg.norm(error)
    if current_error_norm < tol:
        return states, True, trajectory, ""

    reason = ""
    for _ in range(max_iters):
        J = _jacobian(robot, transforms)
        J_pos = J[:3, :]
        delta = np.linalg.pinv(J_pos) @ error
        states, current_error_norm, transforms, improved, accepted = _accept_step(
            robot, states, delta, target, transforms, current_error_norm
        )
        error = target - transforms[-1][:3, 3]
        trajectory.append(states.copy())
        if current_error_norm < tol:
            return states, True, trajectory, ""
        if not accepted:
            reason = "All candidate steps violated limits or failed to reduce the residual (possible singularity or range block)."
            break
    if not reason:
        reason = "Max iterations reached before residual tolerance was met."
    return states, False, trajectory, reason


def gradient_descent_ik(
    robot: RobotModel,
    target: np.ndarray,
    initial_states: Sequence[float],
    step_size: float = 0.05,
    max_iters: int = 400,
    tol: float = 1e-3,
) -> Tuple[np.ndarray, bool, List[np.ndarray], str]:
    states = np.array(initial_states, dtype=float)
    trajectory: List[np.ndarray] = [states.copy()]
    transforms = robot.homogenous_transforms(states)
    current = transforms[-1][:3, 3]
    error = target - current
    current_error_norm = np.linalg.norm(error)
    if current_error_norm < tol:
        return states, True, trajectory, ""

    reason = ""
    for _ in range(max_iters):
        J = _jacobian(robot, transforms)
        J_pos = J[:3, :]
        delta = step_size * (J_pos.T @ error)
        states, current_error_norm, transforms, improved, accepted = _accept_step(
            robot, states, delta, target, transforms, current_error_norm
        )
        error = target - transforms[-1][:3, 3]
        trajectory.append(states.copy())
        if current_error_norm < tol:
            return states, True, trajectory, ""
        if not accepted:
            reason = "All candidate steps violated limits or failed to reduce the residual (possible singularity or range block)."
            break
    if not reason:
        reason = "Max iterations reached before residual tolerance was met."
    return states, False, trajectory, reason


def screw_enhanced_ik(
    robot: RobotModel,
    target: np.ndarray,
    initial_states: Sequence[float],
    step_size: float = 0.07,
    damping_base: float = 0.01,
    momentum: float = 0.1,
    max_iters: int = 400,
    tol: float = 1e-3,
) -> Tuple[np.ndarray, bool, List[np.ndarray], str]:
    """IK variant inspired by screw-theory refinements with adaptive damping.

    The method normalizes Jacobian columns (screw axes), adapts damping based on
    a manipulability estimate, and blends in a small momentum term to avoid
    stalls near shallow gradients (see e.g., He et al. 2015 on screw-theoretic
    improvements and related trajectory smoothing papers).
    """

    states = np.array(initial_states, dtype=float)
    prev_delta = np.zeros_like(states)
    trajectory: List[np.ndarray] = [states.copy()]

    transforms = robot.homogenous_transforms(states)
    current = transforms[-1][:3, 3]
    error = target - current
    current_error_norm = np.linalg.norm(error)
    if current_error_norm < tol:
        return states, True, trajectory, ""

    reason = ""
    for _ in range(max_iters):
        J = _jacobian(robot, transforms)
        J_pos = J[:3, :]

        col_norms = np.linalg.norm(J_pos, axis=0, keepdims=True) + 1e-8
        weighted_J = J_pos / col_norms

        singular_values = np.linalg.svd(weighted_J, compute_uv=False)
        valid_s = singular_values[singular_values > 1e-6]
        if len(valid_s) == 0:
            manipulability = 0.0
        else:
            manipulability = float(np.prod(valid_s) ** (1 / len(valid_s)))

        adaptive_damping = damping_base / (manipulability + 1e-6)
        lhs = weighted_J @ weighted_J.T + adaptive_damping * np.eye(3)
        delta = step_size * (weighted_J.T @ np.linalg.solve(lhs, error))
        delta += momentum * prev_delta

        states, current_error_norm, transforms, improved, accepted = _accept_step(
            robot, states, delta, target, transforms, current_error_norm
        )
        error = target - transforms[-1][:3, 3]
        prev_delta = delta
        trajectory.append(states.copy())
        if current_error_norm < tol:
            return states, True, trajectory, ""
        if not accepted:
            reason = "All candidate steps violated limits or failed to reduce the residual (possible singularity or range block)."
            break

    if not reason:
        reason = "Max iterations reached before residual tolerance was met."
    return states, False, trajectory, reason


def joint_summary(robot: RobotModel) -> List[dict]:
    summary = []
    for idx, (joint, link) in enumerate(zip(robot.joints, robot.links), start=1):
        ix, iy, iz = link.inertia_tensor()
        summary.append(
            {
                "Joint": idx,
                "Type": joint.joint_type,
                "Axis": joint.axis.tolist(),
                "Length (m)": link.length,
                "Cross-sectional area (m²)": link.cross_section_area,
                "Mass (kg)": link.mass,
                "Inertia (Ix, Iy, Iz)": (ix, iy, iz),
                "Range (min, max)": (joint.min_state, joint.max_state),
                "Torque/Force budget": joint.max_torque if joint.joint_type == "revolute" else joint.max_force,
                "Note": joint.note,
            }
        )
    return summary


def spherical_adjust(target: np.ndarray, delta_radius: float) -> np.ndarray:
    r = np.linalg.norm(target)
    if r == 0:
        return target
    new_r = max(0.01, r + delta_radius)
    return target * (new_r / r)
