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
    body_length: float = 0.05
    body_mass: float = 0.2
    body_inertia: Tuple[float, float, float] = (0.0, 0.0, 0.0)


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
            cumulative_mass += link.mass + joint.body_mass
            cumulative_length += link.length + max(joint.body_length, 0.0)

            torque_from_weight = cumulative_mass * self.gravity * cumulative_length

            axis = joint.axis / (np.linalg.norm(joint.axis) + 1e-9)
            inertia_tensor = np.diag(joint.body_inertia)
            projected_inertia = float(axis @ inertia_tensor @ axis)
            inertia_torque = projected_inertia  # assuming 1 rad/s² nominal acceleration

            torque = torque_from_weight + inertia_torque
            torques.append(torque)
            if joint.joint_type == "revolute":
                joint.max_torque = torque
            else:
                joint.max_force = torque_from_weight / max(link.length, 1e-6)
        torques.reverse()
        return torques


AXES = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]


def _numerical_jacobian(robot: RobotModel, states: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Finite-difference positional Jacobian via forward kinematics.

    This derivative-free Jacobian avoids the cross-product analytic form and relies only
    on homogeneous transforms (akin to DH-based forward kinematics, e.g., Addison 2021),
    helping the solver behave near singular configurations or axis edge cases.
    """

    base_tf = robot.homogenous_transforms(states)[-1]
    base_pos = base_tf[:3, 3]
    J_cols = []
    for idx, (joint, state) in enumerate(zip(robot.joints, states)):
        step = eps if joint.joint_type == "revolute" else max(eps, 0.25 * eps + 1e-6)
        bumped = states.copy()
        bumped[idx] = np.clip(state + step, joint.min_state, joint.max_state)
        bumped_tf = robot.homogenous_transforms(bumped)[-1]
        bumped_pos = bumped_tf[:3, 3]
        J_cols.append((bumped_pos - base_pos) / step)
    return np.array(J_cols).T  # 3 x n


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
    joint_body_lengths: Sequence[float] | None = None,
    joint_body_masses: Sequence[float] | None = None,
    joint_body_inertias: Sequence[Sequence[float]] | None = None,
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
    if joint_body_lengths is None:
        joint_body_lengths = [0.05] * solved_dof
    if joint_body_masses is None:
        joint_body_masses = [0.2] * solved_dof
    if joint_body_inertias is None:
        joint_body_inertias = [(0.0, 0.0, 0.0)] * solved_dof

    def _safe_scalar(value: float | int | None, fallback: float) -> float:
        if value is None:
            return fallback
        try:
            val = float(value)
        except (TypeError, ValueError):
            return fallback
        if not math.isfinite(val):
            return fallback
        return val

    joints: List[Joint] = []
    links: List[Link] = []

    for i in range(solved_dof):
        if joint_axes is not None and i < len(joint_axes):
            axis = np.array(joint_axes[i], dtype=float)
        else:
            axis = AXES[i % len(AXES)]
        if np.linalg.norm(axis) < 1e-8:
            axis = AXES[i % len(AXES)]
        axis = axis / np.linalg.norm(axis)
        default_min = -math.pi if joint_types[i] == "revolute" else -link_lengths[i]
        default_max = math.pi if joint_types[i] == "revolute" else link_lengths[i]
        body_length = abs(_safe_scalar(joint_body_lengths[i], 0.05))
        body_mass = _safe_scalar(joint_body_masses[i], 0.2)
        inertia_tuple = tuple(
            _safe_scalar(component, 0.0) for component in joint_body_inertias[i]
        )

        joints.append(
            Joint(
                joint_type=joint_types[i],
                axis=axis,
                note=joint_notes[i],
                min_state=joint_mins[i] if joint_mins[i] is not None else default_min,
                max_state=joint_maxes[i] if joint_maxes[i] is not None else default_max,
                body_length=body_length,
                body_mass=body_mass,
                body_inertia=inertia_tuple,
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


def _monotonic_step(
    robot: RobotModel,
    states: np.ndarray,
    delta: np.ndarray,
    target: np.ndarray,
    transforms: List[np.ndarray],
    current_error_norm: float,
    scales: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01),
) -> Tuple[np.ndarray, float, List[np.ndarray], bool]:
    """Backtracking line search that only accepts error-reducing steps.

    Each solver proposes a delta; we then try progressively smaller scales until
    the positional residual shrinks. If no scale improves the error, the solver
    reports the stall so callers can explain why convergence failed instead of
    wandering aimlessly.
    """

    mins = np.array([j.min_state for j in robot.joints])
    maxs = np.array([j.max_state for j in robot.joints])

    best_states = states
    best_error = current_error_norm
    best_transforms = transforms

    for scale in scales:
        candidate = np.clip(states + scale * delta, mins, maxs)
        cand_transforms = robot.homogenous_transforms(candidate)
        cand_error_norm = np.linalg.norm(target - cand_transforms[-1][:3, 3])
        if cand_error_norm < best_error - 1e-9:
            best_states = candidate
            best_error = cand_error_norm
            best_transforms = cand_transforms
            if cand_error_norm < current_error_norm - 1e-9:
                return candidate, cand_error_norm, cand_transforms, True

    # Fall back to the best reduction even if the scale loop did not strictly improve
    improved = best_error < current_error_norm - 1e-9
    return best_states, best_error, best_transforms, improved


def damped_least_squares_ik(
    robot: RobotModel,
    target: np.ndarray,
    initial_states: Sequence[float],
    max_iters: int = 200,
    damping: float = 0.01,
    tol: float = 1e-3,
) -> Tuple[np.ndarray, bool, List[np.ndarray]]:
    states = np.array(initial_states, dtype=float)
    trajectory: List[np.ndarray] = [states.copy()]
    transforms = robot.homogenous_transforms(states)
    current = transforms[-1][:3, 3]
    error = target - current
    current_error_norm = np.linalg.norm(error)
    if current_error_norm < tol:
        return states, True, trajectory

    for _ in range(max_iters):
        J = _jacobian(robot, transforms)
        J_pos = J[:3, :]
        damping_matrix = (damping**2) * np.eye(J_pos.shape[0])
        delta = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + damping_matrix) @ error
        states, current_error_norm, transforms, improved = _monotonic_step(
            robot, states, delta, target, transforms, current_error_norm
        )
        error = target - transforms[-1][:3, 3]
        trajectory.append(states.copy())
        if current_error_norm < tol:
            return states, True, trajectory
        if not improved:
            break
    return states, False, trajectory


def matrix_projection_ik(
    robot: RobotModel,
    target: np.ndarray,
    initial_states: Sequence[float],
    max_iters: int = 400,
    tol: float = 1e-3,
) -> Tuple[np.ndarray, bool, List[np.ndarray]]:
    """Solve IK via basic matrix projection using the translational Jacobian.

    This method builds the transform from the current end-effector pose to the
    target (translation-only), then repeatedly applies a least-squares update
    using the positional Jacobian. It is intentionally simple matrix math to
    satisfy scenarios where a direct projection along a single local axis should
    reach the target without solver wandering.
    """

    states = np.array(initial_states, dtype=float)
    trajectory: List[np.ndarray] = [states.copy()]
    transforms = robot.homogenous_transforms(states)
    current = transforms[-1][:3, 3]
    error = target - current
    current_error_norm = np.linalg.norm(error)
    if current_error_norm < tol:
        return states, True, trajectory

    for _ in range(max_iters):
        J_pos = _jacobian(robot, transforms)[:3, :]
        delta, *_ = np.linalg.lstsq(J_pos, error, rcond=None)
        states, current_error_norm, transforms, improved = _monotonic_step(
            robot, states, delta, target, transforms, current_error_norm, scales=(1.0, 0.75, 0.5, 0.25, 0.1, 0.05)
        )
        error = target - transforms[-1][:3, 3]
        trajectory.append(states.copy())
        if current_error_norm < tol:
            return states, True, trajectory
        if not improved:
            break

    return states, False, trajectory


def newton_raphson_ik(
    robot: RobotModel,
    target: np.ndarray,
    initial_states: Sequence[float],
    max_iters: int = 200,
    tol: float = 1e-3,
) -> Tuple[np.ndarray, bool, List[np.ndarray]]:
    states = np.array(initial_states, dtype=float)
    trajectory: List[np.ndarray] = [states.copy()]
    transforms = robot.homogenous_transforms(states)
    current = transforms[-1][:3, 3]
    error = target - current
    current_error_norm = np.linalg.norm(error)
    if current_error_norm < tol:
        return states, True, trajectory

    for _ in range(max_iters):
        J = _jacobian(robot, transforms)
        J_pos = J[:3, :]
        delta = np.linalg.pinv(J_pos) @ error
        states, current_error_norm, transforms, improved = _monotonic_step(
            robot, states, delta, target, transforms, current_error_norm
        )
        error = target - transforms[-1][:3, 3]
        trajectory.append(states.copy())
        if current_error_norm < tol:
            return states, True, trajectory
        if not improved:
            break
    return states, False, trajectory


def gradient_descent_ik(
    robot: RobotModel,
    target: np.ndarray,
    initial_states: Sequence[float],
    step_size: float = 0.05,
    max_iters: int = 400,
    tol: float = 1e-3,
) -> Tuple[np.ndarray, bool, List[np.ndarray]]:
    states = np.array(initial_states, dtype=float)
    trajectory: List[np.ndarray] = [states.copy()]
    transforms = robot.homogenous_transforms(states)
    current = transforms[-1][:3, 3]
    error = target - current
    current_error_norm = np.linalg.norm(error)
    if current_error_norm < tol:
        return states, True, trajectory

    for _ in range(max_iters):
        J = _jacobian(robot, transforms)
        J_pos = J[:3, :]
        delta = step_size * (J_pos.T @ error)
        states, current_error_norm, transforms, improved = _monotonic_step(
            robot, states, delta, target, transforms, current_error_norm
        )
        error = target - transforms[-1][:3, 3]
        trajectory.append(states.copy())
        if current_error_norm < tol:
            return states, True, trajectory
        if not improved:
            break
    return states, False, trajectory


def screw_enhanced_ik(
    robot: RobotModel,
    target: np.ndarray,
    initial_states: Sequence[float],
    step_size: float = 0.07,
    damping_base: float = 0.01,
    momentum: float = 0.1,
    max_iters: int = 400,
    tol: float = 1e-3,
) -> Tuple[np.ndarray, bool, List[np.ndarray]]:
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
        return states, True, trajectory

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

        states, current_error_norm, transforms, improved = _monotonic_step(
            robot, states, delta, target, transforms, current_error_norm
        )
        error = target - transforms[-1][:3, 3]
        prev_delta = delta
        trajectory.append(states.copy())
        if current_error_norm < tol:
            return states, True, trajectory
        if not improved:
            break

    return states, False, trajectory


def forward_finite_ik(
    robot: RobotModel,
    target: np.ndarray,
    initial_states: Sequence[float],
    max_iters: int = 400,
    tol: float = 1e-3,
) -> Tuple[np.ndarray, bool, List[np.ndarray]]:
    """Inverse kinematics using only forward-kinematics evaluations.

    A finite-difference Jacobian (via homogeneous transforms) feeds a damped least
    squares update. Manipulability-based damping and a monotonic backtracking step help
    avoid solver stalls near singularities while keeping residuals shrinking.
    """

    states = np.array(initial_states, dtype=float)
    trajectory: List[np.ndarray] = [states.copy()]
    transforms = robot.homogenous_transforms(states)
    current = transforms[-1][:3, 3]
    error = target - current
    current_error_norm = np.linalg.norm(error)
    if current_error_norm < tol:
        return states, True, trajectory

    for _ in range(max_iters):
        J_pos = _numerical_jacobian(robot, states)
        svd_vals = np.linalg.svd(J_pos, compute_uv=False)
        manipulability = float(np.prod(svd_vals[svd_vals > 1e-6]) ** (1 / max(1, len(robot.joints))))
        damping = 0.01 / (manipulability + 1e-6)
        lhs = J_pos @ J_pos.T + (damping**2) * np.eye(3)
        delta = J_pos.T @ np.linalg.solve(lhs, error)
        states, current_error_norm, transforms, improved = _monotonic_step(
            robot, states, delta, target, transforms, current_error_norm, scales=(1.0, 0.5, 0.25, 0.1, 0.05)
        )
        error = target - transforms[-1][:3, 3]
        trajectory.append(states.copy())
        if current_error_norm < tol:
            return states, True, trajectory
        if not improved:
            break

    return states, False, trajectory


def joint_summary(robot: RobotModel) -> List[dict]:
    summary = []
    for idx, (joint, link) in enumerate(zip(robot.joints, robot.links), start=1):
        ix, iy, iz = link.inertia_tensor()
        summary.append(
            {
                "Joint": idx,
                "Type": joint.joint_type,
                "Axis": joint.axis.tolist(),
                "Joint body length (m)": joint.body_length,
                "Joint body mass (kg)": joint.body_mass,
                "Joint body inertia (Ix, Iy, Iz)": joint.body_inertia,
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


def reachability_report(
    robot: RobotModel, target: np.ndarray, states: Sequence[float] | None = None
) -> Tuple[bool, List[str]]:
    """Reachability analysis that respects local joint axes and limits.

    The check combines a generous geometric bound with a local translational
    Jacobian rank test so that axis selections (e.g., all Z-axes) that confine
    motion to a plane or line are reported explicitly. The Jacobian uses the
    current/home pose, meaning axes are interpreted in their *local* frames and
    rotated by upstream joints.
    """

    reasons: List[str] = []

    link_lengths = np.array([link.length for link in robot.links])
    prismatic_spans = np.array(
        [
            max(abs(j.min_state), abs(j.max_state)) if j.joint_type == "prismatic" else 0.0
            for j in robot.joints
        ]
    )

    max_reach = float(link_lengths.sum() + prismatic_spans.sum())
    distance = float(np.linalg.norm(target))

    if distance > max_reach + 1e-6:
        reasons.append(
            f"Target is {distance:.3f} m from the origin, exceeding the generous reach bound of {max_reach:.3f} m."
        )

    # Local-axis-aware mobility check at the current/home pose
    if states is None:
        home_states = np.zeros(len(robot.joints))
    else:
        home_states = np.array(states, dtype=float)
    transforms = robot.homogenous_transforms(home_states)
    J_pos = _jacobian(robot, transforms)[:3, :]
    current_pos = transforms[-1][:3, 3]
    displacement = target - current_pos

    col_space_proj = J_pos @ np.linalg.pinv(J_pos) @ displacement
    residual_component = displacement - col_space_proj
    residual_norm = float(np.linalg.norm(residual_component))
    rank = int(np.linalg.matrix_rank(J_pos))

    if rank < 3 and residual_norm > 1e-4:
        reasons.append(
            "Joint axis arrangement limits translation (rank "
            f"{rank}) so the remaining {residual_norm:.4f} m toward the target lies outside the local motion subspace."
        )

    feasible = len(reasons) == 0
    return feasible, reasons


def manipulability_index(robot: RobotModel, states: Sequence[float]) -> Tuple[float, float]:
    """Return (geometric mean singular value, condition number) of the numerical Jacobian."""

    J_pos = _numerical_jacobian(robot, np.array(states, dtype=float))
    sv = np.linalg.svd(J_pos, compute_uv=False)
    positive_sv = sv[sv > 1e-8]
    if len(positive_sv) == 0:
        geom_mean = 0.0
    else:
        geom_mean = float(np.prod(positive_sv) ** (1 / len(positive_sv)))
    cond = float(np.max(sv) / max(np.min(sv), 1e-9))
    return geom_mean, cond


def transform_from_target_to_start(
    robot: RobotModel, target: np.ndarray, start_states: Sequence[float]
) -> np.ndarray:
    """Return the homogeneous transform that maps the start end-effector pose to the target.

    The method computes the current end-effector transform from the provided
    start states, constructs a pure-translation target transform, and returns
    the relative matrix the chain would need to realize. It uses only basic
    matrix algebra and can be surfaced in the UI for transparency when
    diagnosing convergence.
    """

    start_tf = robot.homogenous_transforms(start_states)[-1]
    target_tf = np.eye(4)
    target_tf[:3, 3] = target
    return np.linalg.inv(start_tf) @ target_tf


def spherical_adjust(target: np.ndarray, delta_radius: float) -> np.ndarray:
    r = np.linalg.norm(target)
    if r == 0:
        return target
    new_r = max(0.01, r + delta_radius)
    return target * (new_r / r)


def sample_workspace_points(robot: RobotModel, samples: int = 800, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a point cloud of reachable end-effector locations within joint limits.

    The sampler draws random joint states within each joint's configured bounds
    and evaluates forward kinematics to build a coarse workspace blob. This
    visualization can help diagnose why a target may be unreachable even when
    singularities are permitted.
    """

    if rng is None:
        rng = np.random.default_rng()

    mins = np.array([joint.min_state for joint in robot.joints])
    maxs = np.array([joint.max_state for joint in robot.joints])

    draws = rng.uniform(mins, maxs, size=(max(1, samples), len(robot.joints)))
    points = []
    for state in draws:
        tf = robot.homogenous_transforms(state)[-1]
        points.append(tf[:3, 3])
    return np.array(points)
