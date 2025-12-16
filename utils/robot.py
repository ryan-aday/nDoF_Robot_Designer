from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Literal, Sequence, Tuple

import numpy as np

JointType = Literal["revolute", "prismatic"]


def rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
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
    width: float
    depth: float
    mass: float

    def inertia_tensor(self) -> Tuple[float, float, float]:
        ix = (1 / 12) * self.mass * (self.depth**2 + self.width**2)
        iy = (1 / 12) * self.mass * (self.length**2 + self.depth**2)
        iz = (1 / 12) * self.mass * (self.length**2 + self.width**2)
        return ix, iy, iz


@dataclass
class Joint:
    joint_type: JointType
    axis: np.ndarray
    max_torque: float = 0.0
    max_force: float = 0.0


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
    link_widths: Sequence[float],
    link_depths: Sequence[float],
    link_masses: Sequence[float],
) -> RobotModel:
    redundant = max(0, dof - 6)
    solved_dof = max(1, min(dof, len(link_lengths)))
    revolute_count = solved_dof - prismatic_count if allow_prismatic else solved_dof
    prismatic_count = prismatic_count if allow_prismatic else 0

    joints: List[Joint] = []
    links: List[Link] = []

    for i in range(solved_dof):
        axis = AXES[i % len(AXES)]
        j_type: JointType
        if allow_prismatic and i < prismatic_count:
            j_type = "prismatic"
        else:
            j_type = "revolute"
        joints.append(Joint(joint_type=j_type, axis=axis))
        links.append(
            Link(
                length=link_lengths[i],
                width=link_widths[i],
                depth=link_depths[i],
                mass=link_masses[i],
            )
        )

    notes = []
    if redundant:
        notes.append(f"Robot has {redundant} redundant DoF beyond 6.")
    if not allow_prismatic:
        notes.append("Joint solution restricted to revolute joints only.")

    robot = RobotModel(dof=solved_dof, links=links, joints=joints, redundant_dof=redundant, notes=notes)
    robot.torque_budget()
    return robot


def damped_least_squares_ik(
    robot: RobotModel,
    target: np.ndarray,
    initial_states: Sequence[float],
    max_iters: int = 200,
    damping: float = 0.01,
    tol: float = 1e-3,
) -> Tuple[np.ndarray, bool]:
    states = np.array(initial_states, dtype=float)
    for _ in range(max_iters):
        transforms = robot.homogenous_transforms(states)
        current = transforms[-1][:3, 3]
        error = target - current
        if np.linalg.norm(error) < tol:
            return states, True

        # Build Jacobian using screw axes
        J = []
        for i, joint in enumerate(robot.joints):
            origin = transforms[i][:3, 3]
            z_axis = transforms[i][:3, :3] @ joint.axis
            if joint.joint_type == "revolute":
                v = np.cross(z_axis, target - origin)
                J.append(np.concatenate([z_axis, v]))
            else:
                J.append(np.concatenate([np.zeros(3), z_axis]))
        J = np.array(J).T  # 6 x n

        JT = J[:3, :].T  # position rows only
        damping_matrix = damping * np.eye(JT.shape[1])
        delta = JT.T @ np.linalg.inv(JT @ JT.T + damping_matrix) @ error
        states += delta
    return states, False


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
                "Mass (kg)": link.mass,
                "Inertia (Ix, Iy, Iz)": (ix, iy, iz),
                "Torque/Force budget": joint.max_torque if joint.joint_type == "revolute" else joint.max_force,
            }
        )
    return summary


def spherical_adjust(target: np.ndarray, delta_radius: float) -> np.ndarray:
    r = np.linalg.norm(target)
    if r == 0:
        return target
    new_r = max(0.01, r + delta_radius)
    return target * (new_r / r)
