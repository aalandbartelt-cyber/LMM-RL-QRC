"""Backend-agnostic foundations for a Unitree Go2 locomotion project.

This package deliberately contains no robot/network/Isaac Lab side effects.  It
defines the contracts that the simulator, policy runner and later hardware
adapter must share.
"""

from .types import (
    ACTION_DIM,
    OBSERVATION_DIM,
    DEFAULT_JOINT_ANGLES,
    JOINT2MOTOR_INDEX,
    Go2ObservationConfig,
    RobotState,
    VelocityCommand,
)
from .observation import ObservationBuilder
from .command import CommandLimiter
from .mission_bridge import command_from_mission_output
from .isaac_adapter import IsaacLabStateAdapter
from .control_cycle import ControlOutput, Go2ControlCycle
from .route_evaluator import RouteEvaluation, RouteProgressEvaluator, RouteSample, RouteWaypoint
from .performance import PerformanceDecision, PerformanceGate, PerformanceThresholds
from .mission_scheduler import MissionCommand, PatrolMissionScheduler
from .experiment_io import assess_file, decision_to_json, load_results_by_seed
from .campus_pipeline import CampusMissionPipeline
from .interface_audit import InterfaceAuditResult, PolicyInterfaceMetadata, audit_policy_interface
from .training_launcher import TrainingRunSpec, build_train_command, verify_unitree_rl_lab_checkout
from .route_mission import MissionWaypoint, RouteVelocityController, RouteVelocityOutput, load_route_file, wrap_angle
from .artifact_manifest import ArtifactVerification, PolicyArtifactManifest, sha256_file, verify_policy_manifest
from .policy import PolicyRunner, ZeroPolicy
from .safety import SafetyDecision, SafetySupervisor
from .metrics import EpisodeMetrics, MetricsAccumulator

__all__ = [
    "ACTION_DIM",
    "OBSERVATION_DIM",
    "DEFAULT_JOINT_ANGLES",
    "JOINT2MOTOR_INDEX",
    "Go2ObservationConfig",
    "RobotState",
    "VelocityCommand",
    "ObservationBuilder",
    "CommandLimiter",
    "command_from_mission_output",
    "IsaacLabStateAdapter",
    "ControlOutput",
    "Go2ControlCycle",
    "RouteEvaluation",
    "RouteProgressEvaluator",
    "RouteSample",
    "RouteWaypoint",
    "PerformanceDecision",
    "PerformanceGate",
    "PerformanceThresholds",
    "MissionCommand",
    "PatrolMissionScheduler",
    "assess_file",
    "decision_to_json",
    "load_results_by_seed",
    "CampusMissionPipeline",
    "InterfaceAuditResult",
    "PolicyInterfaceMetadata",
    "audit_policy_interface",
    "TrainingRunSpec",
    "build_train_command",
    "verify_unitree_rl_lab_checkout",
    "MissionWaypoint",
    "RouteVelocityController",
    "RouteVelocityOutput",
    "load_route_file",
    "wrap_angle",
    "ArtifactVerification",
    "PolicyArtifactManifest",
    "sha256_file",
    "verify_policy_manifest",
    "PolicyRunner",
    "ZeroPolicy",
    "SafetyDecision",
    "SafetySupervisor",
    "EpisodeMetrics",
    "MetricsAccumulator",
]
