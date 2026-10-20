"""Cartesian trajectory planning — wraps reBotArm_control_py.trajectory.

The planner is optional: when reBotArm_control_py is not importable,
SkillRuntime falls back to joint-space PTP (the existing behaviour).
"""

try:
    from .cartesian_planner import CartesianPlanner, _AVAILABLE
except ImportError:
    CartesianPlanner = None  # type: ignore[assignment,misc]
    _AVAILABLE = False

__all__ = ["CartesianPlanner", "_AVAILABLE"]
