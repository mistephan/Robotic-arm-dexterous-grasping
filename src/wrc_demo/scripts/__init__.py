"""wrc_demo.scripts -- CLI entry points runnable via ``python -m``.

Schema version: 1
Frame convention: wrc_demo (base +x away from robot, +y left, gripper_end TCP)

Currently hosts ``calib_top_orbbec`` -- the operator-facing top-camera
hand-eye calibration tool. Architecture ported from
``rebot_grasp/scripts/calibrate_top_camera.py`` (state machine +
gravity-comp manual mode). Heavy imports (pyorbbecsdk, the arm SDK) live
inside the individual script so the rest of the package stays
hardware-free.
"""

from __future__ import annotations
