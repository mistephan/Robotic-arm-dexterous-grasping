# WRC intelligence and LinkerHand O6 integration

## What WRC already has

WRC's useful original architecture is retained: deterministic reflexes,
experience-plan replay, LLM decomposition, a warm RGB-D belief store,
GraspGenX plus analytic fallbacks, grasp-outcome memory, VLM failure advice,
ASPIRE traces, and a markdown skill library. The skill library is now loaded
into the LLM context for related English or Chinese tasks.

## Language-to-skill architecture

Commands should compile through a typed semantic frame rather than letting an
LLM invent motor operations:

1. Normalize ASR punctuation and bilingual object aliases.
2. Extract action, theme object, destination object, spatial relation,
   material, and requested grasp mode.
3. Compile routine frames deterministically to one skill call.
4. Use the LLM only for genuinely novel composition or ambiguous reference.
5. Validate every tool schema and capability before execution.
6. Keep metric geometry, IK, collision, force and tactile limits outside the
   LLM.

Implemented deterministic examples include Chinese/English pick-and-place,
base-frame left/right relations, and O6 `open`, `power_grasp`, and `pinch`.

## O6 control boundary

The arm IK controls exactly six B601 joints. O6 finger joints remain outside
that vector and are commanded by the LinkerHand SDK. One `LinkerHandO6`
instance owns both commands and matrix-touch reads so two processes never
transmit on the same CAN interface concurrently.

Tactile close advances finger flexion in small steps. A finger freezes at
first contact. Hard pressure, malformed sensor data, target-without-
contact, or timeout aborts the close. Mango and lime should begin with a soft
five-finger power grasp; thumb-index pinch is intended for smaller rigid
objects, not as the default fruit grasp.

## URDF result

`tools/build_rebot_o6_urdf.py` rebuilds the model from the current SDK RS
URDF, removes the stock two-finger subtree, imports the valid adapter/O6
mechanical subtree, and inserts `o6_palm_center`, `o6_grasp_tcp`, and the
backward-compatible `gripper_end` inside the `<robot>` element. The prior
`rebotB601_linkerhand_o6_with_motor_TCP.urdf` is invalid XML because those
frames occur after `</robot>`.

The URDF/STL-derived hand-base coordinates for `o6_grasp_tcp` are
`[0.0600, 0.0275, 0.1187]` m at the fixed thumb-opposition preset `raw=87`.
The custom grasp frame uses the measured thumb-to-index/middle direction as X
for closing. Z is the closest direction to the old centre ray that is exactly
orthogonal to X, and WRC publishes `[custom Z, custom X, custom Y]` so its
normal `rotation[:,0] == approach` contract is retained. TCP position remains
the physical closure centre but is not treated as an approach direction.

## Required physical commissioning

1. Verify the adapter rotation and measure `link6 -> hand_base_link`.
2. Verify the computed `hand_base_link -> o6_grasp_tcp` in TF/RViz with a
   rigid pointer or touch plate; only update the generator if the measured
   closure centre exceeds the approximately 1 mm mesh-estimation tolerance.
3. With the arm stationary and clear, calibrate right-hand open, soft power,
   and pinch targets at low speed/torque.
4. Record unloaded tactile baselines and noise per taxel; runtime thresholds already
   use baseline-subtracted values. Start with the read-only command below.

   ```bash
   python tools/calibrate_linkerhand_o6.py --mode inspect --samples 30
   ```

   `inspect` and `baseline` never command a finger or instantiate the arm. Any
   finger action requires `--allow-motion`; `pinch` and `power_grasp` also
   require `--object-present`.
5. Verify the existing O6 open/mid swept volumes in the new custom frame.
   Their dimensions are retained from the O6 wizard result; do not reuse B601
   jaw sweep values and do not rerun the wizard unless physical validation
   shows the retained volume no longer bounds the hand.
6. Run staged tests: no arm motion, foam object, lime, mango, then transport.
7. Only after validation set `tcp_calibrated: true` and
   `poses_calibrated: true`.

## Next intelligence increments

- Typed semantic frames for reference resolution, relations, quantities and
  multi-object tasks.
- Scene graph with observed/remembered/in-hand/placed state and uncertainty.
- Preflight the entire grasp, lift, place and return-home trajectory before
  first motion.
- Post-grasp verification from tactile contact distribution plus vision.
- Post-place geometric verification rather than repeating the action.
- Distill validated failure repairs into the skill library automatically,
  but require human approval before a learned entry affects real motion.
