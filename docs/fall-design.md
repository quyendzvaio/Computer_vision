# Rule-based spatiotemporal fall design

This design uses pretrained RTMPose-s and ByteTrack. It does not train a fall
classifier and never confirms a fall from one frame.

## Track sample

Each `(camera_id, track_id)` owns a timestamp-based sliding window. A sample
contains the person bbox, visible keypoints, head/shoulder/hip/upper-body
centres, torso angle and length, confidence and ROI evidence. Invisible joints
are excluded rather than represented as a valid `(0, 0)` point.

If both shoulders or hips are visible, their centre is the arithmetic mean. A
single visible hip may be used with reduced feature confidence. Without a hip,
torso length and angle are unavailable and the track cannot reach a confirmed
fall solely from head/shoulder posture.

## Spatial features

For shoulder centre `S=(sx,sy)` and hip centre `H=(hx,hy)`:

```text
torso_vector = S - H
torso_length = ||S - H||
torso_angle  = degrees(atan2(abs(sx-hx), abs(sy-hy)))
```

`torso_angle` is zero for an upright vertical torso and approaches 90 degrees
when horizontal. Bbox aspect ratio and head–hip geometry are supporting
features only.

The upper-body region is built from visible head, shoulder and hip points. Its
polygon, relevant keypoints and optionally the person bbox/mask are tested
against the normalized ROI mapped to the current full-frame resolution.

## Temporal features

Velocities use monotonic timestamps, never frame number:

```text
velocity(point) = (point[t] - point[t-1]) / delta_time
normalized_velocity = velocity / max(torso_length, epsilon)
angular_velocity = shortest_angle_delta / delta_time
```

The extractor produces normalized head, hip and upper-body vertical velocity,
horizontal velocity, torso angular velocity, optional angular acceleration,
posture persistence, motion magnitude and post-transition inactivity. Feature
confidence is the combination of joint visibility, pose confidence and whether
one or both sides of a joint pair were available.

Camera motion is treated as a quality gate: coherent background/global motion
reduces feature confidence or marks the interval `UNVERIFIABLE`; it is not fall
evidence.

## Configurable score

Each component is clipped to `[0,1]`. The initial untrained rule score is:

```text
0.25 * descent
+ 0.20 * rotation
+ 0.10 * horizontal_motion
+ 0.20 * abnormal_posture
+ 0.15 * persistence
+ 0.10 * inactivity
```

Weights and thresholds live in `edge_runtime/config/rules.yaml`, not business
logic. These values are starting points for manual video validation, not proven
operating thresholds.

## State machine

```text
NORMAL
  └─ fast descent/horizontal motion/rotation ─→ RAPID_TRANSITION
       ├─ insufficient visibility ─→ UNVERIFIABLE
       ├─ transition disappears ─→ NORMAL
       └─ candidate score + posture change ─→ POSSIBLE_FALL
            ├─ quick upright recovery ─→ NORMAL
            ├─ insufficient evidence ─→ UNVERIFIABLE
            └─ persistence + confirmation score + ROI window evidence
                 ─→ CONFIRMED_FALL
                      └─ observed stand-up motion ─→ RECOVERING
                           └─ stable upright posture ─→ NORMAL
```

The fall window starts at `RAPID_TRANSITION`. Confirmation requires ROI evidence
for at least three samples or 250 ms, with a default region overlap ratio of
0.10. This lets a head/upper body or trajectory crossing the ROI qualify even
when the full person bbox does not lie inside it.

## Initial thresholds

| Setting | Initial value |
|---|---:|
| Sliding window | 3500 ms |
| Minimum visible-keypoint ratio | 0.45 |
| Upright torso angle | ≤25° |
| Abnormal torso angle | ≥55° |
| Normalized descent velocity | 0.8 torso lengths/s |
| Normalized horizontal velocity | 0.75 torso lengths/s |
| Torso angular velocity | 60°/s |
| Candidate score | 0.55 |
| Confirmation score | 0.72 |
| Recovery score | 0.35 |
| Maximum rapid transition | 1200 ms |
| Abnormal posture persistence | 800 ms |
| Inactivity observation | 500 ms |
| Stable recovery | 1500 ms |

Phase 3 tests must cover falls, bending, sitting, kneeling, deliberate lying,
leaving the frame, partial bodies, occlusion and camera shake. Thresholds may be
changed through configuration after replay review, but no model training is in
scope.
