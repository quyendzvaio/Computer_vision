# Phase 2: pretrained inference, tracking and ROI evidence

## Verified model contracts

- RTMPose-s body7 is the official OpenMMLab dynamic-batch ONNX artifact. The
  archive and extracted graph are SHA-256 pinned in `models/manifest.yaml`.
  Input is normalized RGB `NCHW [N,3,256,192]`; outputs are COCO-17 SimCC
  distributions `[N,17,384]` and `[N,17,512]`.
- RTMDet-nano uses the official person checkpoint. OpenMMLab does not publish
  the matching person ONNX in that model table, so the repository intentionally
  requires the official MMDeploy export. The wrapper verifies the resulting
  `dets/labels` contract before installation.
- No PPE weights are enabled. Until an explicit-negative pretrained model is
  pinned and verified, PPE is `UNVERIFIABLE`; missing positive detections never
  become violations.

## Coordinate contracts

Detection runs on the whole frame. RTMDet resize is keep-ratio with bottom/right
padding and its boxes are clipped and mapped back to full-frame coordinates.
RTMPose receives a 1.25-padded person box, uses the top-down affine transform,
decodes SimCC with split ratio 2, then applies the inverse affine transform.
Invisible or out-of-frame keypoints have `x=None, y=None`.

Every camera owns a `ByteTracker`. IDs are scoped as `(camera_id, track_id)` and
low-confidence detections participate only in the second ByteTrack association
stage, not in creating new tracks.

ROI polygons stay normalized and are converted at the analytics frame size.
Rules select only their evidence geometry: head/bare-head, torso/no-vest,
feet/no-shoes, or upper-body/person for falls. A positive-area intersection is
required; whole-person containment is never required.

## Deployment split

Ubuntu server services remain in `deployment/server/compose.yaml`. Windows edge
is native and packaged as a WinSW-managed service in `deployment/edge-windows/`.
The Ubuntu server has no AI model or GPU dependency.
