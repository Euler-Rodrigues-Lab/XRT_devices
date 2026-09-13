# Python extraction progress

## Completed in source

- [x] Independent package, lazy optional transport/inference dependencies.
- [x] XRT body schema, skeleton enum and coordinate conversion.
- [x] Human SEW, wrist, fingers, head and body conversion functions.
- [x] WebRTC pose transport and client-created feedback channels.
- [x] Bounded pose processing, stale snapshots, disconnect clearing, shutdown.
- [x] CSV reader including absolute-time and single-frame playback corrections.
- [x] Bounded live bone CSV writer with exclusive file creation and round-trip test.
- [x] MediaPipe Tasks adapter around transferred Python landmark processing.
- [x] Automatic versioned default model downloads with reusable cache, atomic
  installation and explicit local-path overrides; both real downloads verified.
- [x] Optional geo_kin_core integration: shared XR, MediaPipe, CSV and frame-stream adapters.
- [x] G1 and RBY1 dependencies, live entry points and replay paths migrated locally.

## Remaining

- [ ] Validate against actual XRT headset and camera/model files.
- [ ] Compare recorded outputs against source across robot-supported configurations.
- [ ] Transfer extended study recording (tags/actions), process proxy and telemetry.

## Validation

- 20 offline tests passed: shared core adapters, wire conversion/rejection, stale/disconnected snapshots,
  queue behavior, MediaPipe result/hand alignment, CSV replay/round trip, real local
  WebRTC pose/feedback exchange, and HTTP server start/shutdown.
- Official MediaPipe pose-lite and hand Tasks models executed on a blank image
  with the CPU delegate; no detections, as expected. No camera opened.
- Wheel and source distribution built with both package and upstream MIT licenses.
  Installed wheel into an isolated temporary directory and imported it from /tmp
  with Torch and the old server package blocked; package, processing, transport,
  CSV and MediaPipe facade imports passed.
- Full-skeleton synthetic XR conversion returned finite 18-value arm vectors.
- Live headset/camera and hardware behavior remain unverified.
- Final device suite: 20 passed. G1: 16 passed, 4 geo tests deselected;
  RBY1: 11 passed, 5 geo tests deselected. Both robots completed three-frame
  headless MINK replay checks; this does not establish analytic parity or safety.
  CI declares 3.10/3.12; remote CI has
  not run yet. Changes are local and have not been committed or pushed.
- [ ] Publish and pin device revisions for downstream CI; migrate remaining consumers.
- [ ] Validate mirrored-camera handedness and missing-hand behavior on real footage.
- [ ] Profile full input pipeline before considering Rust.

## Provenance

Transport/schema/utils derive from Yunho Cho's MIT-licensed
[XR-Robot-Teleop-Server](https://github.com/yunho-c/XR-Robot-Teleop-Server), revision
`5c21ed8a1e25cee8e37e4dc1805ceb41ffe71c72`. The original license is preserved verbatim
in `THIRD_PARTY_LICENSES/XR-Robot-Teleop-Server.txt`, including its placeholder notice.
Only pose transport and supporting helpers were transferred; sources/transforms
that caused video/Torch imports are absent.

Human processing derives from kczttm's existing Python device code. XR conversion
was extracted from public proj_XRT revision `fcfa9d4b1385916b4d49383f1d7b41706bf68b21`,
`projects/shared_devices/xr_robot_teleop_client.py`. MediaPipe/CSV source provenance
is retained in the local cross-repository tracker without exposing private paths here.

This package is MIT-licensed and has no geo_kin license check or analytic robot IK.
