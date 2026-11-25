# headpose

A macOS Swift tool that tracks head distance, yaw and pitch from the built-in camera using Apple Vision, for people who want the head position as a listener pose.

## Demo

No demo is included. Running the tracker needs a camera, and this repo's tests and bench never open one.

## Quickstart

```
swift build -c release          # builds headpose, headpose-bench, HeadPoseCore
swift test                      # 9 geometry tests, no camera
.build/release/headpose-bench > results/bench.json   # synthetic bench
.build/release/headpose         # real run: opens the camera (needs camera permission)
```

`headpose music` additionally needs `models/htdemucs.mlpackage`, a CoreML export of Demucs (source separation). Head tracking itself needs no model download. The model is not committed (810 MB). Upstream Demucs is https://github.com/facebookresearch/demucs (MIT per upstream; the license of this particular CoreML export was not verified). Convert or download it yourself and place it at `models/htdemucs.mlpackage`.

## How it works

- Apple Vision gives a face box, eye landmarks, yaw and pitch per frame.
- Distance comes from the pixel spacing of the eyes: z = k * cos(yaw) / eyePx, with k from the focal length and a 63 mm interpupillary distance (or a calibrated value from `calib.json`).
- `estimatePose` in `Sources/HeadPoseCore/Geometry.swift` turns those inputs into a head position in camera space, then world space using the lid angle and accelerometer, and a yaw/pitch for the listener. The tracker calls this same function.
- A One-Euro filter smooths the angles.

## Results

| Metric | Ours | Market standard | Goal |
|---|---|---|---|
| Yaw / pitch MAE, noise-free (synthetic, generator shares the pinhole model with the method, a consistency check only) | 0.000 / 0.000 deg | BIWI MAE, 6DRepNet 3.47 deg mean ([README table](https://github.com/thohemp/6DRepNet), [paper](https://arxiv.org/abs/2202.12555)); FSA-Net 4.00, HopeNet 4.90 | not comparable |
| Yaw / pitch MAE, 0.5 px landmark noise and 2 deg angle noise (synthetic, not comparable to BIWI or AFLW2000) | 1.019 / 1.005 deg | AFLW2000 MAE, 6DRepNet 3.97, FSA-Net 5.07, HopeNet 6.16 deg | 2.3 deg (1.5x fewer errors than 3.47) on real faces; not measured |
| Distance MAE, same noisy synthetic set | 0.397 cm | no published figure found | not set |
| `estimatePose` compute per frame | 0.219 us | none | none |

Bench: 5000 synthetic frames, camera 1280x720, f = 1000 px, IPD 63 mm, Apple M5 Pro, `.build/release/headpose-bench`. The competitor numbers are transcribed from the 6DRepNet GitHub README by a fetch tool; the column-to-dataset assignment was inferred from that page's heading, so check against the paper before quoting. Those methods estimate full 3-DoF pose from images; our numbers measure only the geometry with the angle noise assumed, so no claim of beating them is made. The 2 deg Vision angle noise is an assumption, not a measurement on real faces.

## Limits

- macOS only (AVFoundation, Vision, CoreML, IOKit).
- Nothing in this task was run on a camera. Real-face accuracy is unmeasured.
- Yaw and pitch come from Apple Vision; this code does not improve them.
- Without `calib.json`, field of view 65 deg and IPD 63 mm are assumed, so distance is off by the difference between that and your face.
- No CI: the package needs macOS frameworks.

## License

MIT. See LICENSE.
