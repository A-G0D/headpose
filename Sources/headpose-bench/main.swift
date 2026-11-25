// Synthetic benchmark of the pose geometry. Faces come from the same pinhole model estimatePose inverts, so the
// noise-free row only checks consistency; the noisy rows show how landmark and Vision-angle noise reach distance and angle.
import Foundation
import HeadPoseCore
import simd

var seed: UInt64 = 42
func uni() -> Double { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Double(seed >> 11) / Double(1 << 53) }
func noise(_ amp: Double) -> Double { (uni() - 0.5) * 2 * amp } // uniform +-amp

let (w, h, f, ipd) = (1280.0, 720.0, 1000.0, 63.0), up = V3(0, 1, 0)
struct Case { var X, Y, Z, yaw, pitch: Double }
let cases = (0..<5000).map { _ in Case(X: noise(150), Y: noise(100), Z: 300 + uni() * 500, yaw: noise(35 / deg), pitch: noise(20 / deg)) }

func run(pxNoise: Double, angNoiseDeg: Double) -> (dist: Double, yaw: Double, pitch: Double) {
    var (ed, ey, ep) = (0.0, 0.0, 0.0)
    for c in cases {
        let el = atan2(c.Y, hypot(c.X, c.Z)), az = atan2(c.X, c.Z)
        // Vision reports the head angle relative to the camera ray; estimatePose adds the ray back (yawSign = -1)
        let vYaw = -(c.yaw + az) + noise(angNoiseDeg / deg), vPitch = -(c.pitch + el) + noise(angNoiseDeg / deg)
        let eyePx = f * ipd * cos(-(c.yaw + az)) / c.Z + noise(pxNoise)
        let mid = CGPoint(x: w / 2 - f * c.X / c.Z + noise(pxNoise), y: h / 2 + f * c.Y / c.Z + noise(pxNoise))
        let e = estimatePose(mid: mid, eyePx: eyePx, boxPx: 2 * eyePx, w: w, h: h, vYaw: vYaw, vPitch: vPitch, f: f, ipdMm: ipd, up: up, yawSign: -1, held: nil)
        ed += abs(simd_length(e.posCm) * 10 - simd_length(V3(c.X, c.Y, c.Z)))
        ey += abs(e.yaw - c.yaw * deg); ep += abs(e.pitch - c.pitch * deg)
    }
    let n = Double(cases.count)
    return (ed / n / 10, ey / n, ep / n)
}

let clean = run(pxNoise: 0, angNoiseDeg: 0), noisy = run(pxNoise: 0.5, angNoiseDeg: 2)
var sink = 0.0
let n = 200_000, t0 = ContinuousClock.now
for i in 0..<n {
    let e = estimatePose(mid: CGPoint(x: 640 + Double(i % 50), y: 360), eyePx: 80, boxPx: 160, w: w, h: h, vYaw: 0.1, vPitch: 0.05, f: f, ipdMm: ipd, up: up, yawSign: -1, held: nil)
    sink += e.yaw
}
let dt = (ContinuousClock.now - t0).components, us = (Double(dt.seconds) + Double(dt.attoseconds) * 1e-18) / Double(n) * 1e6
func r(_ x: Double) -> String { String(format: "%.4f", x) }
print("""
{"synthetic": true, "frames": \(cases.count), "camera": "\(Int(w))x\(Int(h)) f=\(Int(f))px ipd=\(Int(ipd))mm",
 "noise_free": {"dist_mae_cm": \(r(clean.dist)), "yaw_mae_deg": \(r(clean.yaw)), "pitch_mae_deg": \(r(clean.pitch))},
 "noisy": {"landmark_noise_px": 0.5, "vision_angle_noise_deg": 2, "dist_mae_cm": \(r(noisy.dist)), "yaw_mae_deg": \(r(noisy.yaw)), "pitch_mae_deg": \(r(noisy.pitch))},
 "estimate_pose_us_per_frame": \(r(us)), "sink": \(Int(sink) & 0)}
""")
