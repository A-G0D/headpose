import Foundation
import simd

public typealias V3 = SIMD3<Double>
public let deg = 180 / Double.pi

// ---------- geometry (covered by selftest)

/// World-up in the camera frame (x: your right, y: up the lid, z: optical axis toward you).
/// Base frame assumed CoreMotion-style: x right, y toward hinge, z out of the keyboard; flat reads (0,0,-1).
public func upInCamera(accel r: V3, lidDeg: Double) -> V3 {
    let u = -simd_normalize(r), t = lidDeg / deg
    return V3(u.x, -u.y * cos(t) + u.z * sin(t), -u.y * sin(t) - u.z * cos(t))
}

/// Camera-frame point -> gravity frame. Undefined only if the camera points straight up or down.
public func toWorld(_ p: V3, up: V3) -> V3 {
    let z = simd_normalize(V3(0, 0, 1) - up.z * up) // optical axis flattened onto the horizontal
    return V3(simd_dot(p, simd_cross(up, z)), simd_dot(p, up), simd_dot(p, z))
}

/// Depth (mm) for this frame. Eye spacing is only trusted near frontal: on a turn the far eye's landmarks hide and
/// collapse, which read as the head jumping away and sideways (seen in a recorded run: 25 cm read as 86 cm).
/// A frontal face has eye spacing about half its box height; frames far below that are bad landmarks.
public func depth(held: Double?, k: Double, yaw: Double, eyePx: Double, boxPx: Double) -> (held: Double?, z: Double) {
    if abs(yaw) < 20 / deg, eyePx / boxPx > 0.4 { let z = k * cos(yaw) / eyePx; return (z, z) }
    return (held, held ?? k * max(cos(yaw), 0.5) / eyePx)
}

/// Pinhole: depth = k / eye spacing in px, k = focal px * eye spacing mm. Eye line foreshortens by cos(yaw).
/// Image is unmirrored with y up, so image-right is your left.
public func headInCamera(mid: CGPoint, eyePx: Double, yaw: Double, w: Double, h: Double, k: Double, f: Double, z depthMm: Double? = nil) -> V3 {
    let z = depthMm ?? k * max(cos(yaw), 0.5) / eyePx
    return V3(-(mid.x - w / 2) * z / f, (mid.y - h / 2) * z / f, z)
}

/// Head roll about its own nose axis, against gravity (+ = toward the right shoulder).
/// eyeX, eyeY: direction from your left eye to your right eye in the camera frame's x/y (image x flipped).
/// The ear-to-ear axis is square to the nose, which fixes its depth component; a camera looking up at a turned head
/// sees a slanted eye line with no tilt at all, and this keeps that at zero.
public func worldRoll(eyeX: Double, eyeY: Double, fwd: V3, up: V3) -> Double {
    let ears = simd_normalize(V3(eyeX, eyeY, -(eyeX * fwd.x + eyeY * fwd.y) / min(fwd.z, -0.05)))
    let fw = toWorld(fwd, up: up), headUp = simd_cross(toWorld(ears, up: up), fw)
    let right0 = simd_normalize(simd_cross(fw, V3(0, 1, 0)))
    return atan2(simd_dot(headUp, right0), simd_dot(headUp, simd_cross(right0, fw)))
}

/// Tilt is read from the eye line, which turns unreliable as the head turns away and the far eye hides
/// (recorded: worst-10% tilt jitter 1.5 degrees facing the camera, 12-14 degrees beyond 45 degrees of turn, single-frame
/// swings of 100+). Full trust up to 20 degrees of turn, none past 40; in between it blends toward the last trusted tilt.
public func trustedRoll(raw: Double, yawDeg: Double, last: Double) -> (roll: Double, last: Double) {
    let trust = min(max((40 - abs(yawDeg)) / 20, 0), 1)
    let roll = trust * raw + (1 - trust) * last
    return (roll, trust == 1 ? raw : last)
}

/// Where a point sits relative to the nose: degrees to the right (+) and above (+).
public func bearing(of src: V3, from l: (pos: V3, fwd: V3, up: V3)) -> (right: Double, above: Double) {
    let d = simd_normalize(src - l.pos)
    return (atan2(simd_dot(d, simd_cross(l.fwd, l.up)), simd_dot(d, l.fwd)) * deg, asin(simd_dot(d, l.up)) * deg)
}

/// k from one measured camera-to-eyes range (mm). The range is along the ray, depth is along the axis, so iterate.
public func solveK(eyePx: Double, offAxisPx: Double, rangeMm: Double, ipdMm: Double) -> Double {
    var k = eyePx * rangeMm
    for _ in 0..<5 { k = eyePx * rangeMm / (1 + pow(offAxisPx * ipdMm / k, 2)).squareRoot() }
    return k
}

/// One Euro filter (Casiez, Roussel, Vogel 2012): heavy smoothing at rest, light when moving.
/// `ahead` extrapolates along the filtered velocity and adds back the filter's own lag (tau), so a steady turn has no delay.
public struct OneEuro {
    public var minCut: Double, beta: Double, still: Double // still: speed under which the pose is held, not extrapolated
    public var x = 0.0, dx = 0.0, raw = 0.0, t = 0.0, tau = 0.0, started = false
    public init(minCut: Double, beta: Double, still: Double) { (self.minCut, self.beta, self.still) = (minCut, beta, still) }

    public mutating func step(_ v: Double, at now: Double) {
        guard started else { (x, raw, t, started) = (v, v, now, true); return }
        let dt = max(now - t, 1e-3)
        func alpha(_ cut: Double) -> Double { 1 / (1 + 1 / (2 * .pi * cut * dt)) }
        dx += alpha(dCut) * ((v - raw) / dt - dx)
        let cut = minCut + beta * abs(dx)
        tau = 1 / (2 * .pi * cut)
        x += alpha(cut) * (v - x)
        (raw, t) = (v, now)
    }

    /// Estimate for time `now + lead`; `t` is the capture time, so camera delay is covered too.
    /// Output delay is only predicted up to maxLead: guessing 250 ms of Bluetooth delay ahead overshoots every stop.
    public func ahead(_ now: Double, lead: Double) -> Double {
        let moving = min(max((abs(dx) - still) / still, 0), 1)
        return x + dx * moving * (min(now - t, 0.1) + min(lead, maxLead) + tau)
    }
}
public let dCut = 5.0, maxLead = 0.03

/// Listener for the audio engine, same frame as the output: position in metres (head centre, about 9 cm behind the eyes),
/// forward and up unit vectors.
public func listenerPose(yawDeg: Double, pitchDeg: Double, rollDeg: Double, posCm: V3) -> (pos: V3, fwd: V3, up: V3) {
    let (y, p, r) = (yawDeg / deg, pitchDeg / deg, rollDeg / deg)
    let fwd = V3(sin(y) * cos(p), sin(p), -cos(y) * cos(p))
    let right = simd_normalize(simd_cross(fwd, V3(0, 1, 0)))
    return (posCm / 100 - 0.09 * fwd, fwd, cos(r) * simd_cross(right, fwd) + sin(r) * right)
}

/// Everything the tracker derives from one frame's eye landmarks and Vision's yaw/pitch (radians in, degrees and cm out).
/// `mid`, `eyePx`, `boxPx`, `w`, `h` are in image px (y up); `f` is the focal length in px, `ipdMm` the eye spacing.
public func estimatePose(mid: CGPoint, eyePx: Double, boxPx: Double, w: Double, h: Double, vYaw: Double, vPitch: Double,
                         f: Double, ipdMm: Double, up: V3, yawSign: Double, held: Double?)
    -> (pc: V3, fwd: V3, posCm: V3, fw: V3, yaw: Double, pitch: Double, held: Double?) {
    let d = depth(held: held, k: f * ipdMm, yaw: vYaw, eyePx: eyePx, boxPx: boxPx)
    let pc = headInCamera(mid: mid, eyePx: eyePx, yaw: vYaw, w: w, h: h, k: f * ipdMm, f: f, z: d.z)
    let yawC = yawSign * vYaw - atan2(pc.x, pc.z), pitchC = -vPitch - atan2(pc.y, hypot(pc.x, pc.z))
    let fwd = V3(sin(yawC) * cos(pitchC), sin(pitchC), -cos(yawC) * cos(pitchC)) // where the face points, camera frame
    let fw = toWorld(fwd, up: up)
    return (pc, fwd, toWorld(pc, up: up) / 10, fw, atan2(fw.x, -fw.z) * deg, asin(fw.y) * deg, d.held)
}
