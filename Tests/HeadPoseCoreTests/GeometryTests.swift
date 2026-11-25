import XCTest
import simd
@testable import HeadPoseCore

final class GeometryTests: XCTestCase {
    func testDistanceFromEyeSpacing() { // 1000 px focal, 63 mm eyes, 90 px apart -> 700 mm
        let d = depth(held: nil, k: 1000 * 63, yaw: 0, eyePx: 90, boxPx: 180)
        XCTAssertEqual(d.z, 700, accuracy: 1e-9)
    }
    func testDistanceHoldsOnBadLandmarks() { // eye spacing collapsed to 20% of the box: keep the last depth
        XCTAssertEqual(depth(held: 500, k: 63000, yaw: 0, eyePx: 36, boxPx: 180).z, 500)
    }
    func testHeadInCameraOnAxisAndOffAxis() {
        let c = headInCamera(mid: CGPoint(x: 640, y: 360), eyePx: 90, yaw: 0, w: 1280, h: 720, k: 63000, f: 1000)
        XCTAssertEqual(c.x, 0, accuracy: 1e-9); XCTAssertEqual(c.z, 700, accuracy: 1e-9)
        let o = headInCamera(mid: CGPoint(x: 540, y: 360), eyePx: 90, yaw: 0, w: 1280, h: 720, k: 63000, f: 1000)
        XCTAssertEqual(o.x, 70, accuracy: 1e-9) // image-left of centre = your right = +x
    }
    func testYawPitchOfFrontalFaceOnAxis() {
        let e = estimatePose(mid: CGPoint(x: 640, y: 360), eyePx: 90, boxPx: 180, w: 1280, h: 720, vYaw: 0.3, vPitch: -0.2,
                             f: 1000, ipdMm: 63, up: V3(0, 1, 0), yawSign: -1, held: nil)
        XCTAssertEqual(e.yaw, -0.3 * deg, accuracy: 1e-9)
        XCTAssertEqual(e.pitch, 0.2 * deg, accuracy: 1e-9)
        XCTAssertEqual(simd_length(e.posCm), 70 * cos(0.3), accuracy: 1e-6) // depth = k cos(yaw) / eyePx
    }
    func testOffAxisRayIsSubtractedFromYaw() { // head 45 degrees right of the axis looking along the axis => own yaw 0
        let z = 500.0, x = 500.0, vYaw = -atan2(x, z) // Vision sees it turned toward the camera by the ray angle
        let e = estimatePose(mid: CGPoint(x: 640 - 1000 * x / z, y: 360), eyePx: 1000 * 63 * cos(vYaw) / z, boxPx: 400, w: 1280, h: 720,
                             vYaw: vYaw, vPitch: 0, f: 1000, ipdMm: 63, up: V3(0, 1, 0), yawSign: -1, held: nil)
        XCTAssertEqual(e.yaw, 0, accuracy: 1e-6)
    }
    func testToWorldLevelCameraIsIdentity() {
        let p = toWorld(V3(1, 2, 3), up: V3(0, 1, 0))
        XCTAssertEqual(p.x, 1, accuracy: 1e-12); XCTAssertEqual(p.y, 2, accuracy: 1e-12); XCTAssertEqual(p.z, 3, accuracy: 1e-12)
    }
    func testSolveKRoundTrip() { // on-axis: k = eyePx * range
        XCTAssertEqual(solveK(eyePx: 126, offAxisPx: 0, rangeMm: 500, ipdMm: 63), 63000, accuracy: 1e-9)
        let k = solveK(eyePx: 120, offAxisPx: 200, rangeMm: 500, ipdMm: 63)
        let range = k / 120 * (1 + pow(200 * 63 / k, 2)).squareRoot()
        XCTAssertEqual(range, 500, accuracy: 1e-3) // five fixed-point steps
    }
    func testBearingAndListenerPose() {
        let l = listenerPose(yawDeg: 90, pitchDeg: 0, rollDeg: 0, posCm: V3(0, 0, 0))
        XCTAssertEqual(l.fwd.x, 1, accuracy: 1e-12)
        let b = bearing(of: V3(1, 0, 0) + l.pos, from: l)
        XCTAssertEqual(b.right, 0, accuracy: 1e-9)
    }
    func testOneEuroHoldsConstant() {
        var f = OneEuro(minCut: 1, beta: 0.1, still: 20)
        for i in 0..<30 { f.step(10, at: Double(i) / 30) }
        XCTAssertEqual(f.ahead(1, lead: 0.01), 10, accuracy: 1e-9)
    }
}
