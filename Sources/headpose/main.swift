// headpose: where your head is relative to the MacBook camera, in a gravity-aligned frame.
//
// Build:  swiftc -O -swift-version 5 headpose.swift -o headpose
// Run:    ./headpose                      JSON lines on stdout, ~15 Hz
//         ./headpose calibrate 50 [63]    eyes 50 cm from the camera, facing it; optional own IPD in mm
//         ./headpose view                 window: camera feed with the head direction drawn on it
//         ./headpose audio [file[@x,y,z]] head-tracked binaural sound on headphones; sources stay fixed in the room.
//                                         No file: a steady thud at the screen centre. x,y,z in metres in the output frame.
//                                         `view audio` runs both. HEADPOSE_VOL=0..1 (default 0.6),
//                                         HEADPOSE_OUT=<part of a device name> overrides headphone detection.
//                                         Keys while it runs: [ weaker, ] stronger response to head turns.
//                                         Every view/audio/check run records what the tracker saw to trace.jsonl.
//         ./headpose music [app]          an app's sound (default Spotify), its own output muted while captured, split
//                                         into vocals, drums, bass and other (models/htdemucs.mlpackage) and placed
//                                         around a virtual room behind the screen. `view music`. HEADPOSE_STEMS=0: no split.
//         ./headpose check                walks you through five head poses and reports what the tracker saw
//         ./headpose selftest             checks the geometry
//
// Output frame, origin at the camera: x = your right, y = up (against gravity), z = level, from screen toward you.
// Head point = midpoint between the eyes. yaw + = turned to your right, pitch + = chin up, roll + = toward right shoulder.
import AVFoundation
import AppKit
import AudioToolbox
import CoreAudio
import CoreML
import IOKit
import Synchronization
import IOKit.hid
import Vision
import simd
import HeadPoseCore


/// Worst angle error on a synthetic head turn seen through a 30 fps camera with noise and delay:
/// the filter with prediction, against simply using the newest camera sample.
func filterErrors() -> (rest: Double, worst: Double, rawWorst: Double) {
    var f = OneEuro(minCut: angleCut, beta: angleBeta, still: angleStill), seed: UInt64 = 42, newest = 0.0
    func noise() -> Double { // uniform, sd 0.7 degrees
        seed = seed &* 6364136223846793005 &+ 1442695040888963407
        return (Double(seed >> 11) / Double(1 << 53) - 0.5) * 2.42
    }
    func truth(_ t: Double) -> Double { // minimum-jerk 80 degree turn in 0.8 s (peak 187 deg/s), the shape real head turns have
        let s = min(max((t - 2) / 0.8, 0), 1)
        return 80 * (10 * pow(s, 3) - 15 * pow(s, 4) + 6 * pow(s, 5))
    }
    let camDelay = 0.07, lead = 0.01
    var rest = 0.0, restN = 0.0, worst = 0.0, rawWorst = 0.0, nextFrame = 0.0
    for i in 0..<1000 {
        let now = Double(i) * 0.005
        while nextFrame + camDelay <= now { newest = truth(nextFrame) + noise(); f.step(newest, at: nextFrame); nextFrame += 1.0 / 30 }
        let err = f.ahead(now, lead: lead) - truth(now + lead)
        if now > 0.5, now < 1.9 { rest += err * err; restN += 1 }
        if now >= 2 { worst = max(worst, abs(err)); rawWorst = max(rawWorst, abs(newest - truth(now + lead))) }
    }
    return ((rest / restN).squareRoot(), worst, rawWorst)
}

/// Offline render of a source at the Mac: right-ear minus left-ear level in dB for a given head pose.
func earBalance(yawDeg: Double, posCm: V3) -> Double {
    let noise = AVAudioPCMBuffer(pcmFormat: AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 1)!, frameCapacity: 96000)!
    noise.frameLength = 96000
    for i in 0..<96000 { noise.floatChannelData![0][i] = Float.random(in: -0.3...0.3) } // white noise: all frequencies count
    let s = Spatial([(noise, V3(0, 0, 0))])
    try! s.engine.enableManualRenderingMode(.offline, format: AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2)!, maximumFrameCount: 4096)
    s.setListener(listenerPose(yawDeg: yawDeg, pitchDeg: 0, rollDeg: 0, posCm: posCm))
    try! s.start(rate: 48000)
    let out = AVAudioPCMBuffer(pcmFormat: s.engine.manualRenderingFormat, frameCapacity: 4096)!
    var l = 0.0, r = 0.0
    for _ in 0..<12 {
        _ = try! s.engine.renderOffline(4096, to: out)
        for i in 0..<Int(out.frameLength) {
            l += Double(out.floatChannelData![0][i] * out.floatChannelData![0][i])
            r += Double(out.floatChannelData![1][i] * out.floatChannelData![1][i])
        }
    }
    precondition(l > 0 && r > 0, "silent render")
    return 10 * log10(r / l)
}

/// Offline render of the music scene with white noise: ear similarity (peak interaural cross-correlation within
/// +-1 ms, the usual measure of how far out of the head a sound sits), interaural delay and level, and output level.
func measureMusic(room: Bool, yaw: Double, left: Bool = true, right: Bool = true) -> (iacc: Double, itd: Double, ild: Double, gainDb: Double) {
    let rate = 48000.0
    let feed = StereoFeed(rate: rate, taps: musicScene(rate: rate, room: room).map(\.tap))
    let s = musicSpatial(feed, scene: musicScene(rate: rate, room: room))
    try! s.engine.enableManualRenderingMode(.offline, format: AVAudioFormat(standardFormatWithSampleRate: rate, channels: 2)!, maximumFrameCount: 4096)
    let ears = nominalEars * 100 // listener tick works in cm
    s.setListener(listenerPose(yawDeg: yaw, pitchDeg: 0, rollDeg: 0, posCm: ears + 9 * V3(sin(yaw / deg), 0, -cos(yaw / deg))))
    try! s.start(rate: rate)
    var g = SystemRandomNumberGenerator()
    let noise = (0..<(26 * 4096)).map { _ in Float.random(in: -0.3...0.3, using: &g) }, silent = [Float](repeating: 0, count: 26 * 4096)
    let out = AVAudioPCMBuffer(pcmFormat: s.engine.manualRenderingFormat, frameCapacity: 4096)!
    var L: [Float] = [], R: [Float] = []
    for i in 0..<26 {
        noise.withUnsafeBufferPointer { n in silent.withUnsafeBufferPointer { z in // block by block, as live capture arrives
            let at = i * 4096
            feed.write(left: (left ? n.baseAddress! : z.baseAddress!) + at, right: (right ? n.baseAddress! : z.baseAddress!) + at, stride: 1, frames: 4096)
        } }
        _ = try! s.engine.renderOffline(4096, to: out)
        if i >= 6 { // skip the first half second: priming and the reverb filling up
            L += UnsafeBufferPointer(start: out.floatChannelData![0], count: 4096)
            R += UnsafeBufferPointer(start: out.floatChannelData![1], count: 4096)
        }
    }
    let l = L.reduce(0) { $0 + Double($1 * $1) }, r = R.reduce(0) { $0 + Double($1 * $1) }
    var best = 0, bestV = -Double.infinity
    for lag in -48...48 { // +-1 ms
        var v = 0.0
        for i in 64..<(L.count - 64) { v += Double(L[i] * R[i + lag]) }
        if v > bestV { (best, bestV) = (lag, v) }
    }
    let inputPower = Double(noise.reduce(0) { $0 + $1 * $1 }) / Double(noise.count) * Double((left ? 1 : 0) + (right ? 1 : 0)) / 2
    return (bestV / (l * r).squareRoot(), Double(best) / rate * 1e6, 10 * log10(r / max(l, 1e-20)),
            10 * log10((l + r) / 2 / Double(L.count) / inputPower))
}

func selftest() {
    func near(_ a: V3, _ b: V3, _ tol: Double = 1e-6) { precondition(simd_length(a - b) < tol, "\(a) != \(b)") }
    let s20 = sin(20 / deg), c20 = cos(20 / deg)
    let up = upInCamera(accel: V3(0, 0, -1), lidDeg: 110)
    near(up, V3(0, c20, s20)) // lid 110 on a level desk: camera looks 20 degrees above level
    near(toWorld(V3(0, 0, 600), up: up), V3(0, 600 * s20, 600 * c20))
    // front edge raised 10 degrees with lid at 100 is the same camera attitude as level with lid at 110
    near(upInCamera(accel: V3(0, sin(10 / deg), -cos(10 / deg)), lidDeg: 100), up)
    // right side dropped 5 degrees: a point on the image's up axis lands to your right of true vertical
    precondition(toWorld(V3(0, 100, 500), up: upInCamera(accel: V3(sin(5 / deg), 0, -cos(5 / deg)), lidDeg: 90)).x > 0)
    near(headInCamera(mid: CGPoint(x: 640, y: 360), eyePx: 100, yaw: 0, w: 1280, h: 720, k: 50000, f: 900), V3(0, 0, 500))
    precondition(headInCamera(mid: CGPoint(x: 700, y: 360), eyePx: 100, yaw: 0, w: 1280, h: 720, k: 50000, f: 900).x < 0)
    let k = solveK(eyePx: 100, offAxisPx: 200, rangeMm: 500, ipdMm: 63) // must reproduce the 500 mm range
    let p = headInCamera(mid: CGPoint(x: 640, y: 560), eyePx: 100, yaw: 0, w: 1280, h: 720, k: k, f: k / 63)
    precondition(abs(simd_length(p) - 500) < 0.01, "range \(simd_length(p))")
    // listener: facing the screen means forward -z, up +y; rolling right tips the up vector to +x
    let lp = listenerPose(yawDeg: 0, pitchDeg: 0, rollDeg: 0, posCm: V3(0, 0, 60))
    near(lp.fwd, V3(0, 0, -1)); near(lp.up, V3(0, 1, 0)); near(lp.pos, V3(0, 0, 0.69))
    precondition(listenerPose(yawDeg: 30, pitchDeg: 0, rollDeg: 0, posCm: .zero).fwd.x > 0)
    precondition(listenerPose(yawDeg: 0, pitchDeg: 0, rollDeg: 20, posCm: .zero).up.x > 0)

    // roll: level camera, frontal face, eye line dipping 20 degrees toward your right = 20 degrees of roll
    precondition(abs(worldRoll(eyeX: cos(20 / deg), eyeY: -sin(20 / deg), fwd: V3(0, 0, -1), up: V3(0, 1, 0)) * deg - 20) < 1e-6)
    // camera looking up 36 degrees at a head turned 45 degrees with no tilt: the eye line slants in the image, roll must stay 0
    let up36 = upInCamera(accel: V3(0, 0, -1), lidDeg: 126)
    func toCamera(_ v: V3) -> V3 { let z = simd_normalize(V3(0, 0, 1) - up36.z * up36); return v.x * simd_cross(up36, z) + v.y * up36 + v.z * z }
    let f45 = V3(sin(45 / deg), 0, -cos(45 / deg)), r45 = toCamera(simd_cross(f45, V3(0, 1, 0)))
    precondition(abs(atan2(r45.y, r45.x)) * deg > 15, "test head should show a slanted eye line")
    precondition(abs(worldRoll(eyeX: r45.x, eyeY: r45.y, fwd: toCamera(f45), up: up36)) * deg < 1e-6)
    let b = bearing(of: V3(1, 0, -1), from: (V3(0, 0, 0), V3(0, 0, -1), V3(0, 1, 0)))
    precondition(abs(b.right - 45) < 1e-6 && abs(b.above) < 1e-6)

    // held depth: a frontal frame sets it, a turned frame or one with collapsed eye landmarks keeps it
    let d1 = depth(held: nil, k: 50000, yaw: 0.05, eyePx: 100, boxPx: 200)
    precondition(abs(d1.z - 50000 * cos(0.05) / 100) < 1e-9 && d1.held == d1.z)
    precondition(depth(held: 600, k: 50000, yaw: 0.05, eyePx: 30, boxPx: 200) == (600, 600)) // collapsed landmarks
    precondition(depth(held: 600, k: 50000, yaw: 0.9, eyePx: 60, boxPx: 200) == (600, 600))  // turned 52 degrees
    precondition(depth(held: nil, k: 50000, yaw: 0.9, eyePx: 60, boxPx: 200).held == nil)

    // tilt: trusted facing the camera, held while turned far, blended in between
    precondition(trustedRoll(raw: 10, yawDeg: 15, last: 0) == (10, 10))
    precondition(trustedRoll(raw: 90, yawDeg: 55, last: 4) == (4, 4))
    precondition(trustedRoll(raw: 20, yawDeg: 30, last: 0).roll == 10)

    let e = filterErrors()
    print(String(format: "filter: rest jitter %.2f deg rms; fast turn worst error %.1f deg predicted vs %.1f deg unpredicted", e.rest, e.worst, e.rawWorst))
    precondition(e.rest < 0.6 && e.worst < 0.65 * e.rawWorst, "filter out of tune")

    // the source is at the Mac: turn right and it must move to the left ear, and the reverse
    let ahead = earBalance(yawDeg: 0, posCm: V3(0, 0, 60)), right = earBalance(yawDeg: 60, posCm: V3(0, 0, 60))
    let left = earBalance(yawDeg: -60, posCm: V3(0, 0, 60)), side = earBalance(yawDeg: 0, posCm: V3(50, 0, 60))
    print(String(format: "ears (right minus left, dB): facing %.1f, turned right %.1f, turned left %.1f, moved right %.1f", ahead, right, left, side))
    precondition(abs(ahead) < 1.5 && right < -8 && left > 8 && side < -5, "binaural render does not follow the head clearly")
    // music: channels to the right ears, then the room's ear similarity and whether direction survives it
    for (ch, name) in [(0, "left"), (1, "right")] {
        let m = measureMusic(room: true, yaw: 0, left: ch == 0, right: ch == 1)
        print(String(format: "music %@ channel only: right minus left %.1f dB", name, m.ild))
        precondition(ch == 0 ? m.ild < -3 : m.ild > 3, "app audio channels are not reaching the right ears")
    }
    // pass marks: ear similarity <= 0.8 (real rooms 0.3-0.7); a 60 degree turn keeps >= 90% of the dry interaural delay
    // (a +-27 degree speaker pair gives about 230 us, not the 250 us of one source) and >= 3 dB between the ears
    var dryDelay = 0.0
    for room in [false, true] {
        let front = measureMusic(room: room, yaw: 0), turned = measureMusic(room: room, yaw: 60)
        print(String(format: "music %@: ear similarity %.2f | turned 60 right: right ear later by %.0f us, right minus left %.1f dB | level %+.1f dB vs input",
                     room ? "with room" : "dry      ", front.iacc, turned.itd, turned.ild, front.gainDb))
        if !room { dryDelay = turned.itd; continue }
        precondition(front.iacc <= 0.8 && turned.itd >= 0.9 * dryDelay && dryDelay > 150 && turned.ild <= -3, "room is not out of head, or it hides direction")
    }
    // uneven pulls, right first: both channels must come out sample-aligned with the input
    let mono = AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 1)!
    let ramp = (0..<20000).map { Float($0) }
    var heard: [[Float]] = [[], [], []]
    let delayed = StereoFeed(rate: 48000, taps: [.init(channel: 0, delay: 0, gain: 1), .init(channel: 1, delay: 0, gain: 1), .init(channel: 0, delay: 10, gain: 0.5)])
    ramp.withUnsafeBufferPointer { delayed.write(left: $0.baseAddress!, right: $0.baseAddress!, stride: 1, frames: 20000) }
    for (ch, n) in [(2, 256), (1, 256), (1, 23), (0, 256), (0, 23), (2, 23), (0, 256), (1, 100), (1, 179), (2, 256), (0, 23)] {
        let buf = AVAudioPCMBuffer(pcmFormat: mono, frameCapacity: 512)!
        var ts = AudioTimeStamp(), silent = ObjCBool(false)
        _ = delayed.render(ch, &silent, &ts, AVAudioFrameCount(n), buf.mutableAudioBufferList)
        heard[ch] += Array(UnsafeBufferPointer(start: buf.floatChannelData![0], count: n))
    }
    let both = min(heard[0].count, heard[1].count)
    precondition(both > 500 && heard[0].prefix(both) == heard[1].prefix(both) && heard[0][both - 1] - heard[0][0] == Float(both - 1), "channels drift apart")
    precondition((10..<heard[2].count).allSatisfy { heard[2][$0] == 0.5 * heard[0][$0 - 10] }, "delayed tap is not a clean delayed copy")
    print("music taps stay sample-aligned over uneven pulls, delays exact")
    separatorCheck()
    print("selftest ok")
}

/// Streams a mix of four Apple loops (one instrument each, so the true stems are known) through the live separation
/// path: 48 kHz capture, resampling, 1 s hops, crossfades. Pass marks: drums and bass >= 10 dB SI-SDR against the
/// truth (a centred window scored 13-15 dB), and the stems summed back >= 20 dB against the mix.
func separatorCheck() {
    let dir = "/Library/Audio/Apple Loops/Apple/"
    let paths = ["13 Drummer/Kyle - Golden State.caf", "Apple Loops for GarageBand/Upright Funk Bass 01.caf",
                 "Apple Loops for GarageBand/Upbeat Electric Piano 01.caf"] // drums, bass, other
    guard let model = Separator.loadModel(), let stems = Optional(paths.compactMap { loadStereo(dir + $0, rate: 44100, frames: 16 * 44100) }), stems.count == 3 else {
        print("separation check skipped: model or GarageBand loops missing"); return
    }
    let n = 16 * 44100
    let truth = stems.map { s -> [[Float]] in // each stem at -20 dB RMS
        let rms = (s[0] + s[1]).reduce(0.0) { $0 + Double($1 * $1) } / Double(2 * n)
        return s.map { $0.map { $0 * Float(0.1 / rms.squareRoot()) } }
    }
    let mix = (0..<2).map { c in (0..<n).map { i in truth.reduce(Float(0)) { $0 + $1[c][i] } } }
    // to 48 kHz, as the tap delivers it
    let f44 = AVAudioFormat(standardFormatWithSampleRate: 44100, channels: 2)!, f48 = AVAudioFormat(standardFormatWithSampleRate: 48000, channels: 2)!
    let b44 = AVAudioPCMBuffer(pcmFormat: f44, frameCapacity: AVAudioFrameCount(n))!
    b44.frameLength = AVAudioFrameCount(n)
    for c in 0..<2 { mix[c].withUnsafeBufferPointer { b44.floatChannelData![c].update(from: $0.baseAddress!, count: n) } }
    let b48 = AVAudioPCMBuffer(pcmFormat: f48, frameCapacity: AVAudioFrameCount(n * 48000 / 44100 + 4096))!
    var given = false
    AVAudioConverter(from: f44, to: f48)!.convert(to: b48, error: nil) { _, st in if given { st.pointee = .endOfStream; return nil }; given = true; st.pointee = .haveData; return b44 }
    let sep = Separator(model: model, captureRate: 48000, taps: [.init(channel: 0, delay: 0, gain: 1)])
    var got = [[Float]](repeating: [], count: 7)
    for start in Swift.stride(from: 0, to: Int(b48.frameLength), by: 4800) { // 0.1 s blocks, as captured
        let k = min(4800, Int(b48.frameLength) - start)
        sep.capture.write(left: b48.floatChannelData![0] + start, right: b48.floatChannelData![1] + start, stride: 1, frames: k)
        sep.take(sep.capture.drain())
        while sep.step() {}
        let out = sep.output.drain()
        for c in 0..<7 { got[c] += out[c] }
    }
    // output sample k is input time k - look, plus the resampler's own delay: find it on the summed mix
    func sisdr(_ e: ArraySlice<Float>, _ r: ArraySlice<Float>) -> Double {
        let dot = zip(e, r).reduce(0.0) { $0 + Double($1.0 * $1.1) }, rr = r.reduce(0.0) { $0 + Double($1 * $1) }, a = dot / max(rr, 1e-20)
        var sig = 0.0, err = 0.0
        for (x, y) in zip(e, r) { let t = a * Double(y); sig += t * t; err += (Double(x) - t) * (Double(x) - t) }
        return 10 * log10(sig / max(err, 1e-20))
    }
    let r0 = 6 * 44100, r1 = 12 * 44100 // six segment joins inside
    var lag = 0, best = -Double.infinity
    for d in -200...200 {
        let v = sisdr(got[0][(r0 + sep.look + d)..<(r1 + sep.look + d)], mix[0][r0..<r1])
        if v > best { (lag, best) = (d, v) }
    }
    precondition(got[0].count > r1 + sep.look + 200, "separator produced too little audio")
    func score(_ ch: Int, _ stem: Int) -> Double { sisdr(got[ch][(r0 + sep.look + lag)..<(r1 + sep.look + lag)], ArraySlice(zip(truth[stem][0], truth[stem][1]).map { ($0 + $1) / 2 }[r0..<r1])) }
    let drums = score(3, 0), bass = score(4, 1)
    print(String(format: "separation, streamed: drums %.1f dB, bass %.1f dB vs the true stems; stems sum to the mix at %.1f dB (resampler delay %d samples)", drums, bass, best, lag))
    precondition(drums >= 10 && bass >= 10 && best >= 20, "streamed separation is worse than the pass marks")
}

// ---------- sensors (undocumented Apple SPU HID devices; formats from macimu and hinge-angle, checked on Mac17,9)

var hidManagers: [IOHIDManager] = [] // devices die with their manager
func hidDevice(_ page: Int, _ usage: Int, _ product: String) -> IOHIDDevice? {
    let m = IOHIDManagerCreate(kCFAllocatorDefault, 0)
    IOHIDManagerSetDeviceMatching(m, [kIOHIDDeviceUsagePageKey: page, kIOHIDDeviceUsageKey: usage] as CFDictionary)
    IOHIDManagerOpen(m, 0)
    hidManagers.append(m)
    let d = (IOHIDManagerCopyDevices(m) as? Set<IOHIDDevice>)?
        .first { IOHIDDeviceGetProperty($0, kIOHIDProductKey as CFString) as? String == product }
    if let d { IOHIDDeviceOpen(d, 0) }
    return d
}

/// Lid angle in whole degrees: feature report 1, little-endian u16, low 9 bits.
func lidAngle(_ d: IOHIDDevice) -> Double? {
    var buf = [UInt8](repeating: 0, count: 3), n = 3
    guard IOHIDDeviceGetReport(d, kIOHIDReportTypeFeature, 1, &buf, &n) == kIOReturnSuccess, n == 3 else { return nil }
    return Double((UInt16(buf[2]) << 8 | UInt16(buf[1])) & 0x1FF)
}

var gravity = V3(0, 0, -1) // level base until the accelerometer reports
var accelSeen = false
let accelBuf = UnsafeMutablePointer<UInt8>.allocate(capacity: 64)
func startAccel() {
    var it: io_iterator_t = 0
    IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("AppleSPUHIDDriver"), &it)
    while case let s = IOIteratorNext(it), s != 0 {
        if IORegistryEntryCreateCFProperty(s, "Product" as CFString, nil, 0)?.takeRetainedValue() as? String == "accel" {
            // the driver drops these back to 0 when this process exits
            for (key, v) in [("SensorPropertyReportingState", 1), ("SensorPropertyPowerState", 1), ("ReportInterval", 20000)] {
                IORegistryEntrySetCFProperty(s, key as CFString, v as CFNumber)
            }
        }
        IOObjectRelease(s)
    }
    guard let d = hidDevice(0xFF00, 3, "accel") else { return }
    IOHIDDeviceRegisterInputReportCallback(d, accelBuf, 64, { _, _, _, _, _, r, len in
        guard len == 22 else { return }
        let p = UnsafeRawPointer(r) // x,y,z: int32 LE at byte 6, Q16 g
        let v = V3(Double(p.loadUnaligned(fromByteOffset: 6, as: Int32.self)),
                   Double(p.loadUnaligned(fromByteOffset: 10, as: Int32.self)),
                   Double(p.loadUnaligned(fromByteOffset: 14, as: Int32.self))) / 65536
        gravity = accelSeen ? gravity + 0.1 * (v - gravity) : v
        accelSeen = true
    }, nil)
    IOHIDDeviceScheduleWithRunLoop(d, CFRunLoopGetMain(), CFRunLoopMode.defaultMode.rawValue)
}

// ---------- head-tracked binaural audio

func note(_ m: String) { FileHandle.standardError.write((m + "\n").data(using: .utf8)!) }

/// Per-frame record of the last run, so a wrong angle can be traced to what the camera measured.
var traceFile: FileHandle?
func trace(_ line: String) { traceFile?.write((line + "\n").data(using: .utf8)!) }

/// How strongly a head turn moves the sound: 1 = as measured. Set by ear with the [ and ] keys.
var turnGain = 1.0, savedTerminal = termios()
func watchKeys() -> DispatchSourceRead? {
    guard isatty(0) != 0 else { return nil }
    tcgetattr(0, &savedTerminal)
    var raw = savedTerminal
    raw.c_lflag &= ~tcflag_t(ICANON | ECHO) // keys arrive at once, unechoed; Ctrl-C still works
    tcsetattr(0, TCSANOW, &raw)
    atexit { tcsetattr(0, TCSANOW, &savedTerminal) }
    for sig in [SIGINT, SIGTERM] { signal(sig) { _ in exit(0) } } // so the terminal is restored
    let keys = DispatchSource.makeReadSource(fileDescriptor: 0, queue: .main)
    keys.setEventHandler {
        var c: UInt8 = 0
        guard read(0, &c, 1) == 1, c == UInt8(ascii: "[") || c == UInt8(ascii: "]") else { return }
        turnGain = min(max(turnGain + (c == UInt8(ascii: "]") ? 0.1 : -0.1), 0.2), 1.5)
        note(String(format: "turn strength %.1f", turnGain))
    }
    keys.resume()
    return keys
}
let angleCut = 1.0, angleBeta = 0.1, angleStill = 20.0 // degrees; picked from a sweep against filterErrors()

/// Steady thud, about two a second: a low body with a short pitch drop, plus a 4 ms noise knock,
/// because the ears need a sharp broadband onset to place a sound.
func thud() -> AVAudioPCMBuffer {
    let sr = 48000.0, n = Int(sr * 0.45)
    let buf = AVAudioPCMBuffer(pcmFormat: AVAudioFormat(standardFormatWithSampleRate: sr, channels: 1)!, frameCapacity: AVAudioFrameCount(n))!
    buf.frameLength = AVAudioFrameCount(n)
    let out = buf.floatChannelData![0]
    var phase = 0.0
    for i in 0..<n {
        let t = Double(i) / sr
        phase += 2 * .pi * (55 + 110 * exp(-t / 0.025)) / sr
        out[i] = Float(0.8 * sin(phase) * exp(-t / 0.11)) + (i < 960 ? 0.3 * Float.random(in: -1...1) * exp(-Float(i) / 190) : 0)
        if i > n - 480 { out[i] *= Float(n - i) / 480 } // no click at the loop point
    }
    return buf
}

/// Whole file as stereo float at `rate`, looped or cut to `frames`.
func loadStereo(_ path: String, rate: Double, frames: Int) -> [[Float]]? {
    guard let file = try? AVAudioFile(forReading: URL(fileURLWithPath: path)),
          let src = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)), (try? file.read(into: src)) != nil,
          let outF = AVAudioFormat(standardFormatWithSampleRate: rate, channels: 2), let conv = AVAudioConverter(from: file.processingFormat, to: outF),
          let dst = AVAudioPCMBuffer(pcmFormat: outF, frameCapacity: AVAudioFrameCount(Double(file.length) * rate / file.processingFormat.sampleRate) + 4096) else { return nil }
    var given = false
    conv.convert(to: dst, error: nil) { _, st in if given { st.pointee = .endOfStream; return nil }; given = true; st.pointee = .haveData; return src }
    let n = Int(dst.frameLength)
    return n == 0 ? nil : (0..<2).map { c in (0..<frames).map { dst.floatChannelData![c][$0 % n] } }
}

func loadMono(_ path: String) -> AVAudioPCMBuffer? {
    guard let f = try? AVAudioFile(forReading: URL(fileURLWithPath: path)),
          let full = AVAudioPCMBuffer(pcmFormat: f.processingFormat, frameCapacity: AVAudioFrameCount(f.length)),
          (try? f.read(into: full)) != nil, full.frameLength > 0,
          let mono = AVAudioPCMBuffer(pcmFormat: AVAudioFormat(standardFormatWithSampleRate: f.processingFormat.sampleRate, channels: 1)!,
                                      frameCapacity: full.frameLength) else { return nil }
    mono.frameLength = full.frameLength
    let ch = Int(f.processingFormat.channelCount)
    for i in 0..<Int(full.frameLength) {
        mono.floatChannelData![0][i] = (0..<ch).reduce(0) { $0 + full.floatChannelData![$1][i] } / Float(ch)
    }
    return mono
}

func audioGet<T>(_ id: AudioObjectID, _ sel: AudioObjectPropertySelector, into v: inout T) -> Bool {
    var a = AudioObjectPropertyAddress(mSelector: sel, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
    var n = UInt32(MemoryLayout<T>.size)
    return AudioObjectGetPropertyData(id, &a, 0, nil, &n, &v) == noErr
}

/// First output device that looks like headphones: Bluetooth, or named like them (the wired jack shows as "External Headphones").
func headphones() -> (id: AudioDeviceID, name: String)? {
    let sys = AudioObjectID(kAudioObjectSystemObject)
    var a = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyDevices, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
    var n: UInt32 = 0
    AudioObjectGetPropertyDataSize(sys, &a, 0, nil, &n)
    var ids = [AudioDeviceID](repeating: 0, count: Int(n) / MemoryLayout<AudioDeviceID>.size)
    AudioObjectGetPropertyData(sys, &a, 0, nil, &n, &ids)
    let want = ProcessInfo.processInfo.environment["HEADPOSE_OUT"]?.lowercased()
    for id in ids {
        var outs = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyStreams, mScope: kAudioObjectPropertyScopeOutput, mElement: kAudioObjectPropertyElementMain)
        var size: UInt32 = 0, cf: Unmanaged<CFString>?, transport: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(id, &outs, 0, nil, &size) == noErr, size > 0,
              audioGet(id, kAudioObjectPropertyName, into: &cf), let name = cf?.takeRetainedValue() as String? else { continue }
        _ = audioGet(id, kAudioDevicePropertyTransportType, into: &transport)
        let l = name.lowercased()
        let bluetooth = transport == kAudioDeviceTransportTypeBluetooth || transport == kAudioDeviceTransportTypeBluetoothLE
        if want.map({ l.contains($0) }) ?? (bluetooth || ["headphone", "headset", "airpods", "buds"].contains { l.contains($0) }) { return (id, name) }
    }
    return nil
}

/// Sources fixed in the room, listener = the tracked head, rendered with Apple's HRTF. All calls on one queue.
final class Spatial {
    let engine = AVAudioEngine(), env = AVAudioEnvironmentNode()
    var players: [(node: AVAudioPlayerNode, buf: AVAudioPCMBuffer)] = []
    var device: AudioDeviceID? = 0 // 0 is never a real device, so the first route() always acts
    var lead = 0.0, camLag = 0.0, failures = 0
    var extraStatus: (() -> String)?
    var mixers: [AVAudioMixing] = [] // every spatialised input, same order as sourceAt
    var offsets: [V3]? // metres from the screen centre; set = sources move with the screen as the lid or base moves
    var sourceAt: [V3] = [], listener: (pos: V3, fwd: V3, up: V3)?, level: (l: Float, r: Float) = (0, 0)
    var angles = [OneEuro](repeating: OneEuro(minCut: angleCut, beta: angleBeta, still: angleStill), count: 3) // yaw, pitch, roll
    var place = [OneEuro](repeating: OneEuro(minCut: 1, beta: 0.1, still: 4), count: 3) // cm

    init(_ sources: [(AVAudioPCMBuffer, V3)], live: [(node: AVAudioSourceNode, format: AVAudioFormat, at: V3)] = []) {
        engine.attach(env)
        env.outputType = .headphones
        env.distanceAttenuationParameters.referenceDistance = 0.3
        // no reverb: the factory room presets are louder than the direct sound and erase its direction (measured)
        for (buf, at) in sources {
            let p = AVAudioPlayerNode()
            engine.attach(p)
            engine.connect(p, to: env, format: buf.format) // mono in, so the environment node spatialises it
            p.renderingAlgorithm = .HRTFHQ
            p.position = AVAudio3DPoint(x: Float(at.x), y: Float(at.y), z: Float(at.z))
            players.append((p, buf))
            mixers.append(p)
            sourceAt.append(at)
        }
        for (node, format, at) in live {
            engine.attach(node)
            engine.connect(node, to: env, format: format)
            node.renderingAlgorithm = .HRTFHQ
            node.position = AVAudio3DPoint(x: Float(at.x), y: Float(at.y), z: Float(at.z))
            mixers.append(node)
            sourceAt.append(at)
        }
    }

    var primary = Int.max // the first `primary` sources are the real ones; the rest are reflections
    var centre: V3 { let p = sourceAt.prefix(primary); return p.reduce(.zero, +) / Double(max(p.count, 1)) }

    func start(rate: Double) throws {
        engine.connect(env, to: engine.mainMixerNode, format: AVAudioFormat(standardFormatWithSampleRate: rate, channels: 2))
        engine.mainMixerNode.removeTap(onBus: 0)
        engine.mainMixerNode.installTap(onBus: 0, bufferSize: 2048, format: nil) { [weak self] buf, _ in
            guard let d = buf.floatChannelData, buf.format.channelCount > 1, buf.frameLength > 0 else { return }
            var l: Float = 0, r: Float = 0
            for i in 0..<Int(buf.frameLength) { l += d[0][i] * d[0][i]; r += d[1][i] * d[1][i] }
            guard let self else { return } // loudest buffer since the last status line; the thud is silent between hits
            self.level = (max(self.level.l, l / Float(buf.frameLength)), max(self.level.r, r / Float(buf.frameLength)))
        }
        try engine.start()
        for (p, buf) in players { p.stop(); p.scheduleBuffer(buf, at: nil, options: .loops); p.play() }
    }

    func pt(_ v: V3) -> AVAudio3DPoint { AVAudio3DPoint(x: Float(v.x), y: Float(v.y), z: Float(v.z)) }

    func setListener(_ l: (pos: V3, fwd: V3, up: V3)) {
        env.listenerPosition = pt(l.pos)
        env.listenerVectorOrientation = AVAudio3DVectorOrientation(forward: pt(l.fwd), up: pt(l.up))
    }

    /// Every second: start on headphones when they appear, stop when they go. Never plays on the speakers.
    func route(volume: Float) {
        let hp = headphones()
        // also restart when the engine has stopped under us (Bluetooth profile or format change)
        guard hp?.id != device || (hp != nil && !engine.isRunning) else { return status() }
        device = hp?.id
        engine.stop()
        guard let hp else { return note("audio: waiting for headphones") }
        var id = hp.id, frames: UInt32 = 128 // 128 frames is under 3 ms a buffer, where the device allows it
        var a = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyBufferFrameSize, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        if failures % 2 == 0 { AudioObjectSetPropertyData(hp.id, &a, 0, nil, 4, &frames) } // every other retry leaves the buffer alone
        AudioUnitSetProperty(engine.outputNode.audioUnit!, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &id, 4)
        engine.mainMixerNode.outputVolume = volume
        do {
            let rate = engine.outputNode.outputFormat(forBus: 0).sampleRate
            try start(rate: rate)
            _ = audioGet(hp.id, kAudioDevicePropertyBufferFrameSize, into: &frames)
            lead = engine.outputNode.presentationLatency + Double(frames) / rate
            note(String(format: "audio: playing on %@, %d-frame buffer, output delay %.0f ms%@", hp.name, frames, lead * 1000,
                        failures > 0 ? " (after \(failures) failed tries)" : ""))
            failures = 0
        } catch {
            // seen on Bluetooth buds right after they reconnect: CoreAudio error 'what'. Try again each second.
            if failures == 0 { note("audio: \(hp.name) refused to start (\(error.localizedDescription)); retrying every second") }
            failures += 1
            device = 0
        }
    }

    func feed(_ s: Sample) {
        if let offsets { for i in sourceAt.indices { sourceAt[i] = s.screenCm / 100 + offsets[i]; mixers[i].position = pt(sourceAt[i]) } }
        for (i, v) in [s.yaw, s.pitch, s.roll].enumerated() { angles[i].step(v, at: s.t) }
        for i in 0..<3 { place[i].step(s.posCm[i], at: s.t) }
        camLag += 0.1 * (CACurrentMediaTime() - s.t - camLag)
    }

    /// Once a second: where the first source sits relative to the nose, and the level that reached each ear.
    func status() {
        guard engine.isRunning, let l = listener, !sourceAt.isEmpty else { return }
        let bearing = bearing(of: centre, from: l).right
        func db(_ ms: Float) -> Double { 10 * log10(Double(max(ms, 1e-12))) }
        note(String(format: "source %3.0f deg %@ of your nose | left ear %.0f dB | right ear %.0f dB | camera delay %.0f ms%@",
                    abs(bearing), bearing < 0 ? "left " : "right", db(level.l), db(level.r), camLag * 1000, extraStatus?() ?? ""))
        trace(String(format: "{\"status\":true,\"bearing\":%.1f,\"gain\":%.1f,\"lag\":%.3f,\"left_db\":%.1f,\"right_db\":%.1f}", bearing, turnGain, camLag, db(level.l), db(level.r)))
        level = (0, 0)
    }

    func tick() {
        guard engine.isRunning, angles[0].started else { return }
        let now = CACurrentMediaTime(), a = angles.map { $0.ahead(now, lead: lead) }
        let pos = V3(place.map { $0.ahead(now, lead: lead) })
        var yaw = a[0]
        if !sourceAt.isEmpty { // turn strength scales the turn away from the sources, so facing them stays centred
            let d = centre * 100 - pos, toSource = atan2(d.x, -d.z) * deg
            yaw = toSource + turnGain * (yaw - toSource)
        }
        let l = listenerPose(yawDeg: yaw, pitchDeg: a[1], rollDeg: a[2], posCm: pos)
        listener = l
        setListener(l)
    }
}

// ---------- app audio capture (Core Audio process tap)

/// Stereo ring between the tap's IO thread (one writer) and the render thread (one reader), read out through
/// any number of taps (a channel, a delay, a gain): the two speakers plus their wall reflections.
/// The engine pulls each source node in uneven pieces (seen: 256 then 23 frames) in its own order. Every tap keeps its
/// own cursor into one shared history, and whichever runs past the end fetches more for all, so taps stay sample-aligned.
final class StereoFeed: @unchecked Sendable {
    struct Tap { var channel: Int, delay: Int, gain: Float }
    let cap: Int, rate: Double, taps: [Tap], channels: Int, cushion: Double
    let data: UnsafeMutablePointer<Float>
    let written = Atomic<Int>(0), peakBits = Atomic<UInt32>(0)
    var read = 0, primed = false, dropped = 0, starved = 0 // render thread only
    let hcap = 1 << 15
    let hist: [UnsafeMutablePointer<Float>]
    var produced = 0, cursor: [Int]
    var target: Int { Int(rate * cushion) } // cushion against the two device clocks drifting, or bursty writers

    /// cushion: 30 ms for the tap's steady trickle; the separator writes a second at a time and needs more.
    init(rate: Double, channels: Int = 2, taps: [Tap] = [Tap(channel: 0, delay: 0, gain: 1), Tap(channel: 1, delay: 0, gain: 1)], cushion: Double = 0.03) {
        precondition(taps.allSatisfy { $0.delay >= 0 && $0.delay < (1 << 14) && $0.channel < channels })
        (self.rate, self.channels, self.taps, self.cushion, cursor) = (rate, channels, taps, cushion, Array(repeating: 0, count: taps.count))
        cap = cushion > 0.1 ? 1 << 19 : 1 << 16 // room for 3 cushions plus a burst before the skip rule acts
        hist = (0..<channels).map { _ in UnsafeMutablePointer<Float>.allocate(capacity: 1 << 15) }
        data = .allocate(capacity: cap * channels)
        data.initialize(repeating: 0, count: cap * channels)
    }

    /// Writer: `stride` 2 for interleaved stereo, 1 for separate channel buffers.
    func write(left: UnsafePointer<Float>, right: UnsafePointer<Float>, stride: Int, frames: Int) {
        write([left, right], stride: stride, frames: frames)
    }

    func write(_ chans: [UnsafePointer<Float>], stride: Int, frames: Int) {
        var w = written.load(ordering: .relaxed), peak: Float = Float(bitPattern: peakBits.load(ordering: .relaxed))
        for i in 0..<frames {
            for (c, p) in chans.enumerated() { let v = p[i * stride]; data[(w % cap) * channels + c] = v; peak = max(peak, abs(v)) }
            w += 1
        }
        peakBits.store(peak.bitPattern, ordering: .relaxed)
        written.store(w, ordering: .releasing)
    }

    /// For a worker that reads the raw stream (not through taps): every frame written since the last call.
    func drain() -> [[Float]] {
        let w = written.load(ordering: .acquiring)
        if w - read > cap / 2 { read = w - cap / 2; dropped += 1 }
        let out = (0..<channels).map { c in (read..<w).map { data[($0 % cap) * channels + c] } }
        read = w
        return out
    }

    /// Appends n stereo frames to the history: captured audio, or silence while the cushion refills.
    func fetch(_ n: Int) {
        let w = written.load(ordering: .acquiring)
        if w - read > target * 3 { read = w - target; dropped += 1 } // reader fell behind: skip ahead
        if !primed { primed = w - read >= target }
        let take = primed ? min(n, w - read) : 0
        if primed && take < n { primed = false; starved += 1 }
        for c in 0..<channels {
            for i in 0..<n { hist[c][(produced + i) % hcap] = i < take ? data[((read + i) % cap) * channels + c] : 0 }
        }
        produced += n
        read += take
    }

    func node(_ tap: Int, format: AVAudioFormat) -> AVAudioSourceNode {
        AVAudioSourceNode(format: format) { [self] flags, ts, frames, abl in render(tap, flags, ts, frames, abl) }
    }

    func render(_ j: Int, _: UnsafeMutablePointer<ObjCBool>, _: UnsafePointer<AudioTimeStamp>, _ frames: AVAudioFrameCount,
                _ abl: UnsafeMutablePointer<AudioBufferList>) -> OSStatus {
        let n = min(Int(frames), 8192), t = taps[j]
        cursor[j] = max(cursor[j], produced - hcap / 2) // a tap nobody pulled for a while rejoins near the front
        if cursor[j] + n > produced { fetch(cursor[j] + n - produced) }
        let out = UnsafeMutableAudioBufferListPointer(abl)[0].mData!.assumingMemoryBound(to: Float.self)
        for i in 0..<n {
            let k = cursor[j] + i - t.delay
            out[i] = k >= 0 ? t.gain * hist[t.channel][k % hcap] : 0
        }
        cursor[j] += n
        return noErr
    }

    /// Loudest input sample since the last call, in dBFS.
    func takePeak() -> Double { 20 * log10(Double(max(Float(bitPattern: peakBits.exchange(0, ordering: .relaxed)), 1e-6))) }
}

/// Music scene, in the output frame (metres, camera at the origin) for a typical seat: ears 36 cm above and 58 cm in
/// front of the camera. A stereo pair 2.3 m away behind the laptop at +-27 degrees, in a 5 x 2.75 x 5.8 m room.
/// Each speaker and each of its six first reflections (image-source method) is its own HRTF source, delayed by its
/// extra path length; walls, floor and ceiling reflect 90, 66 and 84 percent of the pressure. Room settings were
/// picked by measurement: ear similarity 0.68 (real rooms 0.3-0.7) with left/right still >= 5.9 dB apart.
let nominalEars = V3(0, 0.36, 0.58), nominalScreen = V3(0, -0.07, 0.06)
struct MusicSource { var tap: StereoFeed.Tap; var at: V3; var reflection: Bool }
func musicScene(rate: Double, room: Bool) -> [MusicSource] {
    let speakers = [V3(-1.15, -0.08, -1.45), V3(1.15, -0.08, -1.45)]
    let surfaces: [(axis: Int, at: Double, reflects: Float)] = [(0, -2.5, 0.9), (0, 2.5, 0.9), (1, -1.05, 0.66), (1, 1.7, 0.84), (2, -3.2, 0.9), (2, 2.6, 0.9)]
    var out = speakers.enumerated().map { MusicSource(tap: .init(channel: $0, delay: 0, gain: makeup), at: $1, reflection: false) }
    guard room else { return out }
    for (c, s) in speakers.enumerated() {
        for w in surfaces {
            var image = s
            image[w.axis] = 2 * w.at - s[w.axis]
            let extra = (simd_length(image - nominalEars) - simd_length(s - nominalEars)) / 343
            out.append(MusicSource(tap: .init(channel: c, delay: Int(extra * rate), gain: w.reflects * makeup), at: image, reflection: true))
        }
    }
    return out
}

/// Live instrument separation with HTDemucs (Meta, MIT; CoreML conversion: 7.8 s stereo window at 44.1 kHz in,
/// drums, bass, other, vocals out as stereo pairs). Every `hop` it separates the newest 7.8 s and keeps only the stretch
/// that ends `look` before the newest sample. Measured on a mix of four Apple loops with known stems: 0.25 s of
/// look-ahead gives the same stems as a centred window to within 23-28 dB, and a window took 0.27 s on this Mac's GPU.
/// The music therefore runs about hop + look + compute + cushion behind the app (about 2.5 s); head tracking does not.
/// Output channels: mix L, mix R (stems summed, for the room), vocals, drums, bass (mono), other L, other R.
final class Separator: @unchecked Sendable {
    static let window = 343980, rate = 44100.0
    let model: MLModel, hop = 44100, look = 11025, fade = 882 // 1 s, 0.25 s, 20 ms crossfade
    let capture: StereoFeed, output: StereoFeed
    var history: [[Float]] = [[], []], historyStart = 0, nextEnd = 0, prevTail: [[Float]]? // worker thread only
    var converter: AVAudioConverter?
    let runs = Atomic<Int>(0), busyMs = Atomic<Int>(0), skipped = Atomic<Int>(0)

    static func loadModel() -> MLModel? {
        let dir = calibURL.deletingLastPathComponent().appendingPathComponent("models")
        let compiled = dir.appendingPathComponent("htdemucs.mlmodelc"), pkg = dir.appendingPathComponent("htdemucs.mlpackage")
        if !FileManager.default.fileExists(atPath: compiled.path) {
            guard FileManager.default.fileExists(atPath: pkg.path), let c = try? MLModel.compileModel(at: pkg) else { return nil }
            try? FileManager.default.moveItem(at: c, to: compiled)
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all // GPU: 0.27 s a window here, CPU alone 0.73 s
        return try? MLModel(contentsOf: compiled, configuration: cfg)
    }

    init(model: MLModel, captureRate: Double, taps: [StereoFeed.Tap]) {
        self.model = model
        capture = StereoFeed(rate: captureRate, cushion: 0)
        output = StereoFeed(rate: Separator.rate, channels: 7, taps: taps, cushion: 1.35) // > hop + compute jitter
        if captureRate != Separator.rate {
            converter = AVAudioConverter(from: AVAudioFormat(standardFormatWithSampleRate: captureRate, channels: 2)!,
                                         to: AVAudioFormat(standardFormatWithSampleRate: Separator.rate, channels: 2)!)
            converter?.sampleRateConverterQuality = AVAudioQuality.high.rawValue
        }
    }

    func start() {
        let t = Thread { [self] in
            while true {
                take(capture.drain())
                if !step() { Thread.sleep(forTimeInterval: 0.02) }
            }
        }
        t.qualityOfService = .userInitiated
        t.start()
    }

    /// Appends captured audio (any rate) to the 44.1 kHz history.
    func take(_ chans: [[Float]]) {
        guard let n = chans.first?.count, n > 0 else { return }
        guard let conv = converter else { for c in 0..<2 { history[c] += chans[c] }; return }
        let inB = AVAudioPCMBuffer(pcmFormat: conv.inputFormat, frameCapacity: AVAudioFrameCount(n))!
        inB.frameLength = AVAudioFrameCount(n)
        for c in 0..<2 { chans[c].withUnsafeBufferPointer { inB.floatChannelData![c].update(from: $0.baseAddress!, count: n) } }
        let outB = AVAudioPCMBuffer(pcmFormat: conv.outputFormat, frameCapacity: AVAudioFrameCount(Double(n) * Separator.rate / conv.inputFormat.sampleRate) + 64)!
        var given = false
        _ = conv.convert(to: outB, error: nil) { _, st in
            if given { st.pointee = .noDataNow; return nil }
            given = true; st.pointee = .haveData; return inB
        }
        for c in 0..<2 { history[c] += UnsafeBufferPointer(start: outB.floatChannelData![c], count: Int(outB.frameLength)) }
    }

    /// Separates one hop if enough new audio has arrived. Returns false when there was nothing to do.
    @discardableResult func step() -> Bool {
        let have = historyStart + history[0].count
        if nextEnd == 0 { nextEnd = hop }
        while have - nextEnd > 3 * hop { nextEnd += hop; prevTail = nil; skipped.add(1, ordering: .relaxed) } // fell behind: drop a second
        guard have >= nextEnd else { return false }
        let w = Separator.window, e = nextEnd
        let mix = try! MLMultiArray(shape: [1, 2, NSNumber(value: w)], dataType: .float32)
        let mp = mix.dataPointer.assumingMemoryBound(to: Float.self)
        for c in 0..<2 { for i in 0..<w { let k = e - w + i - historyStart; mp[c * w + i] = k >= 0 && k < history[c].count ? history[c][k] : 0 } }
        let t0 = CACurrentMediaTime()
        guard let res = try? model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["mix": MLFeatureValue(multiArray: mix)])),
              let src = res.featureValue(for: "sources")?.multiArrayValue else { nextEnd += hop; return true }
        busyMs.add(Int((CACurrentMediaTime() - t0) * 1000), ordering: .relaxed)
        runs.add(1, ordering: .relaxed)
        let st = src.strides.map(\.intValue), sp = src.dataPointer.assumingMemoryBound(to: Float.self)
        func at(_ ch: Int, _ i: Int) -> Float { sp[ch * st[1] + i * st[2]] } // ch: drums 0-1, bass 2-3, other 4-5, vocals 6-7
        let from = w - look - hop
        var out = (0..<7).map { _ in [Float](repeating: 0, count: hop + fade) }
        for i in 0..<(hop + fade) {
            let j = from + i
            let l = at(0, j) + at(2, j) + at(4, j) + at(6, j), r = at(1, j) + at(3, j) + at(5, j) + at(7, j)
            let v: [Float] = [l, r, (at(6, j) + at(7, j)) / 2, (at(0, j) + at(1, j)) / 2, (at(2, j) + at(3, j)) / 2, at(4, j), at(5, j)]
            for c in 0..<7 { out[c][i] = v[c] }
        }
        if let prev = prevTail { // crossfade into the previous segment's look-ahead tail
            for c in 0..<7 { for i in 0..<fade { let g = (Float(i) + 0.5) / Float(fade); out[c][i] = prev[c][i] * (1 - g) + out[c][i] * g } }
        }
        prevTail = out.map { Array($0[hop...]) }
        let ptrs = out.map { a -> UnsafeMutablePointer<Float> in let p = UnsafeMutablePointer<Float>.allocate(capacity: hop); a.withUnsafeBufferPointer { p.update(from: $0.baseAddress!, count: hop) }; return p }
        output.write(ptrs.map { UnsafePointer($0) }, stride: 1, frames: hop)
        ptrs.forEach { $0.deallocate() }
        nextEnd += hop
        let keep = nextEnd - w - hop // drop history no window will need again
        if keep > historyStart { for c in 0..<2 { history[c].removeFirst(keep - historyStart) }; historyStart = keep }
        return true
    }
}

/// Instruments around the room, from the typical seat; angles from straight ahead, + to the right.
/// Vocals centre and a little raised, drums left and further back, bass right and low, other spread wide.
/// The room's reflections come from the original stereo pair (stems summed), which keeps the source count down.
func stemScene(rate: Double, room: Bool) -> [MusicSource] {
    func place(_ azDeg: Double, _ dist: Double, _ y: Double) -> V3 {
        let a = azDeg / deg, h = (dist * dist - pow(y - nominalEars.y, 2)).squareRoot()
        return V3(nominalEars.x + h * sin(a), y, nominalEars.z - h * cos(a))
    }
    let direct: [(ch: Int, at: V3)] = [(2, place(0, 1.9, 0.2)), (3, place(-25, 2.6, -0.05)), (4, place(25, 2.3, -0.3)), (5, place(-50, 2.3, 0.0)), (6, place(50, 2.3, 0.0))]
    let pair = musicScene(rate: rate, room: room)
    return direct.map { MusicSource(tap: .init(channel: $0.ch, delay: 0, gain: makeup), at: $0.at, reflection: false) } + pair.filter(\.reflection)
}

/// Builds the music engine graph for a feed. Shared by the live mode and the selftest so both measure the same thing.
func musicSpatial(_ feed: StereoFeed, scene: [MusicSource]) -> Spatial {
    let mono = AVAudioFormat(standardFormatWithSampleRate: feed.rate, channels: 1)!
    let s = Spatial([], live: scene.indices.map { (feed.node($0, format: mono), mono, scene[$0].at) })
    s.primary = scene.filter { !$0.reflection }.count
    s.offsets = scene.map { $0.at - nominalScreen }
    s.env.distanceAttenuationParameters.referenceDistance = 1 // 1/distance beyond 1 m, as in air
    let room = scene.contains { $0.reflection }
    s.env.reverbParameters.enable = room
    if room {
        s.env.reverbParameters.loadFactoryReverbPreset(.mediumRoom) // late tail only; the early sound is modelled above
        s.env.reverbParameters.level = roomTailDb
    }
    for (i, src) in scene.enumerated() {
        s.mixers[i].reverbBlend = room ? roomTailBlend : 0
        if src.reflection { s.mixers[i].occlusion = reflectionDullDb } // walls absorb highs more than lows
    }
    return s
}
var roomTailDb: Float = -6, roomTailBlend: Float = 0.25, reflectionDullDb: Float = -4
let makeup: Float = 1.58 // +4 dB: 2.3 m of distance and the room cost 4.5 dB against the input (measured)

func audioProcesses(matching app: String) -> [AudioObjectID] {
    let sys = AudioObjectID(kAudioObjectSystemObject)
    var a = AudioObjectPropertyAddress(mSelector: kAudioHardwarePropertyProcessObjectList, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
    var n: UInt32 = 0
    AudioObjectGetPropertyDataSize(sys, &a, 0, nil, &n)
    var ids = [AudioObjectID](repeating: 0, count: Int(n) / 4)
    AudioObjectGetPropertyData(sys, &a, 0, nil, &n, &ids)
    return ids.filter { id in
        var cf: Unmanaged<CFString>?, pid: pid_t = 0
        _ = audioGet(id, kAudioProcessPropertyBundleID, into: &cf)
        _ = audioGet(id, kAudioProcessPropertyPID, into: &pid)
        var path = [CChar](repeating: 0, count: 4096)
        proc_pidpath(pid, &path, 4096)
        return ((cf?.takeRetainedValue() as String?) ?? "").lowercased().contains(app) || String(cString: path).lowercased().contains(app)
    }
}

/// Taps the app's audio (its own output muted while we read it) and streams it into `feed`. The tap and its private
/// aggregate device go away with this process. Returns the tap's sample rate.
// shortcut: the tap is bound to the app's processes at start; if the app quits and reopens, restart headpose.
var tapCleanup: (() -> Void)?
func tapApp(_ processes: [AudioObjectID]) -> (rate: Double, start: (StereoFeed) -> Void)? {
    let desc = CATapDescription(stereoMixdownOfProcesses: processes)
    desc.uuid = UUID()
    desc.name = "headpose"
    desc.isPrivate = true
    desc.muteBehavior = .mutedWhenTapped
    var tap = AudioObjectID(kAudioObjectUnknown)
    guard AudioHardwareCreateProcessTap(desc, &tap) == noErr else { return nil }
    var asbd = AudioStreamBasicDescription()
    guard audioGet(tap, kAudioTapPropertyFormat, into: &asbd), asbd.mSampleRate > 0 else { AudioHardwareDestroyProcessTap(tap); return nil }
    let spec: [String: Any] = [
        kAudioAggregateDeviceNameKey: "headpose tap", kAudioAggregateDeviceUIDKey: UUID().uuidString,
        kAudioAggregateDeviceIsPrivateKey: true, kAudioAggregateDeviceIsStackedKey: false, kAudioAggregateDeviceTapAutoStartKey: true,
        kAudioAggregateDeviceTapListKey: [[kAudioSubTapUIDKey: desc.uuid.uuidString, kAudioSubTapDriftCompensationKey: true]],
    ]
    var agg = AudioObjectID(kAudioObjectUnknown)
    guard AudioHardwareCreateAggregateDevice(spec as CFDictionary, &agg) == noErr else { AudioHardwareDestroyProcessTap(tap); return nil }
    return (asbd.mSampleRate, { feed in
        var proc: AudioDeviceIOProcID?
        AudioDeviceCreateIOProcIDWithBlock(&proc, agg, nil) { _, input, _, _, _ in
            let abl = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: input))
            guard let first = abl.first, let p0 = first.mData?.assumingMemoryBound(to: Float.self) else { return }
            if abl.count >= 2, let p1 = abl[1].mData?.assumingMemoryBound(to: Float.self) { // one buffer per channel
                feed.write(left: p0, right: p1, stride: 1, frames: Int(first.mDataByteSize) / 4)
            } else { // interleaved (stereo) or a single channel
                let ch = Int(max(first.mNumberChannels, 1))
                feed.write(left: p0, right: p0 + (ch > 1 ? 1 : 0), stride: ch, frames: Int(first.mDataByteSize) / 4 / ch)
            }
        }
        AudioDeviceStart(agg, proc)
        tapCleanup = {
            AudioDeviceStop(agg, proc); AudioDeviceDestroyIOProcID(agg, proc!)
            AudioHardwareDestroyAggregateDevice(agg); AudioHardwareDestroyProcessTap(tap)
        }
        atexit { tapCleanup?() }
    })
}

// ---------- camera + face

struct Calib: Codable { var kPerWidth: Double; var ipdMm: Double }
/// mm from the camera down the lid to the centre of the built-in display (the camera sits at the panel's top edge)
let screenDrop = CGDisplayIsBuiltin(CGMainDisplayID()) != 0 ? Double(CGDisplayScreenSize(CGMainDisplayID()).height) / 2 : 98
let calibURL = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().appendingPathComponent("calib.json")

// Vision reports yaw/pitch of the face as it appears, i.e. relative to the ray from the camera, so the ray's
// own azimuth/elevation is subtracted (a small-angle sum). Camera frame -> gravity frame is an exact rotation.
// yawSign is checked live against the nose position and flipped if it disagrees (see signVotes).
// shortcut: the ray-relative reading is unverified against a measured head pose.
var yawSign = -1.0

/// Live view: head direction (thick arrow), true vertical (thin line), eye points, one readout. Mirrored like the feed.
final class Overlay: NSView {
    var pts: [CGPoint] = [] // image px, unmirrored: eye, eye, head, arrow tip, vertical tip
    var text = "", imageW = 1280.0

    func show(_ new: [CGPoint], _ t: String, w: Double) {
        // halve the frame-to-frame jump so the arrow does not buzz
        pts = pts.count == new.count ? zip(pts, new).map { CGPoint(x: ($0.x + $1.x) / 2, y: ($0.y + $1.y) / 2) } : new
        (text, imageW, needsDisplay) = (t, w, true)
    }

    override func draw(_: NSRect) {
        guard pts.count == 5 else { return }
        let s = bounds.width / imageW
        let p = pts.map { CGPoint(x: (imageW - $0.x) * s, y: $0.y * s) }
        NSGraphicsContext.saveGraphicsState()
        let shadow = NSShadow()
        shadow.shadowBlurRadius = 5
        shadow.shadowColor = NSColor.black.withAlphaComponent(0.7)
        shadow.set()
        func line(_ a: CGPoint, _ b: CGPoint, _ width: CGFloat) {
            let l = NSBezierPath(); l.move(to: a); l.line(to: b); l.lineWidth = width; l.lineCapStyle = .round; l.stroke()
        }
        NSColor.white.withAlphaComponent(0.55).set()
        line(p[2], p[4], 2)
        NSColor.white.set()
        for e in p[0...1] { NSBezierPath(ovalIn: NSRect(x: e.x - 3, y: e.y - 3, width: 6, height: 6)).fill() }
        line(p[2], p[3], 6)
        let d = CGPoint(x: p[3].x - p[2].x, y: p[3].y - p[2].y), len = max(hypot(d.x, d.y), 1)
        let u = CGPoint(x: d.x / len, y: d.y / len), k = min(len / 30, 1) // head shrinks when the arrow points at the lens
        let tip = NSBezierPath()
        tip.move(to: CGPoint(x: p[3].x + u.x * 20 * k, y: p[3].y + u.y * 20 * k))
        tip.line(to: CGPoint(x: p[3].x - u.y * 12 * k, y: p[3].y + u.x * 12 * k))
        tip.line(to: CGPoint(x: p[3].x + u.y * 12 * k, y: p[3].y - u.x * 12 * k))
        tip.close(); tip.fill()
        NSGraphicsContext.restoreGraphicsState()

        let str = NSAttributedString(string: text, attributes: [
            .font: NSFont.monospacedDigitSystemFont(ofSize: 22, weight: .semibold), .foregroundColor: NSColor.white])
        let box = NSRect(x: 20, y: 20, width: str.size().width + 40, height: str.size().height + 18)
        NSColor.black.withAlphaComponent(0.5).set()
        NSBezierPath(roundedRect: box, xRadius: box.height / 2, yRadius: box.height / 2).fill()
        str.draw(at: CGPoint(x: box.minX + 20, y: box.minY + 9))
    }
}

struct Sample { var t: Double, posCm: V3, yaw: Double, pitch: Double, roll: Double, screenCm: V3, visionYaw: Double, visionPitch: Double, eyeSlant: Double }

final class Tracker: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let lid: IOHIDDevice
    let calib: Calib?
    let calibrate: (rangeMm: Double, ipdMm: Double)?
    var last = 0.0
    var overlay: Overlay?
    var heldDepth: Double?, lastRoll = 0.0
    var json = true
    var onPose: ((Sample) -> Void)?, onMiss: (() -> Void)?
    var traced = 0
    var signVotes = 0.0
    var samples: [(eye: Double, off: Double, w: Double)] = []

    init(lid: IOHIDDevice, calib: Calib?, calibrate: (Double, Double)?) {
        (self.lid, self.calib, self.calibrate) = (lid, calib, calibrate)
    }

    func captureOutput(_: AVCaptureOutput, didOutput sb: CMSampleBuffer, from _: AVCaptureConnection) {
        let t = CMSampleBufferGetPresentationTimeStamp(sb).seconds
        guard t - last >= (overlay == nil && onPose == nil ? 1.0 / 15 : 0), let pb = CMSampleBufferGetImageBuffer(sb) else { return }
        last = t
        let handler = VNImageRequestHandler(cvPixelBuffer: pb)
        let rects = VNDetectFaceRectanglesRequest(), marks = VNDetectFaceLandmarksRequest()
        try? handler.perform([rects])
        guard let face = rects.results?.max(by: { $0.boundingBox.width < $1.boundingBox.width }) else { return noFace(t) }
        marks.inputFaceObservations = [face]
        try? handler.perform([marks])
        guard let lm = marks.results?.first?.landmarks, let e1 = lm.leftEye, let e2 = lm.rightEye,
              let lidDeg = lidAngle(lid) else { return noFace(t) }

        let w = Double(CVPixelBufferGetWidth(pb)), h = Double(CVPixelBufferGetHeight(pb))
        // eye-outline centres, not pupils: they do not move with gaze
        func centre(_ r: VNFaceLandmarkRegion2D) -> CGPoint {
            let pts = r.pointsInImage(imageSize: CGSize(width: w, height: h))
            return CGPoint(x: pts.map(\.x).reduce(0, +) / CGFloat(pts.count), y: pts.map(\.y).reduce(0, +) / CGFloat(pts.count))
        }
        let eyes = [e1, e2].map(centre).sorted { $0.x < $1.x }
        let (a, b) = (eyes[0], eyes[1]) // a = image-left = your right eye
        let eyePx = hypot(b.x - a.x, b.y - a.y), mid = CGPoint(x: (a.x + b.x) / 2, y: (a.y + b.y) / 2)
        let vYaw = face.yaw?.doubleValue ?? 0, vPitch = face.pitch?.doubleValue ?? 0 // pitch: + = nodding down (Vision header)

        if let nose = lm.nose, abs(vYaw) > 0.17, abs(signVotes) < 1 {
            // the nose sits in front of the eyes, so in the image it shifts toward the side the head turns to (image-left = your right)
            let n = centre(nose), off = ((n.x - mid.x) * (b.x - a.x) + (n.y - mid.y) * (b.y - a.y)) / (eyePx * eyePx)
            signVotes += yawSign * vYaw * -off
            if signVotes <= -1 { yawSign = -yawSign; signVotes = 0; note("yaw sign was backwards against the nose position: flipped") }
            else if signVotes >= 1 { note("yaw sign confirmed against the nose position") }
        }

        if let c = calibrate {
            guard abs(vYaw) < 0.15 else { return }
            samples.append((eyePx / cos(vYaw), hypot(mid.x - w / 2, mid.y - h / 2), w))
            guard samples.count == 45 else { return }
            let eye = samples.map(\.eye).sorted()[22], off = samples.map(\.off).sorted()[22]
            let k = solveK(eyePx: eye, offAxisPx: off, rangeMm: c.rangeMm, ipdMm: c.ipdMm)
            try! JSONEncoder().encode(Calib(kPerWidth: k / w, ipdMm: c.ipdMm)).write(to: calibURL)
            let fov = 2 * atan(w / 2 / (k / c.ipdMm)) * deg
            FileHandle.standardError.write("calibrated: wrote \(calibURL.path); implied horizontal FOV \(String(format: "%.1f", fov)) deg\n".data(using: .utf8)!)
            exit(0)
        }

        // shortcut: uncalibrated falls back to a guessed 65 degree horizontal FOV and the 63 mm adult mean; run calibrate.
        let ipd = calib?.ipdMm ?? 63
        let f = calib.map { $0.kPerWidth * w / $0.ipdMm } ?? w / 2 / tan(32.5 / deg)
        let up = upInCamera(accel: gravity, lidDeg: lidDeg)
        let est = estimatePose(mid: mid, eyePx: eyePx, boxPx: face.boundingBox.height * h, w: w, h: h, vYaw: vYaw, vPitch: vPitch,
                               f: f, ipdMm: ipd, up: up, yawSign: yawSign, held: heldDepth)
        heldDepth = est.held
        let (pc, fwd, p, yaw, pitch) = (est.pc, est.fwd, est.posCm, est.yaw, est.pitch)
        let az = atan2(p.x, p.z), el = atan2(p.y, hypot(p.x, p.z)), camRoll = atan2(up.x, up.y)
        let rawRoll = worldRoll(eyeX: Double(b.x - a.x), eyeY: Double(a.y - b.y), fwd: fwd, up: up) * deg
        let tr = trustedRoll(raw: rawRoll, yawDeg: vYaw * deg, last: lastRoll)
        let roll = tr.roll
        lastRoll = tr.last
        if json { print(String(format: "{\"t\":%.3f,\"face\":true,\"dist_cm\":%.1f,\"x_cm\":%.1f,\"y_cm\":%.1f,\"z_cm\":%.1f,\"az_deg\":%.1f,\"el_deg\":%.1f,"
                + "\"yaw_deg\":%.1f,\"pitch_deg\":%.1f,\"roll_deg\":%.1f,\"lid_deg\":%.0f,\"cam_elev_deg\":%.1f,\"cam_roll_deg\":%.1f,"
                + "\"accel\":%@,\"calibrated\":%@}",
            t, simd_length(p), p.x, p.y, p.z, az * deg, el * deg, yaw, pitch, roll,
            lidDeg, asin(up.z) * deg, camRoll * deg, accelSeen ? "true" : "false", calib != nil ? "true" : "false")) }
        if traced < 20000 {
            traced += 1
            let box = face.boundingBox
            trace(String(format: "{\"t\":%.3f,\"vy\":%.1f,\"vp\":%.1f,\"slant\":%.1f,\"eyePx\":%.1f,\"bw\":%.0f,\"bh\":%.0f,\"mx\":%.0f,\"my\":%.0f,"
                    + "\"yaw\":%.1f,\"pitch\":%.1f,\"roll\":%.1f,\"x\":%.1f,\"y\":%.1f,\"z\":%.1f,\"lid\":%.0f,\"elev\":%.1f}",
                t, vYaw * deg, vPitch * deg, atan2(Double(b.y - a.y), Double(b.x - a.x)) * deg, Double(eyePx), box.width * w, box.height * h,
                mid.x, mid.y, yaw, pitch, roll, p.x, p.y, p.z, lidDeg, asin(up.z) * deg))
        }
        onPose?(Sample(t: t, posCm: p, yaw: yaw, pitch: pitch, roll: roll, screenCm: toWorld(V3(0, -screenDrop, 0), up: up) / 10,
                       visionYaw: vYaw * deg, visionPitch: vPitch * deg, eyeSlant: atan2(Double(b.y - a.y), Double(b.x - a.x)) * deg))
        if let overlay {
            func px(_ q: V3) -> CGPoint { CGPoint(x: w / 2 - f * q.x / q.z, y: h / 2 + f * q.y / q.z) }
            overlay.show([a, b, px(pc), px(pc + 150 * fwd), px(pc + 80 * up)],
                String(format: "%@%.0f cm   yaw %.0f°   pitch %.0f°   roll %.0f°", calib == nil ? "~" : "",
                       simd_length(p), yaw.rounded() + 0, pitch.rounded() + 0, roll.rounded() + 0), w: w)
        }
    }

    func noFace(_ t: Double) {
        onMiss?()
        if traced < 20000 { traced += 1; trace(String(format: "{\"t\":%.3f,\"face\":false}", t)) }
        overlay?.show([], "", w: 1280)
        if calibrate == nil, json { print(String(format: "{\"t\":%.3f,\"face\":false}", t)) }
    }
}

/// `check`: five held head poses, 2 s recorded each. Shows what the tracker measured, so a wrong angle can be traced to its source.
final class PoseCheck {
    let poses = ["face the screen", "TURN your head RIGHT about 45 degrees (nose toward right shoulder)", "TURN your head LEFT about 45 degrees",
                 "TILT your right ear toward your right shoulder", "TILT your left ear toward your left shoulder"]
    var step = -1, recording = false, seen: [Sample] = [], missed = 0, rows: [[String: Any]] = []

    func next() {
        step += 1
        guard step < poses.count else {
            try? JSONSerialization.data(withJSONObject: rows, options: [.prettyPrinted]).write(to: calibURL.deletingLastPathComponent().appendingPathComponent("check.json"))
            note("\nsaved. Tell Claude it is done."); exit(0)
        }
        note("\n\(step + 1)/\(poses.count)  \(poses[step]) and hold...")
        DispatchQueue.main.asyncAfter(deadline: .now() + 3) { (self.seen, self.missed, self.recording) = ([], 0, true) }
        DispatchQueue.main.asyncAfter(deadline: .now() + 5) { self.recording = false; self.report(); self.next() }
    }

    func report() {
        func mid(_ f: (Sample) -> Double) -> Double { seen.isEmpty ? .nan : seen.map(f).sorted()[seen.count / 2] }
        let found = 100 * Double(seen.count) / Double(max(seen.count + missed, 1))
        let at = mid { bearing(of: $0.screenCm / 100, from: listenerPose(yawDeg: $0.yaw, pitchDeg: $0.pitch, rollDeg: $0.roll, posCm: $0.posCm)).right }
        note(String(format: "     face found %.0f%% | camera sees: turn %.0f, nod %.0f, eye-line slant %.0f | in the room: yaw %.0f, pitch %.0f, roll %.0f | screen is %.0f deg %@ of your nose",
                    found, mid(\.visionYaw), mid(\.visionPitch), mid(\.eyeSlant), mid(\.yaw), mid(\.pitch), mid(\.roll), abs(at), at < 0 ? "left" : "right"))
        rows.append(["pose": poses[step], "face_found_pct": found, "frames": seen.count, "vision_yaw": mid(\.visionYaw), "vision_pitch": mid(\.visionPitch),
                     "eye_slant": mid(\.eyeSlant), "yaw": mid(\.yaw), "pitch": mid(\.pitch), "roll": mid(\.roll), "screen_right_of_nose": at,
                     "dist_cm": mid { simd_length($0.posCm) }].mapValues { ($0 as? Double)?.isNaN == true ? NSNull() : $0 })
    }
}

// ---------- main

setvbuf(stdout, nil, _IOLBF, 0)
let args = CommandLine.arguments
if args.count > 1, args[1] == "selftest" { selftest(); exit(0) }
func die(_ m: String) -> Never { FileHandle.standardError.write((m + "\n").data(using: .utf8)!); exit(1) }

var calibrate: (Double, Double)?
let wantView = args.contains("view"), wantMusic = args.contains("music")
let wantAudio = args.contains("audio") || wantMusic, wantCheck = args.count > 1 && args[1] == "check"
if args.count > 1, !wantView, !wantAudio, !wantCheck {
    guard args[1] == "calibrate", args.count > 2, let cm = Double(args[2]), cm > 15, cm < 200 else {
        die("usage: headpose [view] [audio [file[@x,y,z] ...]] | headpose check | headpose calibrate <cm from camera to eyes> [ipd_mm] | headpose selftest")
    }
    calibrate = (cm * 10, args.count > 3 ? Double(args[3]) ?? 63 : 63)
}
guard let lid = hidDevice(0x20, 0x8A, "las"), lidAngle(lid) != nil else { die("no lid angle sensor on this Mac") }
startAccel()

let sem = DispatchSemaphore(value: 0)
AVCaptureDevice.requestAccess(for: .video) { ok in if !ok { die("camera access denied for this terminal (System Settings > Privacy > Camera)") }; sem.signal() }
sem.wait()
let cams = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: .video, position: .unspecified).devices
guard let cam = cams.first(where: { $0.localizedName.contains("MacBook") }) ?? cams.first,
      let input = try? AVCaptureDeviceInput(device: cam) else { die("no built-in camera") }
// Center Stage re-crops the frame as you move, which would change the focal length under us
AVCaptureDevice.centerStageControlMode = .app
AVCaptureDevice.isCenterStageEnabled = false

let session = AVCaptureSession()
session.sessionPreset = .hd1280x720
let output = AVCaptureVideoDataOutput()
output.alwaysDiscardsLateVideoFrames = true
let calib = try? JSONDecoder().decode(Calib.self, from: Data(contentsOf: calibURL))
let tracker = Tracker(lid: lid, calib: calib, calibrate: calibrate)
tracker.json = !wantCheck && (!(wantView || wantAudio) || isatty(1) == 0)
output.setSampleBufferDelegate(tracker, queue: .main) // one thread: sensors and frames both run on main
session.addInput(input)
session.addOutput(output)
if let c = output.connection(with: .video), c.isVideoMirroringSupported {
    c.automaticallyAdjustsVideoMirroring = false
    c.isVideoMirrored = false
}
session.startRunning()
if calibrate != nil { FileHandle.standardError.write("hold still, face the camera...\n".data(using: .utf8)!) }
else if calib == nil { FileHandle.standardError.write("uncalibrated: distances are rough until you run: headpose calibrate <cm>\n".data(using: .utf8)!) }
var timers: [DispatchSourceTimer] = []
var keySource: DispatchSourceRead?
if wantView || wantAudio || wantCheck {
    let path = calibURL.deletingLastPathComponent().appendingPathComponent("trace.jsonl").path
    FileManager.default.createFile(atPath: path, contents: nil)
    traceFile = FileHandle(forWritingAtPath: path)
}
if wantAudio {
    keySource = watchKeys()
    if keySource != nil { note("keys: [ weaker, ] stronger response to head turns") }
}
let poseCheck = PoseCheck()
if wantCheck {
    tracker.onPose = { if poseCheck.recording { poseCheck.seen.append($0) } }
    tracker.onMiss = { if poseCheck.recording { poseCheck.missed += 1 } }
    poseCheck.next()
}
if wantAudio {
    let rest = args.dropFirst().filter { $0 != "view" && $0 != "audio" && $0 != "music" }
    let files = wantMusic ? [] : rest
    var feed: StereoFeed?, scene: [MusicSource] = [], separator: Separator?
    if wantMusic {
        let app = (rest.first ?? "spotify").lowercased()
        var procs = audioProcesses(matching: app)
        if procs.isEmpty { note("music: waiting for \(app) to play something...") }
        while procs.isEmpty { sleep(1); procs = audioProcesses(matching: app) }
        guard let tap = tapApp(procs) else { die("music: macOS refused to capture \(app)") }
        let room = ProcessInfo.processInfo.environment["HEADPOSE_ROOM"] != "0"
        if ProcessInfo.processInfo.environment["HEADPOSE_STEMS"] != "0", let model = Separator.loadModel() {
            scene = stemScene(rate: Separator.rate, room: room)
            let sep = Separator(model: model, captureRate: tap.rate, taps: scene.map(\.tap))
            tap.start(sep.capture)
            sep.start()
            separator = sep
            note("music: splitting into vocals, drums, bass and other; the music runs about 2.5 s behind the app")
        } else {
            scene = musicScene(rate: tap.rate, room: room)
            let f = StereoFeed(rate: tap.rate, taps: scene.map(\.tap))
            tap.start(f)
            feed = f
        }
        feed = feed ?? separator?.output
        note("music: capturing \(app) (\(procs.count) process\(procs.count == 1 ? "" : "es"), \(Int(tap.rate)) Hz); its own output is muted while this runs")
    }
    let sources = wantMusic ? [] : files.isEmpty ? [(thud(), V3(0, 0, 0))] : files.map { arg -> (AVAudioPCMBuffer, V3) in
        let parts = arg.split(separator: "@", maxSplits: 1).map(String.init)
        let xyz = parts.count > 1 ? parts[1].split(separator: ",").compactMap { Double($0) } : [0, 0, 0]
        guard xyz.count == 3, let buf = loadMono(parts[0]) else { die("cannot read \(arg): want an audio file, or file@x,y,z in metres") }
        return (buf, V3(xyz[0], xyz[1], xyz[2]))
    }
    let spatial = feed.map { musicSpatial($0, scene: scene) } ?? Spatial(sources), q = DispatchQueue(label: "headpose.audio", qos: .userInteractive)
    let volume = Float(ProcessInfo.processInfo.environment["HEADPOSE_VOL"] ?? "") ?? (wantMusic ? 1 : 0.6)
    if !wantMusic { spatial.offsets = files.isEmpty ? [.zero] : nil }
    if let feed {
        var silentFor = 0
        spatial.extraStatus = {
            let peak = feed.takePeak()
            silentFor = peak < -100 ? silentFor + 1 : 0
            if silentFor == 5 { note("music: nothing captured for 5 s. If the app is playing, allow your terminal under System Settings > Privacy & Security > Screen & System Audio Recording") }
            if let sep = separator {
                let runs = sep.runs.load(ordering: .relaxed)
                return String(format: " | music in %.0f dB, gaps %d | split %d s, %d ms each, skipped %d", peak, feed.starved, runs,
                              sep.busyMs.load(ordering: .relaxed) / max(runs, 1), sep.skipped.load(ordering: .relaxed))
            }
            return String(format: " | music in %.0f dB, skips %d, gaps %d", peak, feed.dropped, feed.starved)
        }
    }
    tracker.onPose = { sample in q.async { spatial.feed(sample) } }
    for (ms, work) in [(5, { spatial.tick() }), (1000, { spatial.route(volume: volume) })] { // listener at 200 Hz, device check at 1 Hz
        let timer = DispatchSource.makeTimerSource(queue: q)
        timer.schedule(deadline: .now(), repeating: .milliseconds(ms))
        timer.setEventHandler(handler: work)
        timer.resume()
        timers.append(timer)
    }
}
guard wantView else { RunLoop.main.run(); exit(0) }

let app = NSApplication.shared
app.setActivationPolicy(.accessory) // no Dock icon, never takes keyboard focus
let screen = NSScreen.main!.visibleFrame
let win = NSWindow(contentRect: NSRect(x: screen.maxX - 980, y: screen.maxY - 580, width: 960, height: 540),
                   styleMask: [.titled, .closable, .resizable], backing: .buffered, defer: false)
win.title = "headpose"
win.contentAspectRatio = NSSize(width: 16, height: 9)
let preview = AVCaptureVideoPreviewLayer(session: session)
preview.videoGravity = .resizeAspectFill
if let c = preview.connection, c.isVideoMirroringSupported {
    c.automaticallyAdjustsVideoMirroring = false
    c.isVideoMirrored = true
}
let video = NSView(frame: win.contentView!.bounds)
video.layer = preview
video.wantsLayer = true
let overlay = Overlay(frame: video.bounds)
for v in [video, overlay] { v.autoresizingMask = [.width, .height]; win.contentView!.addSubview(v) }
tracker.overlay = overlay
NotificationCenter.default.addObserver(forName: NSWindow.willCloseNotification, object: win, queue: .main) { _ in exit(0) }
win.orderFrontRegardless()
FileHandle.standardError.write("window \(win.windowNumber)\n".data(using: .utf8)!)
app.run()
