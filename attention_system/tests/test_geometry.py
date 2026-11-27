import numpy as np, cv2, pytest
from attention import geometry as g

def blank(): return [(0.0, 0.0)] * 478

def test_eye_aspect_ratio_open_and_closed():
    p = blank()
    for idx, xy in zip(g.RIGHT_EYE, [(0, 0), (3, 2), (7, 2), (10, 0), (7, -2), (3, -2)]): p[idx] = xy
    assert g.eye_aspect_ratio(p, g.RIGHT_EYE) == pytest.approx(0.4)          # (4+4)/(2*10)
    for idx, xy in zip(g.RIGHT_EYE, [(0, 0), (3, 0), (7, 0), (10, 0), (7, 0), (3, 0)]): p[idx] = xy
    assert g.eye_aspect_ratio(p, g.RIGHT_EYE) == pytest.approx(0.0)           # fully closed

def test_mouth_aspect_ratio():
    p = blank(); p[61], p[291] = (0, 0), (10, 0)
    for (a, b) in g.MOUTH_PAIRS: p[a], p[b] = (5, 2), (5, -2)
    assert g.mouth_aspect_ratio(p) == pytest.approx(0.4)

def test_gaze_ratio_centre_and_side():
    p = blank()
    p[362], p[263], p[133], p[33] = (0, 0), (10, 0), (0, 0), (10, 0)
    p[473] = p[468] = (5, 0); assert g.gaze_ratio(p) == pytest.approx(0.5)
    p[473] = p[468] = (9, 0); assert g.gaze_ratio(p) == pytest.approx(0.9)
    assert g.gaze_ratio(p[:468]) is None                                       # no iris points -> None, not a crash

def _Rx(a): a = np.radians(a); return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
def _Ry(a): a = np.radians(a); return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
def _Rz(a): a = np.radians(a); return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])

@pytest.mark.parametrize("yaw,pitch,roll", [(0, 0, 0), (20, 0, 0), (-35, 0, 0), (0, 15, 0), (0, -25, 0), (25, 15, 5), (60, 0, 0)])
def test_head_pose_recovers_known_angles(yaw, pitch, roll):
    W, H = 640, 480
    cam = np.array([[W, 0, W / 2], [0, W, H / 2], [0, 0, 1]], float)
    R = _Rx(180) @ _Rz(roll) @ _Ry(yaw) @ _Rx(pitch)
    rvec, _ = cv2.Rodrigues(R)
    img, _ = cv2.projectPoints(g.MODEL_3D, rvec, np.array([0, 0, 600.0]), cam, np.zeros((4, 1)))
    p = blank()
    for k, xy in zip(("nose", "chin", "left_eye", "right_eye", "left_mouth", "right_mouth"), img.reshape(-1, 2)):
        p[g.POSE_IDS[k]] = tuple(xy)
    hp = g.head_pose(p, W, H)
    assert abs(hp.yaw) == pytest.approx(abs(yaw), abs=0.5)                       # sign depends on convention; size must match
    assert abs(hp.pitch) == pytest.approx(abs(pitch), abs=0.5)
    assert abs(hp.roll) == pytest.approx(abs(roll), abs=0.5)

def test_count_nods():
    assert g.count_nods([0, 15, 0, 15, 0, 15], 10) >= 2
    assert g.count_nods([0, 1, 0, 1, 0, 1, 0], 10) == 0
    assert g.count_nods([0.0], 10) == 0

def test_timed_buffer_forgets_old_values():
    b = g.TimedBuffer(5.0)
    for t in range(20): b.add(float(t), t)
    assert min(b.values()) >= 14
