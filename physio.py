import cv2
import mediapipe as mp
import numpy as np
import time

# -------------------------- Config (tweak as needed) --------------------------
EXERCISE_NAME = "Split Squat Coach"
FRONT_LEG = "left"          # default tracked front leg: "left" or "right"

# Rep logic (knee angle at the FRONT knee: hip-knee-ankle)
TOP_ANGLE = 165             # near full extension at the top
BOTTOM_ANGLE = 90           # target depth at the bottom (~thigh near parallel)
ANGLE_HYSTERESIS = 4        # buffer to avoid jitter at thresholds

# Cues / form thresholds
VIS_THRESH = 0.5            # min visibility to trust a landmark
EMA_ALPHA = 0.7             # smoothing factor for knee angle
TORSO_MAX_TILT_DEG = 25     # keep torso within ~25° of vertical
KNEE_TOE_X_FRAC = 0.12      # knee should stay within ~12% of frame width from toes (side view heuristic)
HEEL_LIFT_FRAC = 0.04       # heel considered lifted if it rises >4% of frame height vs baseline (top)
MIN_HALF_REP_SEC = 0.8      # if faster than this per half-rep -> "slow down"

# -----------------------------------------------------------------------------

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def deg(x):
    return float(x)

def angle_at_joint(a, b, c):
    """Angle at point b (in degrees) given pixel coords a-b-c."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(np.degrees(radians))
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

def angle_to_vertical(p_up, p_down):
    """Angle (deg) between the segment p_down->p_up and the vertical axis."""
    dx = float(p_up[0] - p_down[0])
    dy = float(p_up[1] - p_down[1])
    # angle to vertical = atan2(|dx|, |dy|)
    return np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6))

def put_panel(img, lines, org=(10, 10), pad=6, line_h=22, fg=(255,255,255), bg=(30,30,30)):
    lines = [str(l) for l in lines]
    w = max([cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0] for l in lines] + [0]) + 2*pad
    h = line_h*len(lines) + 2*pad
    x, y = org
    cv2.rectangle(img, (x, y), (x+w, y+h), bg, -1)
    for i, l in enumerate(lines):
        cv2.putText(img, l, (x+pad, y+pad + (i+1)*line_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, fg, 1, cv2.LINE_AA)

def indices_for_leg(front_leg):
    if front_leg == "left":
        return dict(
            HIP = mp_pose.PoseLandmark.LEFT_HIP.value,
            KNEE = mp_pose.PoseLandmark.LEFT_KNEE.value,
            ANKLE = mp_pose.PoseLandmark.LEFT_ANKLE.value,
            HEEL = mp_pose.PoseLandmark.LEFT_HEEL.value,
            TOE = mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value,
            SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        )
    else:
        return dict(
            HIP = mp_pose.PoseLandmark.RIGHT_HIP.value,
            KNEE = mp_pose.PoseLandmark.RIGHT_KNEE.value,
            ANKLE = mp_pose.PoseLandmark.RIGHT_ANKLE.value,
            HEEL = mp_pose.PoseLandmark.RIGHT_HEEL.value,
            TOE = mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value,
            SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        )

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow is snappier on Windows
    if not cap.isOpened():
        print("Could not open camera. Try index 1 or 2, and close any app using the webcam.")
        return

    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=1,
                        smooth_landmarks=True,
                        enable_segmentation=False,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6)

    front_leg = FRONT_LEG
    idx = indices_for_leg(front_leg)

    # State
    ema_knee_angle = None
    stage = None               # 'down' or 'up'
    reps = 0
    rep_start_t = None
    min_angle_this_rep = 999
    max_angle_this_rep = -999
    last_time = time.time()

    baseline_heel_y = None     # captured near top to detect heel lift

    print("Split Squat Coach")
    print("- Face side-on to the camera with your FRONT leg closest.")
    print("- Press 'F' to switch tracked front leg, 'q' or ESC to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        vis = frame.copy()

        tips = []
        info_lines = []

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            # Check visibility of required landmarks
            needed = [idx["HIP"], idx["KNEE"], idx["ANKLE"], idx["HEEL"], idx["TOE"], idx["SHOULDER"]]
            if all(lm[i].visibility >= VIS_THRESH for i in needed):
                # Pixel coords
                HIP = (int(lm[idx["HIP"]].x*w), int(lm[idx["HIP"]].y*h))
                KNEE = (int(lm[idx["KNEE"]].x*w), int(lm[idx["KNEE"]].y*h))
                ANKLE = (int(lm[idx["ANKLE"]].x*w), int(lm[idx["ANKLE"]].y*h))
                HEEL = (int(lm[idx["HEEL"]].x*w), int(lm[idx["HEEL"]].y*h))
                TOE = (int(lm[idx["TOE"]].x*w), int(lm[idx["TOE"]].y*h))
                SHOULDER = (int(lm[idx["SHOULDER"]].x*w), int(lm[idx["SHOULDER"]].y*h))

                # Knee angle (front knee)
                raw_angle = angle_at_joint(HIP, KNEE, ANKLE)
                ema_knee_angle = raw_angle if ema_knee_angle is None else \
                                 (EMA_ALPHA*raw_angle + (1-EMA_ALPHA)*ema_knee_angle)
                knee_angle = ema_knee_angle

                # Rep state machine: down (angle <= bottom), up (angle >= top)
                now = time.time()
                if stage is None:
                    stage = 'up' if knee_angle >= (TOP_ANGLE + BOTTOM_ANGLE)/2 else 'down'
                    rep_start_t = now
                    min_angle_this_rep = 999
                    max_angle_this_rep = -999

                # Transition up -> down when we pass bottom threshold
                if knee_angle <= (BOTTOM_ANGLE - ANGLE_HYSTERESIS) and stage == 'up':
                    stage = 'down'
                    # record start of descent
                    rep_start_t = now
                    min_angle_this_rep = min(min_angle_this_rep, knee_angle)
                    max_angle_this_rep = max(max_angle_this_rep, knee_angle)

                # Transition down -> up when we pass top threshold
                elif knee_angle >= (TOP_ANGLE + ANGLE_HYSTERESIS) and stage == 'down':
                    stage = 'up'
                    reps += 1
                    # Depth cue
                    if min_angle_this_rep > (BOTTOM_ANGLE + 5):
                        tips.append("Go a bit deeper (aim ~90° at the front knee).")
                    # Tempo cue
                    if rep_start_t is not None:
                        rep_dur = now - rep_start_t
                        if rep_dur/2 < MIN_HALF_REP_SEC:
                            tips.append("Slow the tempo (control down & up).")
                    # reset per-rep metrics
                    rep_start_t = now
                    min_angle_this_rep = 999
                    max_angle_this_rep = -999
                    # update heel baseline at top (for heel-lift detection)
                    baseline_heel_y = HEEL[1]

                # Track per-rep min/max angles
                min_angle_this_rep = min(min_angle_this_rep, knee_angle)
                max_angle_this_rep = max(max_angle_this_rep, knee_angle)

                # Form cues (posture/alignment)
                # 1) Torso upright
                torso_tilt = angle_to_vertical(SHOULDER, HIP)
                if torso_tilt > TORSO_MAX_TILT_DEG:
                    tips.append("Keep torso more upright.")

                # 2) Knee vs toes (horizontal travel heuristic, side view)
                knee_toe_dx = abs(KNEE[0] - TOE[0]) / float(w)
                if knee_toe_dx > KNEE_TOE_X_FRAC and stage == 'down':
                    tips.append("Drop more straight down (knee over mid-foot).")

                # 3) Heel lift (compare to baseline captured at top)
                if baseline_heel_y is not None:
                    if (baseline_heel_y - HEEL[1]) / float(h) > HEEL_LIFT_FRAC:
                        tips.append("Keep front heel down.")

                # Draw pose & UI
                mp_drawing.draw_landmarks(vis, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Angle text near knee
                cv2.putText(vis, f"{int(knee_angle)}°", (KNEE[0]+8, KNEE[1]-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

                # Progress bar: 0 = top, 1 = bottom
                prog = (TOP_ANGLE - knee_angle) / (TOP_ANGLE - BOTTOM_ANGLE)
                prog = float(np.clip(prog, 0.0, 1.0))
                bar_x1, bar_y1 = 20, h-30
                bar_x2, bar_y2 = 40, int(h-30 - prog*(h*0.5))
                cv2.rectangle(vis, (bar_x1, int(h-30 - 0.5*h)), (bar_x2, h-30), (60,60,60), 2)
                cv2.rectangle(vis, (bar_x1, bar_y2), (bar_x2, h-30), (0,200,0), -1)

                # Info panel
                fps = None
                now_time = time.time(); dt = now_time - last_time
                if dt > 0: fps = 1.0/dt
                last_time = now_time

                info_lines = [
                    f"{EXERCISE_NAME} — Front leg: {front_leg.upper()}",
                    f"Reps: {reps}",
                    f"Stage: {'DOWN' if stage=='down' else 'UP'}",
                    f"Knee angle: {int(knee_angle)}°",
                    f"Torso tilt: {int(torso_tilt)}°",
                    f"FPS: {fps:.1f}" if fps else ""
                ]
                put_panel(vis, [l for l in info_lines if l], org=(10,10))

                if tips:
                    put_panel(vis, ["Tips:"] + tips[:2], org=(10, 150), bg=(20,70,140))

            else:
                put_panel(vis, ["Make sure the FRONT hip, knee, ankle, heel, toe,",
                                "and shoulder are visible. Stand side-on."],
                          org=(10,10), bg=(0,0,90))

        else:
            put_panel(vis, ["No person detected.", "Step back and ensure good lighting."],
                      org=(10,10), bg=(0,0,90))

        title = f"{EXERCISE_NAME} — Press F=swap leg, q/ESC=quit"
        cv2.imshow(title, vis)
        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord('q')]:
            break
        if key in [ord('f'), ord('F')]:
            # swap front leg
            front_leg = "right" if front_leg == "left" else "left"
            ema_knee_angle = None
            baseline_heel_y = None
            stage = None
            idx = indices_for_leg(front_leg)

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    main()
