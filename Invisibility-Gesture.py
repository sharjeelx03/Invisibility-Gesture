

import cv2
import mediapipe as mp
import numpy as np
import time
import math
from collections import deque

mp_selfie = mp.solutions.selfie_segmentation
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def get_pinch(hand_landmarks, w, h):
    thumb = hand_landmarks.landmark[4]
    index = hand_landmarks.landmark[8]
    dist_norm = math.hypot(thumb.x - index.x, thumb.y - index.y)
    x1, y1 = int(thumb.x * w), int(thumb.y * h)
    x2, y2 = int(index.x * w), int(index.y * h)
    center = ((x1+x2)//2, (y1+y2)//2)
    return dist_norm, (x1,y1), (x2,y2), center

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Cannot open camera")
        return

    selfie_seg = mp_selfie.SelfieSegmentation(model_selection=1)
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

    invisible_mode = False
    last_toggle = 0
    cooldown = 1.0

    background_model = None  # float32 for smooth blending
    learned_mask = None      # bool HxW, has this pixel ever been seen as background?
    alpha = 0.07  # learning speed for background update
    frame_count = 0

    print(">>> AUTO BACKGROUND MODE")
    print("Move side to side, wave your arms to help it learn the background")
    print("Pinch to toggle | R = reset BG | Q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        seg_res = selfie_seg.process(rgb)
        hand_res = hands.process(rgb)

        mask_raw = seg_res.segmentation_mask  # 0=bg, 1=person
        if mask_raw is None:
            mask_raw = np.zeros((h,w), dtype=np.float32)

        # Soft masks
        person_mask_prob = cv2.GaussianBlur(mask_raw, (21,21), 0)  # 0..1
        person_mask_hard = person_mask_prob > 0.6
        bg_confident = person_mask_prob < 0.25  # definitely background

        # Initialize models
        if background_model is None:
            background_model = frame.astype(np.float32)
            learned_mask = bg_confident.copy()
        else:
            # Update background ONLY where we are confident it's background
            # EMA: bg = bg*(1-a) + frame*a
            if np.any(bg_confident):
                background_model[bg_confident] = (
                    background_model[bg_confident] * (1 - alpha) + 
                    frame[bg_confident].astype(np.float32) * alpha
                )
                learned_mask = learned_mask | bg_confident

        bg_uint8 = background_model.astype(np.uint8)
        learned_percent = (np.mean(learned_mask) * 100) if learned_mask is not None else 0

        # --- Gesture ---
        is_pinch = False
        if hand_res.multi_hand_landmarks:
            for hl in hand_res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(255,255,255), thickness=2))
                
                dist_norm, p1, p2, center = get_pinch(hl, w, h)
                if dist_norm < 0.05:
                    is_pinch = True
                    cv2.circle(frame, p1, 8, (0,255,255), -1)
                    cv2.circle(frame, p2, 8, (0,255,255), -1)
                    cv2.circle(frame, center, 12, (0,0,255), 2)
                    cv2.line(frame, p1, p2, (0,255,255), 2)
                    if time.time() - last_toggle > cooldown:
                        # Only allow invisibility if at least 30% learned
                        if learned_percent > 30 or invisible_mode:
                            invisible_mode = not invisible_mode
                            last_toggle = time.time()
                            print(f"Toggled -> {'INVISIBLE' if invisible_mode else 'VISIBLE'}")

        # --- Invisibility Effect (auto-bg aware) ---
        # Only become invisible where background is learned, otherwise keep you visible
        # This prevents ghosting of your initial position
        learned_3d = np.stack([learned_mask]*3, axis=-1) if learned_mask is not None else np.zeros((h,w,3), dtype=bool)
        mask_3d = np.stack([person_mask_prob]*3, axis=-1)  # soft
        
        # effective mask = person * learned (we can only hide where we know bg)
        effective_mask_3d = mask_3d * learned_3d.astype(np.float32)

        if invisible_mode:
            # Soft blend
            output = (effective_mask_3d * bg_uint8.astype(np.float32) + 
                     (1 - effective_mask_3d) * frame.astype(np.float32)).astype(np.uint8)
        else:
            output = frame.copy()

        # --- HUD ---
        # Top bar
        cv2.rectangle(output, (0,0), (w, 75), (0,0,0), -1)
        status = "INVISIBLE" if invisible_mode else "VISIBLE"
        status_color = (0,0,255) if invisible_mode else (0,255,0)
        cv2.putText(output, f"MODE: {status} | BG Learned: {learned_percent:.0f}%", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        if learned_percent < 95 and not invisible_mode:
            cv2.putText(output, f"MOVE AROUND! Learning background... {learned_percent:.0f}%", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        else:
            hint = "PINCH Thumb+Index to toggle" if not invisible_mode else "PINCH to reappear"
            cv2.putText(output, hint, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Background preview PiP (top-right)
        if bg_uint8 is not None:
            pip_h, pip_w = 140, 250
            pip = cv2.resize(bg_uint8, (pip_w, pip_h))
            # border
            cv2.rectangle(pip, (0,0), (pip_w-1, pip_h-1), (255,255,255), 2)
            # learned overlay
            if learned_mask is not None:
                mask_pip = cv2.resize(learned_mask.astype(np.uint8)*255, (pip_w, pip_h))
                # red where not learned
                overlay = pip.copy()
                overlay[mask_pip < 128] = (0,0,150)
                pip = cv2.addWeighted(pip, 0.7, overlay, 0.3, 0)
            output[10:10+pip_h, w-pip_w-10:w-10] = pip
            cv2.putText(output, "BG Model", (w-pip_w-10, 10+pip_h+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Bottom hints
        cv2.putText(output, "[R] reset BG  [Q] quit", (20, h-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        
        if is_pinch:
            cv2.putText(output, "PINCH!", (w//2 - 50, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        cv2.imshow("Auto Invisibility - No Step Out Needed", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("Resetting background model...")
            background_model = None
            learned_mask = None
            frame_count = 0

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    selfie_seg.close()
    hands.close()

if __name__ == "__main__":
    main()