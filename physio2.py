# knee_alignment_voice.py
# Requirements: pip install opencv-python mediapipe numpy openai

import cv2
import mediapipe as mp
import numpy as np
from openai import OpenAI
import io
import pygame
import threading
import time
import os

api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key)  # Replace with your API key

# Initialize pygame mixer for audio playback
pygame.mixer.init()

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

FRONT_LEG = "left"       # "left" or "right"
BAND_FRAC = 0.15         # corridor half-width relative to hip width
MIN_VIS = 0.5            # landmark visibility threshold
FEEDBACK_COOLDOWN = 3.0  # seconds between voice announcements


class VoiceFeedback:
    def __init__(self):
        self.last_feedback = ""
        self.last_announcement_time = 0
        self.is_speaking = False
        
    def speak(self, text):
        """Speak text using OpenAI TTS in a separate thread."""
        current_time = time.time()
        
        # Only speak if enough time has passed and we're not currently speaking
        if (text != self.last_feedback or 
            current_time - self.last_announcement_time > FEEDBACK_COOLDOWN) and \
           not self.is_speaking:
            
            self.last_feedback = text
            self.last_announcement_time = current_time
            
            # Run TTS in separate thread to avoid blocking video
            thread = threading.Thread(target=self._play_speech, args=(text,))
            thread.daemon = True
            thread.start()
    
    def _play_speech(self, text):
        """Generate and play speech audio."""
        try:
            self.is_speaking = True
            
            # Generate speech using OpenAI TTS
            response = client.audio.speech.create(
                model="tts-1",  # Use "tts-1-hd" for higher quality
                voice="nova",   # Options: alloy, echo, fable, onyx, nova, shimmer
                input=text,
                speed=1.0
            )
            
            # Convert response to audio and play
            audio_data = io.BytesIO(response.content)
            pygame.mixer.music.load(audio_data)
            pygame.mixer.music.play()
            
            # Wait for audio to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
        except Exception as e:
            print(f"TTS Error: {e}")
        finally:
            self.is_speaking = False


def xyv(lm, idx):
    p = lm[idx]
    return p.x, p.y, p.visibility


def draw_minimal_overlay(img, status="warning"):
    """Draw a minimal status indicator in the corner."""
    H, W = img.shape[:2]
    
    if status == "good":
        color = (0, 180, 0)   # green
    elif status == "error":
        color = (0, 0, 200)   # red
    else:
        color = (0, 140, 255) # orange
    
    # Small status circle in top-right corner
    cv2.circle(img, (W - 30, 30), 15, color, -1)
    cv2.circle(img, (W - 30, 30), 15, (255, 255, 255), 2)
    
    return img


# Initialize voice feedback
voice = VoiceFeedback()

cap = cv2.VideoCapture(0)
S = mp_pose.PoseLandmark

if FRONT_LEG.lower() == "left":
    HIP_IDX, KNEE_IDX, ANK_IDX = S.LEFT_HIP.value, S.LEFT_KNEE.value, S.LEFT_ANKLE.value
else:
    HIP_IDX, KNEE_IDX, ANK_IDX = S.RIGHT_HIP.value, S.RIGHT_KNEE.value, S.RIGHT_ANKLE.value

# Set up full-screen OpenCV window
cv2.namedWindow("Knee Alignment", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Knee Alignment", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        feedback = "Please face the camera"
        status = "warning"

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            lhip = xyv(lm, S.LEFT_HIP.value)
            rhip = xyv(lm, S.RIGHT_HIP.value)

            if min(lhip[2], rhip[2]) > MIN_VIS:
                hip_width = abs(lhip[0] - rhip[0])

                hip = xyv(lm, HIP_IDX)
                knee = xyv(lm, KNEE_IDX)
                ankle = xyv(lm, ANK_IDX)

                if min(hip[2], knee[2], ankle[2]) > MIN_VIS and hip_width > 0.02:
                    band_half = BAND_FRAC * hip_width
                    left_bound = hip[0] - band_half
                    right_bound = hip[0] + band_half

                    if knee[0] < left_bound:
                        feedback = "Your knee is caving out"
                        status = "warning"
                    elif knee[0] > right_bound:
                        feedback = "Your knee is caving in"
                        status = "warning"
                    else:
                        feedback = "Perfect alignment"
                        status = "good"

                    # Speak feedback
                    voice.speak(feedback)

                    # Draw pose skeleton
                    mp_draw.draw_landmarks(image, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # Vertical line through hip
                    hx = int(hip[0] * W)
                    cv2.line(image, (hx, 0), (hx, H), (255, 255, 255), 2)

                    # Corridor shading
                    lb, rb = int(left_bound * W), int(right_bound * W)
                    overlay = image.copy()
                    cv2.rectangle(overlay, (lb, 0), (rb, H), (0, 180, 0), -1)
                    image = cv2.addWeighted(overlay, 0.15, image, 0.85, 0)

                    # Markers
                    cv2.circle(image, (int(knee[0]*W), int(knee[1]*H)), 7, (255, 255, 255), -1)
                    cv2.circle(image, (int(ankle[0]*W), int(ankle[1]*H)), 7, (0, 255, 0), -1)

        # Minimal status indicator
        image = draw_minimal_overlay(image, status)

        cv2.imshow("Knee Alignment", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()