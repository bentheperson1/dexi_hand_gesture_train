import cv2
import time
import os
import glob
import random
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "../data/internal/hand_landmarker.task")
saved_data_path = os.path.join(script_dir, "../data/built_dataset/")

multi_hand_mode = False
required_hands = 2

gesture_name = ""

augment_variations = 6

flip_chance = 0.5
brightness_chance = 0.6
contrast_chance = 0.5
rotate_chance = 0.6
hue_chance = 0.4
noise_chance = 0.4
blur_chance = 0.3

brightness_range = 40
contrast_range = 0.35
rotate_range = 15
hue_range = 10
sat_range = 0.3
noise_strength = 18
blur_kernel = 3

capture_count = 0
last_saved_base = None

recording = False
record_countdown_active = False
record_countdown_start = 0
record_countdown_seconds = 3
capture_total_per_record = 5
captures_taken = 0
capture_interval = 5
capture_frame_timer = 0



flash_timer = 0
flash_time = 6

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

latest_result = None
busy = False

def result_callback(result, output_image, timestamp_ms):
	global busy
	global latest_result

	busy = False
	latest_result = result

def clamp01(v):
	return max(0.0, min(1.0, v))

def aug_flip(img):
	return cv2.flip(img, 1)

def aug_brightness(img, amount):
	return cv2.convertScaleAbs(img, alpha=1.0, beta=amount)

def aug_contrast(img, amount):
	return cv2.convertScaleAbs(img, alpha=amount, beta=0)

def aug_rotate(img, angle):
	h, w = img.shape[:2]
	center = (w / 2, h / 2)
	matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
	return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

def aug_hue_saturation(img, hue_shift, sat_scale):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
	hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
	hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
	return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

def aug_noise(img, strength):
	noise = np.random.randint(-strength, strength + 1, img.shape, dtype=np.int16)
	noisy = img.astype(np.int16) + noise
	return np.clip(noisy, 0, 255).astype(np.uint8)

def aug_blur(img, ksize):
	return cv2.GaussianBlur(img, (ksize, ksize), 0)

def make_augmented(img):
	out = img.copy()

	if random.random() < flip_chance:
		out = aug_flip(out)

	if random.random() < brightness_chance:
		out = aug_brightness(out, random.randint(-brightness_range, brightness_range))

	if random.random() < contrast_chance:
		out = aug_contrast(out, random.uniform(1.0 - contrast_range, 1.0 + contrast_range))

	if random.random() < rotate_chance:
		out = aug_rotate(out, random.uniform(-rotate_range, rotate_range))

	if random.random() < hue_chance:
		out = aug_hue_saturation(out, random.randint(-hue_range, hue_range), random.uniform(1.0 - sat_range, 1.0 + sat_range))

	if random.random() < noise_chance:
		out = aug_noise(out, noise_strength)

	if random.random() < blur_chance:
		out = aug_blur(out, blur_kernel)

	return out

def count_existing(name):
	folder = saved_data_path + name
	if not os.path.exists(folder):
		return 0

	return len(glob.glob(f"{folder}/*.jpg"))

def set_gesture(name):
	global gesture_name
	global capture_count
	global last_saved_base

	gesture_name = name

	if not os.path.exists(saved_data_path + gesture_name):
		os.makedirs(saved_data_path + gesture_name)

	capture_count = count_existing(gesture_name)
	last_saved_base = None

def save_capture(cropped_frame):
	global capture_count
	global last_saved_base
	global flash_timer

	timestamp = int(time.time() * 1000)
	base_name = f"{saved_data_path}{gesture_name}/{timestamp}"

	cv2.imwrite(f"{base_name}_orig.jpg", cropped_frame)

	for i in range(augment_variations):
		aug_frame = make_augmented(cropped_frame)
		cv2.imwrite(f"{base_name}_aug{i}.jpg", aug_frame)

	capture_count += augment_variations + 1
	last_saved_base = base_name
	flash_timer = flash_time

def undo_last():
	global capture_count
	global last_saved_base

	if last_saved_base is None:
		return

	removed = 0
	for path in glob.glob(f"{last_saved_base}_*.jpg"):
		os.remove(path)
		removed += 1

	capture_count -= removed
	last_saved_base = None
	print(f"Removed {removed} images")

def draw_text_outlined(frame, text, pos, scale, color, thickness):
	cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness)
	#cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def draw_hud(frame, hand_detected):
	h, w = frame.shape[:2]

	if record_countdown_active:
		remaining = record_countdown_seconds - (time.time() - record_countdown_start)
		status = f"starting in {int(remaining) + 1}"
		status_color = (0, 255, 255)
	elif recording:
		status = "RECORDING"
		status_color = (0, 0, 255)
	else:
		status = "paused"
		status_color = (200, 200, 200)

	hand_text = "hand ok" if hand_detected else "no hand"
	hand_color = (0, 255, 0) if hand_detected else (0, 165, 255)

	cv2.rectangle(frame, (0, 0), (150, 150), (255, 255, 255), -1)
	cv2.rectangle(frame, (0, h - 40), (w, h), (255, 255, 255), -1)

	draw_text_outlined(frame, f"gesture: {gesture_name}", (10, 25), 0.6, (255, 255, 255), 2)
	draw_text_outlined(frame, f"saved: {capture_count}", (10, 50), 0.6, (255, 255, 255), 2)
	draw_text_outlined(frame, hand_text, (10, 75), 0.6, hand_color, 2)
	draw_text_outlined(frame, status, (10, 100), 0.6, status_color, 2)

	mode_text = "mode: two hands" if multi_hand_mode else "mode: one hand"
	draw_text_outlined(frame, mode_text, (10, 125), 0.6, (255, 255, 0), 2)

	draw_text_outlined(frame, "s: save   r: record   z: undo   n: new gesture   q: quit   m: toggle multi-hand mode", (10, h - 15), 0.5, (180, 180, 180), 1)

	if flash_timer > 0:
		cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 255, 0), 8)

options = HandLandmarkerOptions(
	base_options=BaseOptions(model_asset_path=model_path),
	running_mode=VisionRunningMode.LIVE_STREAM,
	num_hands=2,
	min_hand_detection_confidence=0.5,
	min_hand_presence_confidence=0.5,
	min_tracking_confidence=0.5,
	result_callback=result_callback
)

landmarker = HandLandmarker.create_from_options(options)

print("created landmarker")

CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

print("created camera")

set_gesture(input("Enter gesture name: "))

start_time = time.time()

while cap.isOpened():
	ret, frame = cap.read()
	if not ret:
		break

	key = cv2.waitKey(1) & 0xFF

	frame = cv2.flip(frame, 1)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb)

	if not busy:
		busy = True
		timestamp_ms = int((time.time() - start_time) * 1000)
		landmarker.detect_async(mp_image, timestamp_ms)

	hand_detected = False
	cropped_frame = None

	if latest_result is not None and latest_result.hand_landmarks:
		hands = latest_result.hand_landmarks
		h, w, _ = frame.shape

		box_pad = 0.05

		if multi_hand_mode:
			enough_hands = len(hands) >= required_hands
			xs = [lm.x for hand in hands for lm in hand]
			ys = [lm.y for hand in hands for lm in hand]
		else:
			enough_hands = True
			xs = [lm.x for lm in hands[0]]
			ys = [lm.y for lm in hands[0]]

		if enough_hands:
			box_x1 = clamp01(min(xs) - box_pad)
			box_y1 = clamp01(min(ys) - box_pad)
			box_x2 = clamp01(max(xs) + box_pad)
			box_y2 = clamp01(max(ys) + box_pad)

			px1 = int(box_x1 * w)
			py1 = int(box_y1 * h)
			px2 = int(box_x2 * w)
			py2 = int(box_y2 * h)

			cropped_frame = frame[py1:py2, px1:px2].copy()

			if cropped_frame.size > 0:
				hand_detected = True
				cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
				cv2.imshow("cropped hand", cropped_frame)

	want_capture = (key == ord('s'))

	if key == ord('r'):
		if recording:
			recording = False
			captures_taken = 0
		elif record_countdown_active:
			record_countdown_active = False
		else:
			record_countdown_active = True
			record_countdown_start = time.time()

	if record_countdown_active:
		elapsed = time.time() - record_countdown_start

		if elapsed >= record_countdown_seconds:
			record_countdown_active = False
			recording = True
			capture_frame_timer = 0

	if recording and hand_detected:
		capture_frame_timer -= 1

		if capture_frame_timer <= 0:
			want_capture = True
			capture_frame_timer = capture_interval

			captures_taken += 1

			if captures_taken >= capture_total_per_record:
				recording = False
				captures_taken = 0
	else:
		capture_frame_timer = 0

	if key == ord('m'):
		multi_hand_mode = not multi_hand_mode

	if want_capture and hand_detected:
		save_capture(cropped_frame)
		print(f"Saved {augment_variations + 1} images (total {capture_count})")

	if key == ord('z'):
		undo_last()

	if key == ord('n'):
		recording = False
		new_name = input("Enter gesture name: ")

		if new_name.strip() != "":
			set_gesture(new_name.strip())

	if flash_timer > 0:
		flash_timer -= 1

	draw_hud(frame, hand_detected)
	cv2.imshow("hand landmarker", frame)

	if key == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()