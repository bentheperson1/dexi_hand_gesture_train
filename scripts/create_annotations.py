import os
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import tomllib

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "../data/internal/hand_landmarker.task")
images_path = os.path.join(script_dir, "../data/built_dataset/")
annotation_path = os.path.join(script_dir, "../data/annotations/")

multi_hand_gestures = config["multi_hand_gestures"]

image_extensions = (".jpg", ".jpeg", ".png", ".bmp")

box_pad = 0.05
user_id = "custom_00"

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
	base_options=BaseOptions(model_asset_path=model_path),
	running_mode=VisionRunningMode.IMAGE,
	num_hands=2,
	min_hand_detection_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)

print("created landmarker")

if not os.path.exists(annotation_path):
	os.makedirs(annotation_path)

gesture_folders = [f for f in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, f))]

for gesture_name in gesture_folders:
	folder_path = os.path.join(images_path, gesture_name)
	image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

	annotations = {}
	skipped = 0

	required_hands = 2 if gesture_name in multi_hand_gestures else 1

	print(f"\nprocessing '{gesture_name}' ({len(image_files)} images)")

	for image_file in image_files:
		image_full_path = os.path.join(folder_path, image_file)
		img_key = os.path.splitext(image_file)[0]

		try:
			mp_image = mp.Image.create_from_file(image_full_path)
			result = landmarker.detect(mp_image)
		except Exception as e:
			print(f"  failed to read {image_file}: {e}")
			skipped += 1
			continue

		if not result.hand_landmarks:
			skipped += 1
			continue

		hands = result.hand_landmarks

		if len(hands) < required_hands:
			skipped += 1
			continue

		bboxes = []
		landmarks_all = []
		labels = []

		for hand in hands:
			xs = [lm.x for lm in hand]
			ys = [lm.y for lm in hand]

			box_x1 = max(0.0, min(xs) - box_pad)
			box_y1 = max(0.0, min(ys) - box_pad)
			box_x2 = min(1.0, max(xs) + box_pad)
			box_y2 = min(1.0, max(ys) + box_pad)

			bboxes.append([box_x1, box_y1, box_x2 - box_x1, box_y2 - box_y1])
			landmarks_all.append([[lm.x, lm.y] for lm in hand])
			labels.append(gesture_name)

		leading_hand = "right"
		leading_conf = 1.0

		if result.handedness:
			best = max(result.handedness, key=lambda h: h[0].score)
			leading_hand = best[0].category_name.lower()
			leading_conf = best[0].score

		annotations[img_key] = {
			"bboxes": bboxes,
			"labels": labels,
			"landmarks": landmarks_all,
			"leading_hand": leading_hand,
			"leading_conf": leading_conf,
			"user_id": user_id
		}

	output_file = os.path.join(annotation_path, f"{gesture_name}.json")

	with open(output_file, "w") as f:
		json.dump(annotations, f, indent="\t")

	print(f"  saved {len(annotations)} annotations to {output_file} (skipped {skipped})")

print("\ndone")