import os
import glob
import json
import math

import numpy as np
import joblib

import tomllib


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

with open("config.toml", "rb") as f:
    config = tomllib.load(f)
	
script_dir = os.path.dirname(os.path.abspath(__file__))
annotations_dir = os.path.join(script_dir, "../data/annotations/")
cache_file = os.path.join(script_dir, "../data/landmarks_cache.npz")
model_out = os.path.join(script_dir, "../data/gesture_classifier.joblib")

landmark_keys = ["hand_landmarks", "landmarks"]
expected_points = 21

multi_hand_gestures = config["multi_hand_gestures"]
no_multi_label = "none"

include_no_gesture = True
max_per_class = None

rf_trees = 300
rf_max_depth = None
test_split = 0.15
random_seed = 42

def normalize_landmarks(points):
	wrist_x = points[0][0]
	wrist_y = points[0][1]
	
	pts = []
	for p in points:
		pts.append([p[0] - wrist_x, p[1] - wrist_y])
	
	scale = math.hypot(pts[9][0], pts[9][1])
	if scale < 1e-6:
		scale = 1e-6
	
	feat = []
	for p in pts:
		feat.append(p[0] / scale)
		feat.append(p[1] / scale)
	
	return feat

def normalize_two_hands(points_a, points_b):
	if points_a[0][0] > points_b[0][0]:
		points_a, points_b = points_b, points_a
	
	mid_x = (points_a[0][0] + points_b[0][0]) / 2
	mid_y = (points_a[0][1] + points_b[0][1]) / 2
	
	scale = math.hypot(points_a[0][0] - points_b[0][0], points_a[0][1] - points_b[0][1])
	if scale < 1e-6:
		scale = 1e-6
	
	feat = []
	for p in points_a + points_b:
		feat.append((p[0] - mid_x) / scale)
		feat.append((p[1] - mid_y) / scale)
	
	return feat

def get_landmarks(entry):
	for key in landmark_keys:
		if key in entry:
			return entry[key]
	return None

def under_cap(per_class_counts, label):
	if max_per_class is None:
		return True
	
	return per_class_counts.get(label, 0) < max_per_class

def count_sample(per_class_counts, label):
	per_class_counts[label] = per_class_counts.get(label, 0) + 1

def parse_file(path, per_class_counts):
	single_features = []
	single_labels = []
	multi_features = []
	multi_labels = []
	
	with open(path, "r") as f:
		data = json.load(f)
	
	for uid, entry in data.items():
		hand_landmarks = get_landmarks(entry)
		entry_labels = entry.get("labels")
		
		if hand_landmarks is None or entry_labels is None:
			continue
		
		pair_count = min(len(hand_landmarks), len(entry_labels))
		
		hands = []
		labels = []
		
		for i in range(pair_count):
			if len(hand_landmarks[i]) == expected_points:
				hands.append(hand_landmarks[i])
				labels.append(entry_labels[i])
		
		if len(hands) == 0:
			continue
		
		is_multi = False
		multi_gesture = None
		
		for label in labels:
			if label in multi_hand_gestures:
				is_multi = True
				multi_gesture = label
				break
		
		if is_multi:
			if len(hands) >= 2 and under_cap(per_class_counts, multi_gesture):
				multi_features.append(normalize_two_hands(hands[0], hands[1]))
				multi_labels.append(multi_gesture)
				count_sample(per_class_counts, multi_gesture)
			
			continue
		
		for i in range(len(hands)):
			gesture = labels[i]
			
			if gesture == "no_gesture" and not include_no_gesture:
				continue
			
			if not under_cap(per_class_counts, gesture):
				continue
			
			single_features.append(normalize_landmarks(hands[i]))
			single_labels.append(gesture)
			count_sample(per_class_counts, gesture)
		
		if len(hands) >= 2 and under_cap(per_class_counts, no_multi_label):
			multi_features.append(normalize_two_hands(hands[0], hands[1]))
			multi_labels.append(no_multi_label)
			count_sample(per_class_counts, no_multi_label)
	
	return single_features, single_labels, multi_features, multi_labels

def find_sample_entry(json_paths):
	for path in json_paths:
		with open(path, "r") as f:
			data = json.load(f)
		
		if len(data) > 0:
			return next(iter(data.values()))
	
	return None

def build_dataset():
	json_paths = sorted(glob.glob(os.path.join(annotations_dir, "**", "*.json"), recursive = True))
	
	if len(json_paths) == 0:
		print("no json files found under " + annotations_dir)
		return None
	
	first_entry = find_sample_entry(json_paths)
	
	if first_entry is None:
		print("all annotation files are empty")
		return None
	
	print("sample entry keys: " + str(list(first_entry.keys())))
	
	if get_landmarks(first_entry) is None:
		print("no landmark field found - this annotation set is bboxes only.")
		print("you would need the images to extract landmarks with mediapipe.")
		return None
	
	single_features = []
	single_labels = []
	multi_features = []
	multi_labels = []
	per_class_counts = {}
	
	for path in json_paths:
		sf, sl, mf, ml = parse_file(path, per_class_counts)
		single_features.extend(sf)
		single_labels.extend(sl)
		multi_features.extend(mf)
		multi_labels.extend(ml)
		print(os.path.basename(path) + ": " + str(len(sl)) + " single hands, " + str(len(ml)) + " hand pairs")
	
	dataset = {
		"x_single": np.array(single_features, dtype = np.float32),
		"y_single": np.array(single_labels),
		"x_multi": np.array(multi_features, dtype = np.float32),
		"y_multi": np.array(multi_labels)
	}
	
	np.savez_compressed(
		cache_file,
		x_single = dataset["x_single"],
		y_single = dataset["y_single"],
		x_multi = dataset["x_multi"],
		y_multi = dataset["y_multi"]
	)
	print("cached " + str(len(dataset["x_single"])) + " single + " + str(len(dataset["x_multi"])) + " multi samples")
	
	return dataset

def train_forest(x, y, model_name):
	if len(x) == 0:
		print("no samples for " + model_name + ", skipping")
		return None
	
	unique_classes = set(y)
	
	if len(unique_classes) < 2:
		print(model_name + " only has classes " + str(unique_classes) + ", skipping")
		return None
	
	x_train, x_test, y_train, y_test = train_test_split(
		x, y,
		test_size = test_split,
		stratify = y,
		random_state = random_seed,
	)
	
	print("\n" + model_name + ": training on " + str(len(x_train)) + ", testing on " + str(len(x_test)))
	
	clf = RandomForestClassifier(
		n_estimators = rf_trees,
		max_depth = rf_max_depth,
		n_jobs = -1,
		random_state = random_seed,
	)
	clf.fit(x_train, y_train)
	
	preds = clf.predict(x_test)
	print(classification_report(y_test, preds))
	
	return clf

def train():
	if os.path.exists(cache_file):
		data = np.load(cache_file, allow_pickle = True)
		dataset = {
			"x_single": data["x_single"],
			"y_single": data["y_single"],
			"x_multi": data["x_multi"],
			"y_multi": data["y_multi"]
		}
		print("loaded cache: " + str(len(dataset["x_single"])) + " single + " + str(len(dataset["x_multi"])) + " multi samples")
	else:
		dataset = build_dataset()
	
	if dataset is None:
		return
	
	clf_single = train_forest(dataset["x_single"], dataset["y_single"], "single hand model")
	clf_multi = train_forest(dataset["x_multi"], dataset["y_multi"], "two hand model")
	
	if clf_single is None and clf_multi is None:
		return
	
	bundle = {
		"single": clf_single,
		"multi": clf_multi,
		"multi_hand_gestures": multi_hand_gestures,
		"no_multi_label": no_multi_label
	}
	
	joblib.dump(bundle, model_out)
	print("\nsaved model bundle to " + model_out)

if __name__ == "__main__":
	train()