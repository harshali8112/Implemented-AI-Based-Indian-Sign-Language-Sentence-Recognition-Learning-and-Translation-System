"""Shared MediaPipe Holistic preprocessing for training and live inference."""

from __future__ import annotations

from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

try:
    from mediapipe.python.solutions import holistic as holistic_module
except ImportError:
    holistic_module = getattr(getattr(mp, "solutions", None), "holistic", None)

POSITION_DIM = 235
FEATURE_DIM = 470
SEQUENCE_LENGTH = 35

POSE_LANDMARK_COUNT = 33
HAND_LANDMARK_COUNT = 21
NOSE_INDEX = 0
LEFT_SHOULDER_INDEX = 11
RIGHT_SHOULDER_INDEX = 12
LEFT_WRIST_INDEX = 15
RIGHT_WRIST_INDEX = 16

FINGER_JOINTS = (
    (2, 3, 4),
    (5, 6, 8),
    (9, 10, 12),
    (13, 14, 16),
    (17, 18, 20),
)


class HolisticFeatureExtractor:
    """Convert BGR frames into normalized 470-dimensional features."""

    def __init__(self) -> None:
        if holistic_module is None:
            raise ImportError(
                "This project requires the legacy MediaPipe Holistic API. "
                "Install a MediaPipe release below 1.0 with requirements.txt."
            )
        self._holistic = holistic_module.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def close(self) -> None:
        self._holistic.close()

    def __enter__(self) -> "HolisticFeatureExtractor":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def frame_to_feature(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._holistic.process(frame_rgb)
        return landmarks_to_feature(results)


def _landmark_array(landmarks, count: int) -> Optional[np.ndarray]:
    if landmarks is None:
        return None
    return np.asarray(
        [[point.x, point.y, point.z] for point in landmarks.landmark],
        dtype=np.float32,
    ).reshape(count, 3)


def _safe_scale(pose: np.ndarray) -> float:
    if pose is None:
        return 1.0
    distance = float(np.linalg.norm(pose[LEFT_SHOULDER_INDEX] - pose[RIGHT_SHOULDER_INDEX]))
    return max(distance, 1e-4)


def _normalize(points: np.ndarray, nose: np.ndarray, body_scale: float) -> np.ndarray:
    return (points - nose) / body_scale


def _hand_points(
    hand: Optional[np.ndarray],
    pose: Optional[np.ndarray],
    wrist_index: int,
    nose: np.ndarray,
    body_scale: float,
) -> np.ndarray:
    if hand is None:
        wrist = pose[wrist_index] if pose is not None else nose
        hand = np.repeat(wrist[np.newaxis, :], HAND_LANDMARK_COUNT, axis=0)
    return _normalize(hand, nose, body_scale)


def _finger_angles(hand: np.ndarray) -> np.ndarray:
    angles = []
    for mcp_index, pip_index, tip_index in FINGER_JOINTS:
        first = hand[mcp_index] - hand[pip_index]
        second = hand[tip_index] - hand[pip_index]
        first_norm = np.linalg.norm(first)
        second_norm = np.linalg.norm(second)
        if first_norm < 1e-6 or second_norm < 1e-6:
            angles.append(0.0)
            continue
        cosine = np.dot(first, second) / (first_norm * second_norm)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))
        angles.append(float(angle / np.pi))
    return np.asarray(angles, dtype=np.float32)


def landmarks_to_position_feature(results) -> np.ndarray:
    pose = _landmark_array(results.pose_landmarks, POSE_LANDMARK_COUNT)
    left_hand = _landmark_array(results.left_hand_landmarks, HAND_LANDMARK_COUNT)
    right_hand = _landmark_array(results.right_hand_landmarks, HAND_LANDMARK_COUNT)

    if pose is None:
        pose = np.zeros((POSE_LANDMARK_COUNT, 3), dtype=np.float32)
    nose = pose[NOSE_INDEX].copy()
    body_scale = _safe_scale(pose)

    pose_norm = _normalize(pose, nose, body_scale)
    left_norm = _hand_points(left_hand, pose, LEFT_WRIST_INDEX, nose, body_scale)
    right_norm = _hand_points(right_hand, pose, RIGHT_WRIST_INDEX, nose, body_scale)
    left_angles = _finger_angles(left_norm)
    right_angles = _finger_angles(right_norm)

    feature = np.concatenate(
        [left_norm.reshape(-1), right_norm.reshape(-1), pose_norm.reshape(-1), left_angles, right_angles]
    ).astype(np.float32)
    if feature.shape != (POSITION_DIM,):
        raise RuntimeError(f"Expected {POSITION_DIM} position features, got {feature.shape}")
    return feature


def landmarks_to_feature(results) -> np.ndarray:
    position = landmarks_to_position_feature(results)
    return np.concatenate([position, np.zeros(POSITION_DIM, dtype=np.float32)])


def resample_sequence(features: np.ndarray, length: int = SEQUENCE_LENGTH) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != POSITION_DIM:
        raise ValueError(f"Expected (frames, {POSITION_DIM}), got {features.shape}")
    if len(features) == 0:
        raise ValueError("Cannot resample an empty sequence")
    if len(features) == length:
        return features.copy()
    source = np.linspace(0.0, 1.0, len(features), dtype=np.float32)
    target = np.linspace(0.0, 1.0, length, dtype=np.float32)
    return np.stack([np.interp(target, source, features[:, index]) for index in range(POSITION_DIM)], axis=1).astype(np.float32)


def add_velocity(position_sequence: np.ndarray) -> np.ndarray:
    position_sequence = np.asarray(position_sequence, dtype=np.float32)
    delta = np.diff(position_sequence, axis=0, prepend=position_sequence[:1])
    features = np.concatenate([position_sequence, delta], axis=1)
    if features.shape[1] != FEATURE_DIM:
        raise RuntimeError(f"Expected {FEATURE_DIM} features, got {features.shape[1]}")
    return features.astype(np.float32)


def process_position_sequence(position_sequence: np.ndarray) -> np.ndarray:
    return add_velocity(resample_sequence(position_sequence))
