"""Turn sampled OpenVLA image attention into scores and a synchronized 2x2 video."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


LAYER_FRACS = (0.25, 0.5, 0.75, 1.0)


def _center_square(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    h, w = image.shape[:2]
    side = min(h, w)
    x0, y0 = (w - side) // 2, (h - side) // 2
    return image[y0:y0 + side, x0:x0 + side], (x0, y0, side)


def _patch_mask(mask: np.ndarray, shape: tuple[int, int], strategy: str) -> np.ndarray:
    cropped = mask.astype(np.float32) if strategy == "resize-naive" else _center_square(mask.astype(np.float32))[0]
    return cv2.resize(cropped, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def _overlay(image: np.ndarray, grid: np.ndarray, label: str, strategy: str) -> np.ndarray:
    base = np.ascontiguousarray(image.copy())
    if strategy == "resize-naive":
        cropped, (x0, y0, width, height) = base, (0, 0, base.shape[1], base.shape[0])
    else:
        cropped, (x0, y0, side) = _center_square(base)
        width = height = side
    heat = cv2.resize(grid.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    # Per-frame normalization is for visibility only; CSV stores unnormalized mass.
    heat = np.uint8(np.clip(heat / max(float(heat.max()), 1e-12), 0, 1) * 255)
    color = cv2.cvtColor(cv2.applyColorMap(heat, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    # Keep low-attention patches visibly blue rather than showing bare RGB.
    alpha = (0.30 + heat.astype(np.float32) / 255.0 * 0.35)[..., None]
    base[y0:y0 + height, x0:x0 + width] = (cropped * (1 - alpha) + color * alpha).astype(np.uint8)
    cv2.rectangle(base, (0, 0), (340, 38), (0, 0, 0), -1)
    cv2.putText(base, label, (10, 27), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return base


def _project(points: np.ndarray, camera: dict) -> np.ndarray:
    """World coordinates to pixels, using SAPIEN's OpenCV world-to-camera matrix."""
    extrinsic = np.asarray(camera["extrinsic_cv"], dtype=np.float64)
    intrinsic = np.asarray(camera["intrinsic_cv"], dtype=np.float64)
    xyz = points @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    pixels = xyz @ intrinsic.T
    with np.errstate(divide="ignore", invalid="ignore"):
        projected = pixels[:, :2] / pixels[:, 2:3]
    projected[xyz[:, 2] <= 0] = np.nan
    return projected


def _box_corners(entity: dict, top_only: bool = False) -> np.ndarray:
    """Approximate an actor by its oriented, centered bounding box."""
    w, x, y, z = entity["quaternion_wxyz"]
    rotation = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    half = np.asarray(entity["size_xyz"], dtype=np.float64) / 2
    zs = (1,) if top_only else (-1, 1)
    local = np.array([(sx*half[0], sy*half[1], sz*half[2])
                      for sz in zs for sx in (-1, 1) for sy in (-1, 1)])
    return local @ rotation.T + np.asarray(entity["position"], dtype=np.float64)


def _polygon_distance(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    contour = np.asarray(polygon, dtype=np.float32)
    return np.array([max(0.0, -cv2.pointPolygonTest(contour, tuple(map(float, p)), True))
                     for p in points], dtype=np.float32)


def _segment_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    delta = end - start
    t = np.clip(((points - start) @ delta) / max(float(delta @ delta), 1e-9), 0, 1)
    return np.linalg.norm(points - (start + t[:, None] * delta), axis=1)


def _region_grid(geometry: dict, shape: tuple[int, int], strategy: str):
    """Assign each visual patch to its nearest projected tile or arm/cube shape."""
    camera = geometry["camera"]
    width, height = camera["width"], camera["height"]
    if strategy == "resize-naive":
        x0, y0, crop_w, crop_h = 0, 0, width, height
    else:
        side = min(width, height)
        x0, y0, crop_w, crop_h = (width - side) // 2, (height - side) // 2, side, side
    rows, cols = shape
    xx, yy = np.meshgrid(x0 + (np.arange(cols) + 0.5) * crop_w / cols,
                         y0 + (np.arange(rows) + 0.5) * crop_h / rows)
    centers = np.column_stack((xx.ravel(), yy.ravel()))

    tile_polygons = []
    distances = []
    for side_name in ("left", "right"):
        corners = _project(_box_corners(geometry[side_name], top_only=True), camera)
        if not np.isfinite(corners).all():
            raise RuntimeError(f"{side_name} tile cannot be projected into the policy camera")
        polygon = cv2.convexHull(corners.astype(np.float32)).reshape(-1, 2)
        tile_polygons.append(polygon)
        distances.append(_polygon_distance(centers, polygon))

    cube_corners = _project(_box_corners(geometry["cube"]), camera)
    visible_cube_corners = cube_corners[np.isfinite(cube_corners).all(axis=1)]
    cube_polygon = None
    arm_distance = np.full(len(centers), np.inf, dtype=np.float32)
    if len(visible_cube_corners) >= 3:
        cube_polygon = cv2.convexHull(visible_cube_corners.astype(np.float32)).reshape(-1, 2)
        arm_distance = _polygon_distance(centers, cube_polygon)
    else:
        # The cube can leave the camera view during a rollout. Keep any visible
        # corner as a small anchor; if none remain, use the gripper links below.
        for pixel in visible_cube_corners:
            arm_distance = np.minimum(arm_distance,
                                      np.maximum(0, np.linalg.norm(centers - pixel, axis=1) - 12))
    arm_points = []
    gripper_link_names = {"ee_gripper_link", "left_finger_link", "right_finger_link"}
    for link in geometry["robot_links"]:
        if link["name"] not in gripper_link_names:
            continue
        pixel = _project(np.asarray([link["position"]], dtype=np.float64), camera)[0]
        if not np.isfinite(pixel).all():
            continue
        arm_points.append(pixel)
        # A narrow capsule around each joint center, in image pixels.
        arm_distance = np.minimum(arm_distance, np.maximum(0, np.linalg.norm(centers - pixel, axis=1) - 12))
    for start, end in zip(arm_points, arm_points[1:]):
        arm_distance = np.minimum(arm_distance, np.maximum(0, _segment_distance(centers, start, end) - 12))

    # 0=left, 1=arm/cube, 2=right. Arm wins geometric overlaps.
    stacked = np.stack((distances[0], arm_distance, distances[1]), axis=1)
    stacked[:, 1] -= 1e-3
    return stacked.argmin(axis=1).reshape(shape), tile_polygons, cube_polygon, arm_points


def _write_region_preview(image: np.ndarray, zones: np.ndarray, geometry: dict,
                          strategy: str, path: Path, step: int) -> None:
    """Show one policy frame with the actual patch assignments used in the CSV."""
    base = np.ascontiguousarray(image.copy())
    h, w = base.shape[:2]
    if strategy == "resize-naive":
        x0, y0, side_w, side_h = 0, 0, w, h
    else:
        side_w = side_h = min(w, h)
        x0, y0 = (w - side_w) // 2, (h - side_h) // 2
    palette = np.array([[245, 90, 80], [100, 180, 255], [80, 210, 110]], dtype=np.uint8)
    overlay = cv2.resize(palette[zones], (side_w, side_h), interpolation=cv2.INTER_NEAREST)
    region = base[y0:y0 + side_h, x0:x0 + side_w]
    base[y0:y0 + side_h, x0:x0 + side_w] = (region * 0.55 + overlay * 0.45).astype(np.uint8)
    _, tiles, cube, arm = _region_grid(geometry, zones.shape, strategy)
    for polygon, color in zip(tiles, palette[[0, 2]]):
        cv2.polylines(base, [np.rint(polygon).astype(np.int32)], True, color.tolist(), 2)
    if cube is not None:
        cv2.polylines(base, [np.rint(cube).astype(np.int32)], True, palette[1].tolist(), 2)
    if len(arm) > 1:
        cv2.polylines(base, [np.rint(arm).astype(np.int32)], False, palette[1].tolist(), 2)
    cv2.rectangle(base, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(base, f"step {step}  LEFT=red  ARM+CUBE=blue  RIGHT=green",
                (7, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), cv2.cvtColor(base, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Could not write region preview: {path}")


def _write_analysis_csvs(images: list[np.ndarray], samples: list[dict],
                         raw_path: Path, geometry_path: Path, regions_path: Path,
                         preview_path: Path, question_id: int, seed: int, strategy: str) -> None:
    raw_fields = ["question_id", "seed", "policy_step", "layer_fraction", "layer_number",
                  "patch_row", "patch_col", "attention_weight"]
    geometry_fields = ["question_id", "seed", "policy_step", "entity_type", "entity_name",
                       "world_x", "world_y", "world_z", "quat_w", "quat_x", "quat_y", "quat_z",
                       "size_x", "size_y", "size_z", "pixel_u", "pixel_v", "camera_width",
                       "camera_height"] + [f"intrinsic_{r}{c}" for r in range(3) for c in range(3)] + [
                           f"extrinsic_{r}{c}" for r in range(4) for c in range(4)]
    region_fields = ["question_id", "seed", "policy_step", "layer_fraction", "layer_number",
                     "answer_side", "left_attention_mass", "arm_cube_attention_mass",
                     "right_attention_mass", "left_attention_percent", "arm_cube_attention_percent",
                     "right_attention_percent", "image_attention_mass"]
    preview_step = min(40, samples[-1]["step"])
    with raw_path.open("w", newline="", encoding="utf-8") as raw_file, \
         geometry_path.open("w", newline="", encoding="utf-8") as geometry_file, \
         regions_path.open("w", newline="", encoding="utf-8") as regions_file:
        raw_writer = csv.writer(raw_file)
        geometry_writer = csv.DictWriter(geometry_file, fieldnames=geometry_fields)
        region_writer = csv.writer(regions_file)
        raw_writer.writerow(raw_fields)
        geometry_writer.writeheader()
        region_writer.writerow(region_fields)
        for sample in samples:
            step = sample["step"]
            geometry = sample["geometry"]
            camera = geometry["camera"]
            for kind in ("left", "right", "cube", "robot_links"):
                entities = geometry[kind] if kind == "robot_links" else [geometry[kind]]
                for entity in entities:
                    pos = entity["position"]
                    quat = entity["quaternion_wxyz"]
                    pixel = _project(np.asarray([pos], dtype=np.float64), camera)[0]
                    size = entity["size_xyz"] or ["", "", ""]
                    geometry_writer.writerow(dict(zip(geometry_fields[:17], [
                        question_id, seed, step, "robot_link" if kind == "robot_links" else kind,
                        entity["name"], *pos, *quat, *size, *pixel])))
            camera_row = {"question_id": question_id, "seed": seed, "policy_step": step,
                          "entity_type": "camera", "entity_name": "3rd_view_camera",
                          "camera_width": camera["width"], "camera_height": camera["height"]}
            camera_row.update({f"intrinsic_{r}{c}": camera["intrinsic_cv"][r][c]
                               for r in range(3) for c in range(3)})
            extrinsic = np.asarray(camera["extrinsic_cv"], dtype=np.float64)
            if extrinsic.shape == (3, 4):
                extrinsic = np.vstack((extrinsic, [0, 0, 0, 1]))
            camera_row.update({f"extrinsic_{r}{c}": extrinsic[r, c]
                               for r in range(4) for c in range(4)})
            geometry_writer.writerow(camera_row)
            for frac in LAYER_FRACS:
                layer = sample["maps"][frac]
                grid = np.asarray(layer["grid"], dtype=np.float64)
                zones, _, _, _ = _region_grid(geometry, grid.shape, strategy)
                for row, col in np.ndindex(grid.shape):
                    raw_writer.writerow([question_id, seed, step, frac, layer["layer"],
                                         row, col, float(grid[row, col])])
                masses = [float(grid[zones == zone].sum()) for zone in range(3)]
                total = sum(masses)
                region_writer.writerow([question_id, seed, step, frac, layer["layer"],
                                        geometry["answer_side"], *masses,
                                        *(100 * mass / total if total > 0 else 0 for mass in masses), total])
                if step == preview_step and frac == 1.0:
                    _write_region_preview(images[step], zones, geometry, strategy, preview_path, step)


def write_attention_outputs(
    images: list[np.ndarray], samples: list[dict], video_path: Path,
    scores_path: Path, question_id: int, seed: int, image_strategy: str,
    raw_path: Path, geometry_path: Path, regions_path: Path, preview_path: Path,
) -> None:
    if not samples:
        raise ValueError("No attention samples were captured")
    if image_strategy not in ("resize-crop", "resize-naive"):
        raise ValueError(f"Unsupported attention-map image transform: {image_strategy}")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    _write_analysis_csvs(images, samples, raw_path, geometry_path, regions_path,
                         preview_path, question_id, seed, image_strategy)

    rows = []
    for sample in samples:
        for frac in LAYER_FRACS:
            layer = sample["maps"][frac]
            grid = layer["grid"]
            correct = _patch_mask(sample["masks"]["correct"], grid.shape, image_strategy)
            wrong = _patch_mask(sample["masks"]["wrong"], grid.shape, image_strategy)
            if correct.sum() == 0 or wrong.sum() == 0:
                raise RuntimeError(f"Board segmentation missing at question {question_id}, step {sample['step']}")
            correct_mass = float((grid * correct).sum())
            wrong_mass = float((grid * wrong).sum())
            correct_mean = correct_mass / float(correct.sum())
            wrong_mean = wrong_mass / float(wrong.sum())
            rows.append({
                "question_id": question_id, "seed": seed, "policy_step": sample["step"],
                "sim_seconds_from_first_policy_step": sample["step"] / 5.0,
                "layer_fraction": frac, "layer_number": layer["layer"],
                "correct_attention_mass": correct_mass,
                "wrong_attention_mass": wrong_mass,
                "correct_attention_per_patch": correct_mean,
                "wrong_attention_per_patch": wrong_mean,
                "correct_attention_fraction_of_two": correct_mean / max(correct_mean + wrong_mean, 1e-12),
            })
    with scores_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    h, w = images[0].shape[:2]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (2 * w, 2 * h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create attention video: {video_path}")
    try:
        selected = 0
        for step, image in enumerate(images):
            while selected + 1 < len(samples) and samples[selected + 1]["step"] <= step:
                selected += 1
            sample = samples[selected]
            panels = [
                _overlay(image, sample["maps"][frac]["grid"],
                         f"Layer {int(frac * 100)}% | sample {sample['step'] / 5:.1f}s", image_strategy)
                for frac in LAYER_FRACS
            ]
            frame = np.concatenate([
                np.concatenate(panels[:2], axis=1),
                np.concatenate(panels[2:], axis=1),
            ], axis=0)
            # Two identical output frames per 5 Hz policy step preserve simulated time at 10 fps.
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
            writer.write(bgr)
    finally:
        writer.release()
