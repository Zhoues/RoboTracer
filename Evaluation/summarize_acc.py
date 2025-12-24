import argparse
import json
import os
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist, directed_hausdorff


# =========================
# Part 1: Loading + Utilities for 2D Visual Trace
# =========================

def extract_intrinsics_from_matrix(matrix: List[List[float]]) -> Dict[str, float]:
    """
    Extract (fx, fy, cx, cy) from an intrinsics matrix.

    Supported formats:
      - 4x4 matrix: uses [0][0], [1][1], [0][2], [1][2]
      - 3x3 matrix: uses [0][0], [1][1], [0][2], [1][2]
    """
    if matrix is None:
        raise ValueError("Intrinsics matrix is None")

    mat = np.array(matrix, dtype=np.float32)
    if mat.shape == (4, 4):
        fx, fy, cx, cy = mat[0, 0], mat[1, 1], mat[0, 2], mat[1, 2]
    elif mat.shape == (3, 3):
        fx, fy, cx, cy = mat[0, 0], mat[1, 1], mat[0, 2], mat[1, 2]
    else:
        raise ValueError(f"Unsupported intrinsics shape: {mat.shape}")

    return {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)}


def text2pts_normalize(
    text: str,
    width: int,
    height: int,
    is_absolute: bool = False,
    is_normalized_1000: bool = False,
) -> np.ndarray:
    """
    Parse model output text and return points as a numpy array.

    Expected formats:
      - 2D: [(x, y), ...]
      - 3D: [(x, y, d), ...]  (d is absolute depth in meters)

    Normalization options:
      - is_normalized_1000: x,y are in [0,1000], convert to [0,1] by /1000
      - is_absolute: x,y are in pixels, convert to [0,1] by /width and /height

    Returns:
      np.ndarray of shape (N,2) or (N,3). Empty -> shape (0,2).
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
    matches = re.findall(pattern, text)

    points: List[Tuple[float, ...]] = []
    for match in matches:
        nums = [float(s.strip()) for s in match.split(",")]
        if len(nums) < 2:
            continue

        x, y = nums[0], nums[1]
        d = nums[2] if len(nums) >= 3 else None

        if is_normalized_1000:
            x = round(x / 1000.0, 6)
            y = round(y / 1000.0, 6)
        elif is_absolute:
            x = round(x / float(width), 6)
            y = round(y / float(height), 6)

        points.append((x, y, d) if d is not None else (x, y))

    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # If mixed lengths occur (rare), truncate to 2D for safety
    max_dim = max(len(p) for p in points)
    if max_dim == 2:
        return np.array(points, dtype=np.float32)
    else:
        # Keep 3D where possible; if any row is 2D, pad with NaN
        arr = np.full((len(points), 3), np.nan, dtype=np.float32)
        for i, p in enumerate(points):
            arr[i, : len(p)] = np.array(p, dtype=np.float32)
        return arr

def json2pts_normalize(text, width=640, height=480, is_absolute: bool = False, is_normalized_1000: bool = False):
    """
    Parse model output text and return points as a numpy array.

    Expected formats:
      - 2D: [(x, y), ...]
      - 3D: [(x, y, d), ...]  (d is absolute depth in meters)

    Normalization options:
      - is_normalized_1000: x,y are in [0,1000], convert to [0,1] by /1000
      - is_absolute: x,y are in pixels, convert to [0,1] by /width and /height
    """
    match = re.search(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    if not match:
        return np.empty((0, 3), dtype=float)

    json_cleaned = match.group(1).strip()

    try:
        data = json.loads(json_cleaned)
    except json.JSONDecodeError as e:
        return np.empty((0, 3), dtype=float)

    points = []

    def find_points(obj):
        if isinstance(obj, list):
            if len(obj) in [2, 3] and all(isinstance(v, (int, float)) for v in obj):
                points.append(obj)
            else:
                for item in obj:
                    find_points(item)
        elif isinstance(obj, dict):
            for value in obj.values():
                find_points(value)

    find_points(data)

    processed = []
    for pt in points:
        if len(pt) == 2:
            x_norm, y_norm = pt
            if is_normalized_1000:
                x = round(x_norm / 1000.0, 6)
                y = round(y_norm / 1000.0, 6)
            elif is_absolute:
                x = round(x_norm / width, 6)
                y = round(y_norm / height, 6)                
            processed.append([x, y])
        elif len(pt) == 3:
            x_norm, y_norm, d_norm = pt
            if is_normalized_1000:
                x = round(x_norm / 1000.0, 6)
                y = round(y_norm / 1000.0, 6)
                d = d_norm
            elif is_absolute:
                x = round(x_norm / float(width), 6)
                y = round(y_norm / float(height), 6)
                d = d_norm
            processed.append([x, y, d])
    
    return np.array(processed)

def project_3d_to_2d(points_3d: np.ndarray, intrinsics: Dict[str, float]) -> np.ndarray:
    """
    Project 3D points (N,3) in camera coordinates to 2D pixel coordinates (N,2).
    u = fx*X/Z + cx, v = fy*Y/Z + cy
    """
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]

    # Avoid division by zero
    Z_safe = np.where(np.abs(Z) < 1e-6, 1e-6, Z)

    u = (X * fx / Z_safe) + cx
    v = (Y * fy / Z_safe) + cy
    return np.stack([u, v], axis=-1)


def interpolate_trajectory_by_distance(traj: np.ndarray, num_points: int) -> np.ndarray:
    """
    Distance-uniform interpolation for a 2D/3D polyline trajectory.

    Args:
      traj: (N,D)
      num_points: target number of points (>=2)

    Returns:
      (num_points,D)
    """
    traj = np.asarray(traj, dtype=np.float32)
    if traj.shape[0] < 2 or num_points <= 1:
        return traj

    deltas = np.diff(traj, axis=0)
    seg_lens = np.linalg.norm(deltas, axis=1)
    cum = np.insert(np.cumsum(seg_lens), 0, 0.0)
    total = float(cum[-1])

    if total <= 1e-12:
        return np.repeat(traj[:1], num_points, axis=0)

    targets = np.linspace(0.0, total, num_points, dtype=np.float32)

    out_dims = []
    for d in range(traj.shape[1]):
        f = interp1d(cum, traj[:, d], kind="linear")
        out_dims.append(f(targets))
    return np.stack(out_dims, axis=1)


def get_3d_bbox_corners(center, extent, rotation) -> np.ndarray:
    """
    Return 8 corners of a 3D oriented bounding box (OBB) in world/camera coords.
    extent is full size (not half size).
    """
    c = np.array(center, dtype=np.float32).reshape(3)
    e = np.array(extent, dtype=np.float32).reshape(3)
    R = np.array(rotation, dtype=np.float32).reshape(3, 3)

    half = e / 2.0
    local = np.array(
        [
            [-half[0], -half[1], -half[2]],
            [ half[0], -half[1], -half[2]],
            [-half[0],  half[1], -half[2]],
            [ half[0],  half[1], -half[2]],
            [-half[0], -half[1],  half[2]],
            [ half[0], -half[1],  half[2]],
            [-half[0],  half[1],  half[2]],
            [ half[0],  half[1],  half[2]],
        ],
        dtype=np.float32,
    )

    corners = local @ R.T + c
    return corners


def project_3d_bbox_to_2d(center, extent, rotation, intrinsics: Dict[str, float]) -> List[float]:
    """
    Project a 3D OBB to a 2D axis-aligned bbox in pixel coordinates: [u_min,v_min,u_max,v_max].
    """
    corners_3d = get_3d_bbox_corners(center, extent, rotation)
    corners_2d = project_3d_to_2d(corners_3d, intrinsics)
    u_min, v_min = np.min(corners_2d, axis=0)
    u_max, v_max = np.max(corners_2d, axis=0)
    return [float(u_min), float(v_min), float(u_max), float(v_max)]


def is_point_in_mask(point_uv: np.ndarray, mask: np.ndarray) -> bool:
    """
    Check whether a 2D pixel point (u,v) lies inside a binary mask (H,W).
    """
    u, v = float(point_uv[0]), float(point_uv[1])
    H, W = mask.shape[:2]

    ui, vi = int(round(u)), int(round(v))
    if not (0 <= vi < H and 0 <= ui < W):
        return False
    return mask[vi, ui] > 0


def is_point_in_2d_bbox(point_uv: np.ndarray, bbox_2d: List[float]) -> bool:
    """
    Check whether a 2D point (u,v) lies inside a 2D bbox [u_min,v_min,u_max,v_max].
    """
    u, v = float(point_uv[0]), float(point_uv[1])
    u_min, v_min, u_max, v_max = bbox_2d
    return (u_min <= u <= u_max) and (v_min <= v <= v_max)


def discrete_frechet_distance(P: np.ndarray, Q: np.ndarray) -> float:
    """Discrete Fréchet distance between two polylines."""
    n, m = len(P), len(Q)
    ca = np.full((n, m), -1.0, dtype=np.float32)
    dist = cdist(P, Q, "euclidean").astype(np.float32)

    def compute(i: int, j: int) -> float:
        if ca[i, j] > -0.5:
            return float(ca[i, j])
        if i == 0 and j == 0:
            ca[i, j] = dist[0, 0]
        elif i == 0:
            ca[i, j] = max(compute(0, j - 1), float(dist[i, j]))
        elif j == 0:
            ca[i, j] = max(compute(i - 1, 0), float(dist[i, j]))
        else:
            ca[i, j] = max(
                min(compute(i - 1, j), compute(i - 1, j - 1), compute(i, j - 1)),
                float(dist[i, j]),
            )
        return float(ca[i, j])

    return compute(n - 1, m - 1)


def hausdorff_distance(P: np.ndarray, Q: np.ndarray) -> float:
    """Hausdorff distance between two point sets."""
    return float(max(directed_hausdorff(P, Q)[0], directed_hausdorff(Q, P)[0]))


def calculate_rmse_mae(P: np.ndarray, Q: np.ndarray) -> Tuple[float, float]:
    """
    RMSE and MAE between two aligned sequences (same length).
    """
    P = np.asarray(P, dtype=np.float32)
    Q = np.asarray(Q, dtype=np.float32)
    dif = P - Q
    rmse = float(np.sqrt(np.mean(np.sum(dif * dif, axis=1))))
    mae = float(np.mean(np.mean(np.abs(dif), axis=1)))
    return rmse, mae


def calculate_metrics(pred_list: List[np.ndarray], gt_list: List[np.ndarray]) -> Dict[str, float]:
    """
    Compute average DFD / HD / RMSE / MAE over a list of trajectories.
    Each item must be (N,2) in normalized [0,1] coordinates.
    """
    dfd_list, hd_list, rmse_list, mae_list = [], [], [], []

    for i, (pred, gt) in enumerate(zip(pred_list, gt_list)):
        try:
            if pred is None or gt is None or len(pred) == 0 or len(gt) == 0:
                continue

            pred = np.asarray(pred, dtype=np.float32)
            gt = np.asarray(gt, dtype=np.float32)

            dfd = discrete_frechet_distance(pred, gt)
            hd = hausdorff_distance(pred, gt)
            rmse, mae = calculate_rmse_mae(pred, gt)

            # Optional outlier filter (kept from your logic)
            if dfd > 100:
                continue

            dfd_list.append(dfd)
            hd_list.append(hd)
            rmse_list.append(rmse)
            mae_list.append(mae)

        except Exception as e:
            print(f"[WARN] index={i} metric failed: {e}")
            continue

    return {
        "average_discrete_frechet_distance": float(np.mean(dfd_list)) if dfd_list else 0.0,
        "average_hausdorff_distance": float(np.mean(hd_list)) if hd_list else 0.0,
        "average_root_mean_square_error": float(np.mean(rmse_list)) if rmse_list else 0.0,
        "average_mean_absolute_error": float(np.mean(mae_list)) if mae_list else 0.0,
    }

# =========================
# Part 2: Loading + Utilities for 3D Spatial Trace
# =========================


import cv2
from collections import Counter
from types import SimpleNamespace
import numpy as np
import open3d as o3d


def pcd_denoise_dbscan(
    pcd: o3d.geometry.PointCloud,
    eps: float = 0.02,
    min_points: int = 10
) -> o3d.geometry.PointCloud:
    """
    DBSCAN-based denoising.
    This version safely handles point clouds without colors.
    """
    # Cluster labels (noise is labeled as -1)
    pcd_clusters = pcd.cluster_dbscan(
        eps=eps,
        min_points=min_points,
    )

    obj_points = np.asarray(pcd.points)

    # Check whether the point cloud has colors
    has_colors = pcd.has_colors()
    if has_colors:
        obj_colors = np.asarray(pcd.colors)

    pcd_clusters = np.array(pcd_clusters)

    # Count all cluster labels
    counter = Counter(pcd_clusters)

    # Remove noise label (-1)
    if counter and (-1 in counter):
        del counter[-1]

    if counter:
        # Keep the largest cluster
        most_common_label, _ = counter.most_common(1)[0]
        largest_mask = (pcd_clusters == most_common_label)

        largest_cluster_points = obj_points[largest_mask]

        if has_colors:
            largest_cluster_colors = obj_colors[largest_mask]

        # If the largest cluster is too small, return the original point cloud
        if len(largest_cluster_points) < 5:
            return pcd

        largest_cluster_pcd = o3d.geometry.PointCloud()
        largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)

        if has_colors:
            largest_cluster_pcd.colors = o3d.utility.Vector3dVector(largest_cluster_colors)

        pcd = largest_cluster_pcd

    return pcd


def process_pcd(cfg, pcd, run_dbscan: bool = True):
    """
    Basic point cloud cleanup:
      - Statistical outlier removal
      - Voxel downsampling
      - Optional DBSCAN noise removal
    """
    scale = np.linalg.norm(np.asarray(pcd.points).std(axis=0)) * 3.0 + 1e-6
    [pcd, _] = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.2)
    pcd = pcd.voxel_down_sample(voxel_size=max(0.01, scale / 40))

    if cfg.dbscan_remove_noise and run_dbscan:
        pcd = pcd_denoise_dbscan(pcd, eps=cfg.dbscan_eps, min_points=cfg.dbscan_min_points)

    return pcd


def create_object_pcd_from_mask(mask_path, depth_path, intrinsics_data):
    """
    Load only the point cloud region defined by the mask.
    [v3: uses the 'process_pcd' pipeline for filtering/denoising]
    """
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    if mask_img is None:
        print(f"❌ Failed to load mask: {mask_path}")
        return None
    if depth_img is None:
        print(f"❌ Failed to load depth: {depth_path}")
        return None

    # Ensure mask and depth have the same resolution
    if mask_img.shape != depth_img.shape:
        print(
            f"  -> [Warning] Mask/Depth shape mismatch. "
            f"Resizing mask {mask_img.shape} to {depth_img.shape}."
        )
        mask_img = cv2.resize(
            mask_img,
            (depth_img.shape[1], depth_img.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    depth_intrinsics = np.array(intrinsics_data)
    fx, fy = depth_intrinsics[0, 0], depth_intrinsics[1, 1]
    cx, cy = depth_intrinsics[0, 2], depth_intrinsics[1, 2]

    depth_scale = 1000.0  # typically mm->m for depth PNGs
    height, width = depth_img.shape

    v_coords, u_coords = np.mgrid[0:height, 0:width]
    u_coords, v_coords = u_coords.flatten(), v_coords.flatten()
    depth_values = depth_img.flatten()
    mask_values = mask_img.flatten()

    valid_mask = (mask_values > 0) & (depth_values > 0) & (depth_values < depth_scale * 15)

    u_valid, v_valid = u_coords[valid_mask], v_coords[valid_mask]
    depth_valid = depth_values[valid_mask]

    z = depth_valid / depth_scale
    x = (u_valid - cx) * z / fx
    y = (v_valid - cy) * z / fy  # y-down
    points_3d = np.vstack((x, y, z)).T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    # If empty, return directly
    if not pcd.has_points():
        return pcd

    # Build a minimal cfg for process_pcd
    cfg = SimpleNamespace()
    cfg.dbscan_remove_noise = True
    cfg.dbscan_eps = 0.02
    cfg.dbscan_min_points = 10

    processed_pcd = process_pcd(cfg, pcd, run_dbscan=True)

    # Return as numpy array
    return np.asarray(processed_pcd.points)


def backproject_to_3d(points, width, height, intrinsics):
    """
    Back-project normalized (x,y,d) where x,y in [0,1] and d is in meters.
    """
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    xyz = []
    for x, y, d in points:
        x = x * width
        y = y * height
        X = (x - cx) * d / fx
        Y = (y - cy) * d / fy
        Z = d
        xyz.append((X, Y, Z))
    return np.array(xyz)


def point_to_box_distance(point, bbox_center, bbox_extent, bbox_rotation):
    """
    Parameters:
    - point: (3,) in world coordinates
    - bbox_center: (3,) in world coordinates
    - bbox_extent: (3,) full side lengths [dx, dy, dz]
    - bbox_rotation: (3,3) rotation matrix in world coordinates
    """
    rel_point = point - bbox_center
    inv_rot = np.linalg.inv(bbox_rotation)
    local_point = inv_rot @ rel_point

    half_extent = 0.5 * np.array(bbox_extent)

    # Inside the box
    if np.all(np.abs(local_point) <= half_extent):
        return 0.0

    # Clamp to box boundary in local coordinates
    clamped_local = np.clip(local_point, -half_extent, half_extent)

    # Transform back to world coordinates
    closest_world = bbox_rotation @ clamped_local + bbox_center

    return np.linalg.norm(point - closest_world)


def create_occupancy_grid_from_tsdf(
    depth_image,
    object_mask,
    o3d_intrinsics,
    voxel_size: float = 0.02,
    depth_scale: float = 1000.0,
    depth_trunc: float = 5.0,
):
    """
    Create a 3D voxel occupancy grid from a depth image via TSDF fusion.
    object_mask: pixels with value 255 indicate the *movable object* and will be removed from the obstacle depth.

    [v2: compatible with older Open3D versions]
    """
    # 1) Prepare obstacle-only depth image by removing the object region
    depth_image_obstacle = np.copy(depth_image)
    depth_image_obstacle[object_mask == 255] = 0
    o3d_depth_obstacle = o3d.geometry.Image(depth_image_obstacle)

    # 2) Build Open3D intrinsics
    depth_height, depth_width = depth_image_obstacle.shape
    o3d_intrinsics = np.array(o3d_intrinsics)
    fx, fy = o3d_intrinsics[0, 0], o3d_intrinsics[1, 1]
    cx, cy = o3d_intrinsics[0, 2], o3d_intrinsics[1, 2]
    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(depth_width, depth_height, fx, fy, cx, cy)

    # 3) Create RGBDImage (use a dummy color image for compatibility)
    dummy_color = np.zeros((depth_height, depth_width, 3), dtype=np.uint8)
    o3d_color_dummy = o3d.geometry.Image(dummy_color)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color_dummy,
        o3d_depth_obstacle,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )

    # 4) TSDF integration
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=4 * voxel_size,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )

    camera_pose = np.eye(4)
    volume.integrate(rgbd_image, o3d_intrinsics, np.linalg.inv(camera_pose))

    obstacle_pcd_dense = volume.extract_point_cloud()
    if not obstacle_pcd_dense.has_points():
        raise ValueError("TSDF extracted no points. Please check depth image and intrinsics.")

    occupancy_grid_o3d = o3d.geometry.VoxelGrid.create_from_point_cloud(
        obstacle_pcd_dense,
        voxel_size=voxel_size,
    )

    voxels = occupancy_grid_o3d.get_voxels()
    if not voxels:
        raise ValueError("VoxelGrid is empty.")

    voxel_indices = np.array([v.grid_index for v in voxels])
    grid_origin_o3d = occupancy_grid_o3d.origin

    # Use a set for fast collision lookup
    occupied_voxels = set(tuple(idx) for idx in voxel_indices)

    # Return: lookup set, origin, voxel size, and the Open3D VoxelGrid (for visualization/debugging)
    return occupied_voxels, grid_origin_o3d, voxel_size, occupancy_grid_o3d


def calculate_trajectory_collisions(env_voxel_grid, object_points_np, pred_interp):
    """
    Compute per-step collision ratio between a moved object point cloud and the environment voxel grid.

    Parameters:
    - env_voxel_grid: return value of create_occupancy_grid_from_tsdf
                      (occupied_set, grid_origin, voxel_size, voxelgrid_o3d)
    - object_points_np: (M,3) numpy array of object points
    - pred_interp: (N,3) interpolated trajectory in 3D
    """
    # 1) Unpack env voxel data
    try:
        occupied_set, grid_origin, voxel_size, _ = env_voxel_grid
    except (TypeError, ValueError):
        print("❌ Error: 'env_voxel_grid' has an invalid format.")
        print("   It must be the return value of create_occupancy_grid_from_tsdf: (set, origin, size, grid).")
        return None

    if not occupied_set:
        print("  [Warning] Environment voxel grid is empty; all collision ratios will be 0.")

    total_object_points = len(object_points_np)

    # 2) Validate trajectory
    if pred_interp is None or len(pred_interp) < 1:
        print("❌ Error: interpolated trajectory (pred_interp) is empty.")
        return None

    # We assume pred_interp[0] corresponds to the reference position of object_points_np
    start_pos = pred_interp[0]
    collision_ratios = []

    # 3) Iterate along the trajectory
    for current_pos in pred_interp:
        # Translate object points to current pose
        translation = current_pos - start_pos
        translated_object_points = object_points_np + translation

        # Convert points to voxel grid indices
        all_grid_indices = ((translated_object_points - grid_origin) / voxel_size).astype(int)

        # Count collisions (point-level, not unique voxels)
        collision_count = 0
        for idx_tuple in map(tuple, all_grid_indices):
            if idx_tuple in occupied_set:
                collision_count += 1

        collision_ratios.append(collision_count / max(total_object_points, 1))

    return collision_ratios





# =========================
# Part 3: Compute Accuracy for 2D Visual Trace (Direct predict 2D points)
# =========================

def compute_accuracy_2d(output_save_folder, benchmark_folder, task_name, model_name) -> Dict[str, float]:

    answer_file = f"{output_save_folder}/{model_name}_{task_name}.jsonl"
    with open(answer_file, "r", encoding="utf-8") as f:
        answers = [json.loads(line) for line in f if line.strip()]

    pred_points_2d_pixels: List[np.ndarray] = []
    gt_points_2d_pixels: List[np.ndarray] = []
    image_dims_list: List[np.ndarray] = []  # [W, H] for each sample
    start_in_mask_flags: List[bool] = []
    end_in_bbox_flags: List[bool] = []

    max_len = 100  # number of points after interpolation

    for ans in tqdm(answers, desc="Evaluating (2D)"):
        # Paths (raw_data is assumed to be the root folder for image/mask/depth files)
        image_path = f"{benchmark_folder}/raw_data/{ans['image_path']}"
        gt_depth_path = f"{benchmark_folder}/raw_data/{ans['gt_depth_path']}"  # not used in pure 2D metrics, kept for consistency
        mask_path = f"{benchmark_folder}/raw_data/{ans['mask_path']}"

        # Load image to get (H,W)
        img = np.array(Image.open(image_path).convert("RGB"))
        H, W = img.shape[:2]
        image_dims_list.append(np.array([W, H], dtype=np.float32))

        # Intrinsics (prefer GT depth intrinsics for consistency with dataset definition)
        intrinsics_matrix = ans.get("gt_depth_intrinsics", None)
        intrinsics = extract_intrinsics_from_matrix(intrinsics_matrix)

        # Parse model prediction: normalized [0,1000] -> [0,1]
        if any(key in model_name for key in ["RoboTracer"]):
            pred_parsed = text2pts_normalize(
                ans.get("model_prediction", ""),
                width=W,
                height=H,
                is_absolute=False,
                is_normalized_1000=True,
            )
        elif any(key in model_name for key in ["Qwen", "Gemini"]):
            pred_parsed = json2pts_normalize(
                ans.get("model_prediction", ""),
                width=W,
                height=H,
                is_absolute=False,
                is_normalized_1000=True,
            )
        # Some outputs may be (N,3); for 2D evaluation we only use (x,y)
        if pred_parsed.ndim == 2 and pred_parsed.shape[1] >= 2:
            pred_xy_norm = pred_parsed[:, :2]
        else:
            pred_xy_norm = np.zeros((0, 2), dtype=np.float32)

        # Convert normalized [0,1] -> pixel coords
        pred_xy_pixels = pred_xy_norm * np.array([W, H], dtype=np.float32)

        # Ground-truth 3D trajectory -> 2D pixels via projection
        gt_3d = np.array(ans["trajectory"], dtype=np.float32)  # (N,3)
        gt_uv_pixels = project_3d_to_2d(gt_3d, intrinsics).astype(np.float32)

        # Interpolate both to a fixed length for path metrics
        if len(pred_xy_pixels) < 2:
            # Fallback: use a single point if model returned too few points
            pred_xy_pixels = np.repeat(pred_xy_pixels[:1], 2, axis=0) if len(pred_xy_pixels) == 1 else np.zeros((2, 2), dtype=np.float32)

        pred_interp = interpolate_trajectory_by_distance(pred_xy_pixels, max_len)
        gt_interp = interpolate_trajectory_by_distance(gt_uv_pixels, max_len)

        pred_points_2d_pixels.append(pred_interp)
        gt_points_2d_pixels.append(gt_interp)

        # Load target mask
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = mask.astype(np.uint8)

        # Start-in-mask rate (use the first predicted point in pixel coords)
        start_in_mask_flags.append(is_point_in_mask(pred_xy_pixels[0], mask))

        # End-in-bbox rate (use last 1-3 predicted points)
        bbox_center = ans.get("bbox_center", None)
        bbox_extent = ans.get("bbox_extent", None)
        bbox_rotation = ans.get("bbox_rotation", None)

        if bbox_center is None or bbox_extent is None or bbox_rotation is None:
            end_in_bbox_flags.append(False)
        else:
            bbox_2d = project_3d_bbox_to_2d(bbox_center, bbox_extent, bbox_rotation, intrinsics)
            end_pts = pred_xy_pixels[-3:] if len(pred_xy_pixels) >= 3 else pred_xy_pixels[-1:]
            end_ok = any(is_point_in_2d_bbox(p, bbox_2d) for p in end_pts)
            end_in_bbox_flags.append(end_ok)

    print("\n--- 2D Evaluation Results ---")

    # Normalize pixel trajectories to [0,1] for fair metrics across different resolutions
    valid_pred_norm: List[np.ndarray] = []
    valid_gt_norm: List[np.ndarray] = []

    for pred_pix, gt_pix, dims in zip(pred_points_2d_pixels, gt_points_2d_pixels, image_dims_list):
        if dims[0] <= 0 or dims[1] <= 0:
            continue
        pred_norm = pred_pix / dims
        gt_norm = gt_pix / dims
        if np.isnan(pred_norm).any() or np.isnan(gt_norm).any():
            continue
        valid_pred_norm.append(pred_norm)
        valid_gt_norm.append(gt_norm)

    metrics_2d = calculate_metrics(valid_pred_norm, valid_gt_norm) if valid_pred_norm else {}
    print(f"2D Path Metrics (Normalized [0,1]): {metrics_2d}")

    print(f"2D Start-in-Mask Rate: {float(np.mean(start_in_mask_flags)):.4f}")
    print(f"2D End-in-2D-BBox Rate: {float(np.mean(end_in_bbox_flags)):.4f}")





# =========================
# Part 4: Compute Accuracy for 3D Spatial Trace (Predict 3D points), we also derive 2D evaluation from 3D evaluation for the points projected from 3D to 2D.
# =========================
def compute_accuracy_3d(output_save_folder, benchmark_folder, task_name, model_name) -> Dict[str, float]:

    answer_file = f"{output_save_folder}/{model_name}_{task_name}.jsonl"

    max_len = 100  # number of points after interpolation
    start_thresh_m = 0.20
    end_thresh_m = 0.20
    collision_ratio_thresh = 0.20

    with open(answer_file, "r", encoding="utf-8") as f:
        answers = [json.loads(line) for line in f if line.strip()]

    # =========================
    # 2D evaluation accumulators
    # =========================
    pred_points_2d_pixels: List[np.ndarray] = []
    gt_points_2d_pixels: List[np.ndarray] = []
    image_dims_list: List[np.ndarray] = []  # [W, H] for each sample
    start_in_mask_flags: List[bool] = []
    end_in_bbox_flags: List[bool] = []

    # =========================
    # 3D evaluation accumulators
    # =========================
    pred_points_3d: List[np.ndarray] = []
    gt_points_3d: List[np.ndarray] = []
    start_distances: List[float] = []
    end_distances: List[float] = []
    collision_flags: List[bool] = []
    start_success_flags: List[bool] = []
    end_success_flags: List[bool] = []
    success_count = 0

    # Optional: export predicted 3D trajectories if you want to save them later
    pred_trajs_export = []

    for ans in tqdm(answers, desc="Evaluating (3D + derived 2D)"):
        qid = ans.get("question_id", ans.get("id", None))

        # Paths
        gt_depth_path = f"{benchmark_folder}/raw_data/{ans['gt_depth_path']}"
        mask_path = f"{benchmark_folder}/raw_data/{ans['mask_path']}"

        bbox_center = ans.get("bbox_center", None)
        bbox_extent = ans.get("bbox_extent", None)
        bbox_rotation = ans.get("bbox_rotation", None)

        # Load depth ONLY to get (H,W). Do NOT divide by 255.
        depth_img = np.array(Image.open(gt_depth_path))
        H, W = depth_img.shape[:2]

        # Intrinsics (prefer GT depth intrinsics for consistency)
        intrinsics_matrix = ans.get("gt_depth_intrinsics", None)
        if not intrinsics_matrix:
            print(f"[Skip] No intrinsics for question {qid}")
            continue

        try:
            intrinsics = extract_intrinsics_from_matrix(intrinsics_matrix)
        except Exception as e:
            print(f"[Skip] Failed to extract intrinsics for question {qid}: {e}")
            continue

        # Parse model prediction: normalized [0,1000] -> [0,1] for x,y
        try:
            if any(key in model_name for key in ["RoboTracer"]):
                pred_parsed = text2pts_normalize(
                    ans.get("model_prediction", ""),
                    width=W,
                    height=H,
                    is_normalized_1000=True,
                )
            elif any(key in model_name for key in ["Qwen", "Gemini"]):
                pred_parsed = json2pts_normalize(
                    ans.get("model_prediction", ""),
                    width=W,
                    height=H,
                    is_absolute=False,
                    is_normalized_1000=True,
                )
            else:
                raise ValueError(f"Unsupported model: {model_name}")
        except Exception as e:
            print(f"[Skip] Failed to parse prediction for question {qid}: {e}")
            continue

        # Must have depth d for 3D backprojection: (N,3)
        if pred_parsed.ndim != 2 or pred_parsed.shape[1] < 3:
            print(f"[Skip] Prediction has no depth (needs (x,y,d)) for question {qid}")
            continue

        # Backproject predicted (x,y,d) into 3D camera space
        try:
            pred_3d = backproject_to_3d(pred_parsed[:, :3], W, H, intrinsics)
            pred_trajs_export.append({"id": qid, "trajectory": pred_3d.tolist()})
        except Exception as e:
            print(f"[Skip] Backprojection failed for question {qid}: {e}")
            continue

        # Ground-truth 3D trajectory
        gt_3d = np.array(ans["trajectory"], dtype=np.float32)  # (N,3)

        # Interpolate both in 3D for path metrics
        pred_interp_3d = interpolate_trajectory_by_distance(pred_3d, max_len)
        gt_interp_3d = interpolate_trajectory_by_distance(gt_3d, max_len)

        pred_points_3d.append(pred_interp_3d)
        gt_points_3d.append(gt_interp_3d)

        # =========================================================
        # Derived 2D evaluation (project 3D -> 2D)
        # =========================================================
        try:
            # Project interpolated 3D trajectories to 2D pixels
            pred_interp_2d = project_3d_to_2d(pred_interp_3d, intrinsics).astype(np.float32)
            gt_interp_2d = project_3d_to_2d(gt_interp_3d, intrinsics).astype(np.float32)

            pred_points_2d_pixels.append(pred_interp_2d)
            gt_points_2d_pixels.append(gt_interp_2d)
            image_dims_list.append(np.array([W, H], dtype=np.float32))

            # Load target mask
            mask = np.array(Image.open(mask_path))
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = mask.astype(np.uint8)

            # Start-in-mask: use the first predicted raw 2D point
            pred_2d_raw = project_3d_to_2d(pred_3d, intrinsics).astype(np.float32)
            start_in_mask_flags.append(is_point_in_mask(pred_2d_raw[0], mask))

            # End-in-2D-BBox: project destination 3D bbox to 2D bbox and check last 1-3 predicted points
            if bbox_center is None or bbox_extent is None or bbox_rotation is None:
                end_in_bbox_flags.append(False)
            else:
                bbox_2d = project_3d_bbox_to_2d(bbox_center, bbox_extent, bbox_rotation, intrinsics)
                end_pts = pred_2d_raw[-3:] if len(pred_2d_raw) >= 3 else pred_2d_raw[-1:]
                end_ok = any(is_point_in_2d_bbox(p, bbox_2d) for p in end_pts)
                end_in_bbox_flags.append(bool(end_ok))

        except Exception as e:
            print(f"[Warn][2D] Derived 2D evaluation failed for question {qid}: {e}")
            pred_points_2d_pixels.append(np.full((max_len, 2), np.nan, dtype=np.float32))
            gt_points_2d_pixels.append(np.full((max_len, 2), np.nan, dtype=np.float32))
            image_dims_list.append(None)
            start_in_mask_flags.append(False)
            end_in_bbox_flags.append(False)

        # =========================================================
        # 3D start/end distances + collision + success
        # =========================================================

        # Start distance: pred start to target object point cloud (or to GT start)
        start_distance = None
        try:
            obj_points = create_object_pcd_from_mask(mask_path, gt_depth_path, intrinsics_matrix)
            if obj_points is None or len(obj_points) == 0:
                raise ValueError("Empty object point cloud.")

            pred_start = pred_3d[0]
            gt_start = gt_3d[0]

            d_obj = float(np.min(np.linalg.norm(obj_points - pred_start, axis=1)))
            d_gt = float(np.linalg.norm(pred_start - gt_start))
            start_distance = float(min(d_obj, d_gt))
            start_distances.append(start_distance)
        except Exception as e:
            print(f"[Warn][3D] Start point distance failed for question {qid}: {e}")

        # End distance: pred end to destination 3D bbox (use last 1-3 predicted points)
        end_distance = None
        try:
            if bbox_center is None or bbox_extent is None or bbox_rotation is None:
                raise ValueError("Missing destination bbox fields.")

            bc = np.array(bbox_center, dtype=np.float32)
            be = np.array(bbox_extent, dtype=np.float32)
            br = np.array(bbox_rotation, dtype=np.float32).reshape(3, 3)

            end_candidates_3d = [pred_3d[-1], pred_3d[-2], pred_3d[-3]] if len(pred_3d) >= 3 else [pred_3d[-1]]
            end_distance = float(min(point_to_box_distance(p, bc, be, br) for p in end_candidates_3d))
            end_distances.append(end_distance)
        except Exception as e:
            print(f"[Warn][3D] End point distance failed for question {qid}: {e}")

        # Collision analysis
        collision_flag = False
        try:
            depth_f = np.array(Image.open(gt_depth_path)).astype(np.float32)
            mask_u8 = np.array(Image.open(mask_path)).astype(np.uint8)
            if mask_u8.ndim == 3:
                mask_u8 = mask_u8[:, :, 0]

            env_voxel_grid = create_occupancy_grid_from_tsdf(depth_f, mask_u8, intrinsics_matrix)
            collision_ratios = calculate_trajectory_collisions(env_voxel_grid, obj_points, pred_3d)

            if collision_ratios is None:
                collision_flag = False
            else:
                collision_flag = any(r > collision_ratio_thresh for r in collision_ratios)

        except Exception as e:
            print(f"[Warn][3D] Collision check failed for question {qid}: {e}")
            collision_flag = False

        collision_flags.append(bool(collision_flag))

        # Success logic
        start_success = (start_distance is not None) and (start_distance < start_thresh_m)
        end_success = (end_distance is not None) and (end_distance < end_thresh_m)

        start_success_flags.append(bool(start_success))
        end_success_flags.append(bool(end_success))

        if start_success and end_success and (not collision_flag):
            success_count += 1


    # =========================
    # 2D report (same style as your 2D script)
    # =========================
    print("\n--- 2D Evaluation Results (Derived from 3D) ---")

    valid_pred_norm: List[np.ndarray] = []
    valid_gt_norm: List[np.ndarray] = []

    for pred_pix, gt_pix, dims in zip(pred_points_2d_pixels, gt_points_2d_pixels, image_dims_list):
        if dims is None or dims[0] <= 0 or dims[1] <= 0:
            continue
        pred_norm = pred_pix / dims
        gt_norm = gt_pix / dims
        if np.isnan(pred_norm).any() or np.isnan(gt_norm).any():
            continue
        valid_pred_norm.append(pred_norm)
        valid_gt_norm.append(gt_norm)

    metrics_2d = calculate_metrics(valid_pred_norm, valid_gt_norm) if valid_pred_norm else {}
    print(f"2D Path Metrics (Normalized [0,1]): {metrics_2d}")

    print(f"2D Start-in-Mask Rate: {float(np.mean(start_in_mask_flags)):.4f}")
    print(f"2D End-in-2D-BBox Rate: {float(np.mean(end_in_bbox_flags)):.4f}")


    # =========================
    # 3D report
    # =========================
    print("\n--- 3D Evaluation Results ---")

    metrics_3d = calculate_metrics(pred_points_3d, gt_points_3d) if pred_points_3d else {}
    print(f"3D Path Metrics: {metrics_3d}")

    if start_distances:
        print(f"Start Point Distance (mean): {float(np.mean(start_distances)):.4f} m")
    else:
        print("Start Point Distance (mean): N/A")

    if end_distances:
        print(f"End Point Distance (mean): {float(np.mean(end_distances)):.4f} m")
    else:
        print("End Point Distance (mean): N/A")

    print(f"No-Collision Rate: {float(1.0 - np.mean(collision_flags)):.4f}" if collision_flags else "No-Collision Rate: N/A")
    print(f"Start Success Rate (<{start_thresh_m:.2f}m): {float(np.mean(start_success_flags)):.4f}" if start_success_flags else "Start Success Rate: N/A")
    print(f"End Success Rate (<{end_thresh_m:.2f}m): {float(np.mean(end_success_flags)):.4f}" if end_success_flags else "End Success Rate: N/A")

    denom = max(len(answers), 1)
    print(f"Overall Success Rate: {success_count / denom:.4f}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    args = parser.parse_args()

    output_save_folder = './outputs'
    benchmark_folder = './TraceSpatial-Bench'
        
    if any(key in args.model_name for key in ["RoboTracer", "Qwen", "Gemini"]):
        if args.task_name == "2D":
            print(f"Computing accuracy for 2D Visual Trace (Direct predict 2D points, we report this metric in the paper)")
            compute_accuracy_2d(output_save_folder, benchmark_folder, "2D", args.model_name)
        elif args.task_name == "3D":
            print(f"Computing accuracy for 3D Spatial Trace (Direct predict 3D points, we report this metric in the paper), we also derive 2D evaluation from 3D evaluation for the points projected from 3D to 2D (this 2D evaluation is just for checking the consistency of the 2D and 3D evaluation, we do not report this metric in the paper).")
            compute_accuracy_3d(output_save_folder, benchmark_folder, "3D", args.model_name)
        elif str(args.task_name).lower() == "all":
            print(f"Computing accuracy for 2D Visual Trace (Direct predict 2D points, we report this metric in the paper)")
            compute_accuracy_2d(output_save_folder, benchmark_folder, "2D", args.model_name)
            print(f"Computing accuracy for 3D Spatial Trace (Direct predict 3D points, we report this metric in the paper), we also derive 2D evaluation from 3D evaluation for the points projected from 3D to 2D (this 2D evaluation is just for checking the consistency of the 2D and 3D evaluation, we do not report this metric in the paper).")
            compute_accuracy_3d(output_save_folder, benchmark_folder, "3D", args.model_name)
        else:
            print(f"Unknown task name: {args.task_name}")
    else:
        print(f"Unknown model type: {args.model_name}")

if __name__ == '__main__':
    main()