#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
"""
Path renderer for 3DGS models with optical flow generation.
Generates camera paths between two cameras and renders RGB, optical flow, and visualizations.
"""

import os
import sys
import math
import numpy as np
import torch
import cv2
from tqdm import tqdm
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree as KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx

# Project imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene import Scene
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from utils import g_utils
from utils.general_utils import safe_state
from argparse import ArgumentParser


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def writeFlowKITTI(filename, uv):
    """Write optical flow in KITTI format."""
    uv = 64.0 * uv + 2**15
    valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, uv[..., ::-1])


def flow_to_image(flow):
    """Convert optical flow to RGB visualization using HSV color space."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = np.clip((ang / 2.0), 0, 180).astype(np.uint8)
    hsv[..., 1] = np.clip(mag * 8.0, 0, 255).astype(np.uint8)
    hsv[..., 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


# ============================================================================
# ROTATION UTILITIES
# ============================================================================

def rotation_matrix_to_quaternion(R):
    """Convert rotation matrix to quaternion [w, x, y, z]."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    q = np.array([w, x, y, z])
    return q / (np.linalg.norm(q) + 1e-8)


def quaternion_to_rotation_matrix(q):
    """Convert quaternion [w, x, y, z] to rotation matrix."""
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    return R


def rotation_matrix_to_axis_angle(R):
    """Convert rotation matrix to axis-angle representation."""
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if angle < 1e-6:
        return np.array([0, 0, 1]), 0.0
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis = axis / (2 * np.sin(angle) + 1e-8)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    return axis, angle


def axis_angle_to_rotation_matrix(axis, angle):
    """Convert axis-angle to rotation matrix (Rodrigues formula)."""
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    K = np.array([[0, -axis[2], axis[1]], 
                  [axis[2], 0, -axis[0]], 
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def gaussian_smooth_quaternions(quaternions, sigma=4.0):
    """
    Apply Gaussian smoothing to quaternion sequence.
    Ensures quaternions stay on unit sphere and handles sign flips.
    """
    n = len(quaternions)
    if n < 3:
        return quaternions.copy()
    
    # Ensure consistent quaternion signs (avoid double-cover issue)
    aligned_quats = quaternions.copy()
    for i in range(1, n):
        if np.dot(aligned_quats[i], aligned_quats[i-1]) < 0:
            aligned_quats[i] = -aligned_quats[i]
    
    # Smooth each component independently
    smoothed = np.zeros_like(aligned_quats)
    for j in range(4):
        smoothed[:, j] = gaussian_filter1d(aligned_quats[:, j], sigma=sigma, mode='nearest')
    
    # Re-normalize to unit quaternions
    norms = np.linalg.norm(smoothed, axis=1, keepdims=True)
    smoothed = smoothed / (norms + 1e-8)
    
    return smoothed


# ============================================================================
# FRUSTUM CARVING - COMPUTE VISIBLE VOLUME
# ============================================================================

def compute_visible_volume(camera, gaussians, pipeline, background, kernel_size=0.0, 
                          max_dist=10.0, num_rays=1000, seed=None):
    """
    Compute point cloud representing the visible empty volume of a camera.
    Uses depth map to stop at obstacles.
    """
    rng = np.random.default_rng(seed)
    
    with torch.no_grad():
        rendering = render(camera, gaussians, pipeline, background, kernel_size)
    
    depth_map = rendering["depthmid"].squeeze().cpu().numpy()
    H, W = depth_map.shape
    
    # Sample random rays
    us = rng.integers(0, W, size=num_rays)
    vs = rng.integers(0, H, size=num_rays)
    
    depths = depth_map[vs, us]
    depths[depths == 0] = max_dist
    depths = np.minimum(depths, max_dist)
    
    # Intrinsics for reprojection
    fovx = camera.FoVx
    fovy = camera.FoVy
    tan_fovx = np.tan(fovx / 2.0)
    tan_fovy = np.tan(fovy / 2.0)
    
    u_norm = (us + 0.5) / W
    v_norm = (vs + 0.5) / H
    
    # Ray directions in camera space
    x_cam = (2.0 * u_norm - 1.0) * tan_fovx
    y_cam = (2.0 * v_norm - 1.0) * tan_fovy
    z_cam = np.ones_like(x_cam)
    
    rays_dir_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    
    # Sample points along rays
    z_near = 0.2
    min_samples = 4
    max_samples = 200
    
    ray_lengths = np.clip(depths - z_near, 0.0, max_dist - z_near)
    norm_len = ray_lengths / (max_dist - z_near + 1e-12)
    per_ray_counts = np.clip(
        np.round(min_samples + norm_len * (max_samples - min_samples)).astype(int),
        min_samples, max_samples
    )
    
    pts_list = []
    for i_ray in range(len(rays_dir_cam)):
        n_samp = int(per_ray_counts[i_ray])
        if n_samp <= 0:
            continue
        ratios = rng.uniform(0.0, 1.0, size=n_samp)
        zs = z_near + ratios * (depths[i_ray] - z_near)
        pts_cam_i = rays_dir_cam[i_ray][None, :] * zs[:, None]
        pts_list.append(pts_cam_i)
    
    if len(pts_list) == 0:
        return np.zeros((0, 3))
    
    pts_cam = np.vstack(pts_list)
    
    # Cap total points
    total_cap = 2_000_000
    if len(pts_cam) > total_cap:
        idx = np.random.choice(len(pts_cam), total_cap, replace=False)
        pts_cam = pts_cam[idx]
    
    # Transform to world coordinates
    ones = np.ones((pts_cam.shape[0], 1))
    pts_cam_h = np.hstack([pts_cam, ones])
    
    w2c = camera.world_view_transform.cpu().numpy()
    c2w = np.linalg.inv(w2c)
    
    pts_world_h = pts_cam_h @ c2w
    return pts_world_h[:, :3]


def compute_global_visible_volume(scene, gaussians, pipeline, background, kernel_size=0.0, 
                                 max_dist=10.0, num_rays_per_cam=1000, seed=None):
    """Compute union of visible volumes from all training cameras."""
    print("\nComputing global visible volume (frustum carving)...")
    cameras = scene.getTrainCameras()
    all_points = []
    
    n_cams = len(cameras)
    rays_to_use = num_rays_per_cam
    if n_cams > 100:
        rays_to_use = max(100, int(num_rays_per_cam * 100 / n_cams))
        print(f"  Using {rays_to_use} rays per camera ({n_cams} cameras)")
        
    for i, cam in enumerate(tqdm(cameras, desc="Carving frustums")):
        pts = compute_visible_volume(
            cam, gaussians, pipeline, background, kernel_size,
            max_dist=max_dist,
            num_rays=rays_to_use,
            seed=seed if seed is None else seed + i
        )
        all_points.append(pts)
        
    global_cloud = np.vstack(all_points)
    
    # Subsample to keep KDTree size reasonable
    if len(global_cloud) > 2_000_000:
        print(f"  Subsampling from {len(global_cloud)} to 2M points")
        idx = np.random.choice(len(global_cloud), 2_000_000, replace=False)
        global_cloud = global_cloud[idx]
    
    return global_cloud


# ============================================================================
# PATH GENERATION
# ============================================================================

def plan_safe_random_path(cameras, safe_points, safe_v_tree, num_steps=200, 
                         safety_threshold=0.3, noise_scale=1.0, noise_rot_deg=40.0, seed=None):
    """
    Generate smooth randomized path between two cameras using A* pathfinding.
    Stays within safe volume and uses density-based costs.
    """
    rng = np.random.default_rng(seed)
    n_cams = len(cameras)
    
    # Extract camera centers
    centers = []
    for cam in cameras:
        w2c = cam.world_view_transform.transpose(0, 1).cpu().numpy()
        c2w = np.linalg.inv(w2c)
        centers.append(c2w[:3, 3])
    centers = np.array(centers)
    
    # Calculate scene diagonal and minimum distance
    bbox_min = centers.min(axis=0)
    bbox_max = centers.max(axis=0)
    diag = np.linalg.norm(bbox_max - bbox_min)
    min_dist = 0.1 * diag
    
    # Pick two random cameras with sufficient distance
    i0, i1 = None, None
    for _ in range(200):
        a, b = rng.choice(n_cams, size=2, replace=False)
        if np.linalg.norm(centers[a] - centers[b]) >= min_dist:
            i0, i1 = a, b
            break
    if i0 is None:
        dists = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
        i0, i1 = np.unravel_index(np.argmax(dists), dists.shape)

    p0, R0 = centers[i0], np.linalg.inv(cameras[i0].world_view_transform.transpose(0, 1).cpu().numpy())[:3, :3]
    p1, R1 = centers[i1], np.linalg.inv(cameras[i1].world_view_transform.transpose(0, 1).cpu().numpy())[:3, :3]

    # Build navigation graph
    print(f"  Building navigation graph from {len(safe_points)} points...")
    max_nodes = 15000
    if len(safe_points) > max_nodes:
        nodes_idx = np.random.choice(len(safe_points), max_nodes, replace=False)
        nodes = safe_points[nodes_idx]
    else:
        nodes = safe_points
    
    # Add start and end points
    nodes = np.vstack([p0, nodes, p1])
    p0_idx = 0
    p1_idx = len(nodes) - 1
    
    node_tree = KDTree(nodes)
    G = nx.Graph()
    
    # Calculate density-based weights
    density_radius = diag * 0.05
    neighbor_counts = node_tree.query_ball_point(nodes, r=density_radius, return_length=True)
    
    max_density = np.max(neighbor_counts) if len(neighbor_counts) > 0 else 1.0
    densities = neighbor_counts / (max_density + 1e-6)
    
    # Connect nearby nodes
    search_radius = diag * 0.16
    pairs = node_tree.query_pairs(r=search_radius)
    
    for u, v in tqdm(pairs, desc="Building graph edges", leave=False):
        dist = np.linalg.norm(nodes[u] - nodes[v])
        avg_density = (densities[u] + densities[v]) / 2.0
        safety_multiplier = 1.0 + 5.0 * np.exp(-avg_density * 3.0)
        weight = dist * safety_multiplier * (1.0 + rng.uniform(0, noise_scale * 0.5))
        G.add_edge(u, v, weight=weight)
        
    # Sample waypoints
    print("  Sampling intermediate waypoints...")
    safe_indices = np.where(densities < 0.1)[0]
    if len(safe_indices) < 3:
        safe_indices = np.arange(len(nodes))

    num_wp = min(6, max(3, len(safe_indices)))
    wp_indices = rng.choice(safe_indices, size=num_wp, replace=False)
    navigation_order = [p0_idx] + list(wp_indices) + [p1_idx]
    sampled_waypoints = nodes[wp_indices]
    
    # A* pathfinding through waypoints
    all_path_nodes = []
    print(f"  Planning path through {len(navigation_order)} waypoints...")
    
    for i in range(len(navigation_order) - 1):
        start_node = navigation_order[i]
        end_node = navigation_order[i+1]
        
        try:
            segment_indices = nx.astar_path(
                G, start_node, end_node, 
                heuristic=lambda u, v: np.linalg.norm(nodes[u] - nodes[v]),
                weight='weight'
            )
            segment_nodes = nodes[segment_indices]
            if i > 0:
                segment_nodes = segment_nodes[1:]
            all_path_nodes.append(segment_nodes)
        except nx.NetworkXNoPath:
            print(f"  [WARNING] No path found, using linear fallback")
            segment_nodes = np.linspace(nodes[start_node], nodes[end_node], 5)
            if i > 0:
                segment_nodes = segment_nodes[1:]
            all_path_nodes.append(segment_nodes)

    path_nodes = np.vstack(all_path_nodes)

    # Smooth with cubic spline
    dists = np.zeros(len(path_nodes))
    dists[1:] = np.cumsum(np.linalg.norm(np.diff(path_nodes, axis=0), axis=1))
    
    cs = CubicSpline(dists, path_nodes, bc_type='clamped')
    t_smooth = np.linspace(0, dists[-1], num_steps)
    smooth_positions = cs(t_smooth)
    
    # Safety correction
    print("  Applying safety correction...")
    final_positions = []
    
    for pos in smooth_positions:
        dists_to_safe, idxs = safe_v_tree.query(pos, k=10)
        closest_dist = dists_to_safe[0]
        
        if closest_dist > safety_threshold * 0.3:
            local_safe_center = np.mean(safe_points[idxs], axis=0)
            pos = 0.7 * local_safe_center + 0.3 * pos
            
            new_dist, new_idx = safe_v_tree.query(pos)
            if new_dist > safety_threshold * 0.5:
                pos = safe_points[new_idx]
                  
        final_positions.append(pos)
    
    final_positions = np.array(final_positions)
    final_positions[0] = p0
    final_positions[-1] = p1
    
    # Interpolate rotations
    R_rel = R1 @ R0.T
    rel_axis, rel_angle = rotation_matrix_to_axis_angle(R_rel)
    if rel_axis is None:
        rel_axis, rel_angle = np.array([1.0, 0.0, 0.0]), 0.0
    
    t = np.linspace(0.0, 1.0, num_steps)
    profile = 4.0 * (t * (1.0 - t))

    # Smooth rotational perturbation
    m_ctrl = max(4, min(20, num_steps // 20))
    t_ctrl = np.linspace(0.0, 1.0, m_ctrl)
    angle_ctrl = rng.normal(scale=noise_rot_deg, size=m_ctrl)
    cs_angle = CubicSpline(t_ctrl, angle_ctrl)

    axis_ctrl = rng.normal(size=(m_ctrl, 3))
    axis_ctrl = axis_ctrl / (np.linalg.norm(axis_ctrl, axis=1)[:, None] + 1e-12)
    cs_axis_x = CubicSpline(t_ctrl, axis_ctrl[:, 0])
    cs_axis_y = CubicSpline(t_ctrl, axis_ctrl[:, 1])
    cs_axis_z = CubicSpline(t_ctrl, axis_ctrl[:, 2])

    angles_smooth = cs_angle(t) * profile

    c2ws = []
    for i in range(num_steps):
        ti = t[i]
        R_interp = axis_angle_to_rotation_matrix(rel_axis, rel_angle * ti) @ R0
        
        a_x = cs_axis_x(ti)
        a_y = cs_axis_y(ti)
        a_z = cs_axis_z(ti)
        rand_axis = np.array([a_x, a_y, a_z], dtype=float)
        rand_axis /= (np.linalg.norm(rand_axis) + 1e-12)

        rot_pert_angle = np.deg2rad(angles_smooth[i])
        if abs(rot_pert_angle) < 1e-12:
            R_pert = np.eye(3)
        else:
            R_pert = axis_angle_to_rotation_matrix(rand_axis, rot_pert_angle)

        R_final = R_pert @ R_interp

        if i == 0:
            R_final = R0.copy()
        elif i == num_steps - 1:
            R_final = R1.copy()

        c2w = np.zeros((3, 4), dtype=np.float32)
        c2w[:3, :3] = R_final.astype(np.float32)
        c2w[:3, 3] = final_positions[i].astype(np.float32)
        c2ws.append(c2w)
    
    return {
        "indices": (i0, i1), 
        "positions": final_positions, 
        "c2ws": c2ws, 
        "sampled_waypoints": sampled_waypoints,
        "graph_nodes": nodes, 
        "densities": densities
    }


# ============================================================================
# COVERAGE EVALUATION AND CORRECTION
# ============================================================================

def evaluate_view_coverage(camera, gaussians, pipeline, background, kernel_size, coverage_threshold=0.1):
    """Evaluate coverage quality of a camera view."""
    bg_black = torch.zeros((3), dtype=torch.float32, device="cuda")
    override_color = torch.ones((gaussians.get_xyz.shape[0], 3), dtype=torch.float32, device="cuda")
    
    with torch.no_grad():
        coverage_out = render(camera, gaussians, pipeline, bg_black, kernel_size, override_color=override_color)
        coverage_map = coverage_out['render'].mean(dim=0)
        empty_mask = coverage_map < coverage_threshold
        empty_percent = empty_mask.float().mean().item() * 100.0
    
    return empty_percent

def calculate_mean_scale(camera, gaussians, coverage_out):
    """Compute mean gaussian splat scale for visible gaussians."""
    mean_scale = 0.0
    max_scale = 0.0
    avg_top10 = 0.0
    try:
        # 1. Get potential visible gaussians (frustum culled)
        vis_mask = coverage_out.get('visibility_filter', None)
        
        if vis_mask is not None and vis_mask.any():
            # Filter gaussians
            xyz = gaussians.get_xyz[vis_mask]
            scales = gaussians.get_scaling[vis_mask].max(dim=1).values # Use max scale dimension
            
            # 2. Project to screen to check occlusion
            full_proj = camera.full_proj_transform
            ones = torch.ones((xyz.shape[0], 1), device=xyz.device)
            xyz_hom = torch.cat([xyz, ones], dim=1)
            p_hom = xyz_hom @ full_proj
            p_w = p_hom[:, 3:4]
            
            # Transform to Camera space for linear depth check
            w2c = camera.world_view_transform
            p_cam = xyz_hom @ w2c
            z_cam = p_cam[:, 2] # Linear depth Z
            
            # NDC coordinates (-1 to 1)
            p_ndc = p_hom[:, :3] / (p_w + 1e-7)
            
            # Screen coordinates
            coverage_map = coverage_out['render'].mean(dim=0)
            H, W = coverage_map.shape
            u = ((p_ndc[:, 0] + 1) * 0.5 * W).long()
            v = ((p_ndc[:, 1] + 1) * 0.5 * H).long()
            
            # Filter valid screen range
            valid_screen = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z_cam > 0.2)
            
            if valid_screen.any():
                u = u[valid_screen]
                v = v[valid_screen]
                z_cam = z_cam[valid_screen]
                scales_valid = scales[valid_screen]
                
                # rendering['depthmid'] is the rendered depth map
                rendered_depth = coverage_out['depthmid'][v, u]
                
                # Check occlusion: if point depth is close to rendered surface depth
                is_visible = torch.abs(z_cam - rendered_depth) < 0.2
                
                if is_visible.any():
                    visible_scales = scales_valid[is_visible]
                    mean_scale = float(visible_scales.mean().item())
                    max_scale = float(visible_scales.max().item())
                    k = max(1, int(0.1 * visible_scales.shape[0]))
                    topk = torch.topk(visible_scales, k).values
                    avg_top10 = float(topk.mean().item())

    except Exception as e:
        print(f"Error computing mean scale: {e}")
        mean_scale = 0.0
        max_scale = 0.0
        avg_top10 = 0.0

    return mean_scale, max_scale, avg_top10


def evaluate_view_metrics(camera, gaussians, pipeline, background, kernel_size, coverage_threshold=0.1):
    """Evaluate coverage quality and mean scale of a camera view."""
    bg_black = torch.zeros((3), dtype=torch.float32, device="cuda")
    override_color = torch.ones((gaussians.get_xyz.shape[0], 3), dtype=torch.float32, device="cuda")
    
    with torch.no_grad():
        coverage_out = render(camera, gaussians, pipeline, bg_black, kernel_size, override_color=override_color)
        coverage_map = coverage_out['render'].mean(dim=0)
        empty_mask = coverage_map < coverage_threshold
        empty_percent = empty_mask.float().mean().item() * 100.0
        
        mean_scale, max_scale, avg_top10 = calculate_mean_scale(camera, gaussians, coverage_out)
    
    return empty_percent, mean_scale, max_scale, avg_top10


def sample_alternative_rotations(base_rotation, num_samples=20, max_angle_deg=45.0, rng=None):
    """Sample alternative rotations around a base rotation."""
    if rng is None:
        rng = np.random.default_rng()
    
    alternatives = []
    max_angle_rad = np.deg2rad(max_angle_deg)
    
    for _ in range(num_samples):
        axis = rng.normal(size=3)
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        angle = rng.uniform(-max_angle_rad, max_angle_rad)
        R_pert = axis_angle_to_rotation_matrix(axis, angle)
        R_new = R_pert @ base_rotation
        alternatives.append(R_new)
    
    return alternatives


def rotation_distance(R1, R2):
    """Compute angular distance between two rotation matrices in degrees."""
    R_diff = R1 @ R2.T
    tr = np.trace(R_diff)
    cos_theta = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_theta))


def quaternion_slerp(q1, q2, t):
    """Spherical linear interpolation between two quaternions."""
    # Ensure unit quaternions
    q1 = q1 / (np.linalg.norm(q1) + 1e-8)
    q2 = q2 / (np.linalg.norm(q2) + 1e-8)
    
    dot = np.dot(q1, q2)
    
    # If dot is negative, slerp won't take the shorter path. Fix by negating one quaternion.
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # If inputs are too close, use linear interpolation
        result = q1 + t * (q2 - q1)
        return result / (np.linalg.norm(result) + 1e-8)
    
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    
    return s0 * q1 + s1 * q2


def correct_trajectory_for_coverage(path_data, gaussians, scene, pipeline, background, 
                                    kernel_size, max_empty_percent=0.1, max_mean_scale=0.04,
                                    coverage_threshold=0.1, num_rotation_samples=20,
                                    max_rotation_perturbation_deg=45.0, smoothing_sigma=4.0,
                                    seed=None):
    """
    Correct camera rotations to avoid empty space/high scale while matching future constraints.
    Constraint-Based Backward Propagation:
    1. Iterate backwards from end to start.
    2. At each frame, find SAFE rotations.
    3. From safe rotations, filter those 'Reachable' from the next frame's correction (if valid).
    4. Pick the candidate closest to the ORIGINAL rotation to maintain 'randomish' feel.
    """
    rng = np.random.default_rng(seed)
    template_cam = scene.getTrainCameras()[0]
    n_frames = len(path_data)
    
    print(f"\\nCorrecting trajectory for coverage (max empty: {max_empty_percent}%, max mean scale: {max_mean_scale})...")
    print("  Using Constraint-Based Backward Propagation...")
    
    positions = np.array([pos for pos, _ in path_data])
    rotations = [c2w[:3, :3] for _, c2w in path_data]
    
    # Storage for corrected rotations (we build this backwards)
    # We use a dict or list index able to handle reverse order
    corrected_rotations_map = {}
    
    next_corrected_R = None # The finalized rotation for frame i+1
    next_corrected_q = None # Quaternion of next_corrected_R
    
    # Max rotation change allowed per frame to be considered "smooth" transition
    # 5-10 degrees per frame is reasonable for 200-300 frame paths
    max_step_deg = 8.0 
    
    corrections_made = 0
    
    # Iterate backwards
    for i in tqdm(range(n_frames - 1, -1, -1), desc="Evaluating coverage (reverse)"):
        pos = positions[i]
        R_orig = rotations[i]
        q_orig = rotation_matrix_to_quaternion(R_orig)
        
        # 0. Early exit: check if original rotation is already safe
        c2w_orig = np.zeros((3, 4), dtype=np.float32)
        c2w_orig[:3, :3] = R_orig
        c2w_orig[:3, 3] = pos
        template_cam = update_camera_pose(template_cam, c2w_orig)
        e_orig, ms_orig, _, _ = evaluate_view_metrics(template_cam, gaussians, pipeline,
                                                       background, kernel_size, coverage_threshold)

        orig_is_safe = (e_orig <= max_empty_percent and ms_orig <= max_mean_scale)
        reachable_from_next = True
        if next_corrected_R is not None:
            reachable_from_next = rotation_distance(R_orig, next_corrected_R) <= max_step_deg

        if orig_is_safe and reachable_from_next:
            corrected_rotations_map[i] = R_orig
            next_corrected_R = R_orig
            next_corrected_q = q_orig
            continue

        # 1. Generate Candidates (original not safe or not reachable)
        candidates = []

        # Candidate A: Original Rotation
        candidates.append({'R': R_orig, 'q': q_orig, 'type': 'orig',
                          'empty': e_orig, 'scale': ms_orig})

        # Candidate B: Interpolations towards next corrected (if exists)
        if next_corrected_q is not None:
             for t in [0.3, 0.6]:
                 q_interp = quaternion_slerp(q_orig, next_corrected_q, t)
                 R_interp = quaternion_to_rotation_matrix(q_interp)
                 candidates.append({'R': R_interp, 'q': q_interp, 'type': 'interp'})

             candidates.append({'R': next_corrected_R, 'q': next_corrected_q, 'type': 'next'})

        # Candidate C: Random Perturbations around Original (reduced count)
        max_angle_rad = np.deg2rad(max_rotation_perturbation_deg)
        for _ in range(min(num_rotation_samples, 8)):
            axis = rng.normal(size=3)
            axis = axis / (np.linalg.norm(axis) + 1e-8)
            angle = rng.uniform(-max_angle_rad, max_angle_rad)
            R_pert = axis_angle_to_rotation_matrix(axis, angle)
            R_cand = R_pert @ R_orig
            q_cand = rotation_matrix_to_quaternion(R_cand)
            candidates.append({'R': R_cand, 'q': q_cand, 'type': 'pert_orig'})

        # Candidate D: Random Perturbations around Next Corrected (if exists)
        if next_corrected_R is not None:
             for _ in range(min(num_rotation_samples // 2, 4)):
                axis = rng.normal(size=3)
                axis = axis / (np.linalg.norm(axis) + 1e-8)
                angle = rng.uniform(-max_angle_rad, max_angle_rad)
                R_pert = axis_angle_to_rotation_matrix(axis, angle)
                R_cand = R_pert @ next_corrected_R
                q_cand = rotation_matrix_to_quaternion(R_cand)
                candidates.append({'R': R_cand, 'q': q_cand, 'type': 'pert_next'})

        # 2. Filter Safe Candidates
        safe_candidates = []
        best_violation = float('inf')
        best_R_violation = None
        
        for cand in candidates:
            # Skip re-rendering original, we already have its metrics
            if 'empty' not in cand or cand['type'] != 'orig':
                c2w_cand = np.zeros((3, 4), dtype=np.float32)
                c2w_cand[:3, :3] = cand['R']
                c2w_cand[:3, 3] = pos

                template_cam = update_camera_pose(template_cam, c2w_cand)
                e_cand, mean_s, max_s, avg10_s = evaluate_view_metrics(template_cam, gaussians, pipeline,
                                                                       background, kernel_size, coverage_threshold)

                cand['empty'] = e_cand
                cand['scale'] = mean_s
                cand['scale_max'] = max_s
                cand['scale_top10avg'] = avg10_s

            e_cand = cand['empty']
            mean_s = cand['scale']

            is_safe = (e_cand <= max_empty_percent and mean_s <= max_mean_scale)
            if is_safe:
                safe_candidates.append(cand)

            violation = max(0, e_cand - max_empty_percent) + max(0, (mean_s - max_mean_scale) * 1000)
            if violation < best_violation:
                best_violation = violation
                best_R_violation = cand['R']

        # 3. Select Best Candidate
        final_R = R_orig # Default
        
        if len(safe_candidates) > 0:
            # We have safe options. Optimize for smoothness and randomness.
            
            if next_corrected_R is not None:
                # We have a future target constraint
                reachable_safe = []
                for cand in safe_candidates:
                    dist = rotation_distance(cand['R'], next_corrected_R)
                    if dist <= max_step_deg:
                        cand['dist_to_next'] = dist
                        reachable_safe.append(cand)
                
                if len(reachable_safe) > 0:
                    # Pick the one closest to ORIGINAL from the reachable ones
                    # This satisfies Intent (Randomness) + Safety + Smoothness
                    reachable_safe.sort(key=lambda x: rotation_distance(x['R'], R_orig))
                    final_R = reachable_safe[0]['R']
                else:
                    # No safe candidate is easily reachable. 
                    # We must prioritize REACHABILITY or SAFETY?
                    # Prioritize SAFETY, but try to minimize distance to next.
                    # Sort safe candidates by distance to next
                    safe_candidates.sort(key=lambda x: rotation_distance(x['R'], next_corrected_R))
                    final_R = safe_candidates[0]['R']
            else:
                # No future constraint (last frame), just pick safe one closest to original
                safe_candidates.sort(key=lambda x: rotation_distance(x['R'], R_orig))
                final_R = safe_candidates[0]['R']
        else:
            # No safe candidates found at all.
            # Use the least violating candidate
            if best_R_violation is not None:
                final_R = best_R_violation
            else:
                final_R = R_orig

        # Check if we changed anything
        dist_from_orig = rotation_distance(final_R, R_orig)
        if dist_from_orig > 0.1:
            corrections_made += 1
            
        # Store result
        corrected_rotations_map[i] = final_R
        
        # Propagate
        next_corrected_R = final_R
        next_corrected_q = rotation_matrix_to_quaternion(final_R)

    # Reconstruct corrected list
    corrected_rotations = [corrected_rotations_map[i] for i in range(n_frames)]
    
    print(f"  Made corrections to {corrections_made}/{n_frames} frames")
    
    # Smooth rotations path (Final Polish)
    print("  Smoothing rotations with Gaussian filter...")
    quaternions = np.array([rotation_matrix_to_quaternion(R) for R in corrected_rotations])
    smoothed_quats = gaussian_smooth_quaternions(quaternions, sigma=smoothing_sigma)
    smoothed_rotations = [quaternion_to_rotation_matrix(q) for q in smoothed_quats]
    
    # Reconstruct path
    corrected_path = []
    for i in range(n_frames):
        c2w = np.zeros((3, 4), dtype=np.float32)
        c2w[:3, :3] = smoothed_rotations[i]
        c2w[:3, 3] = positions[i]
        corrected_path.append((positions[i], c2w))
    
    print("  Trajectory correction complete!")
    return corrected_path


# ============================================================================
# CAMERA POSE UPDATE
# ============================================================================

def update_camera_pose(camera, c2w):
    """Update camera pose from c2w matrix (3x4)."""
    c2w_4x4 = np.zeros((4, 4), dtype=np.float32)
    c2w_4x4[:3, :] = c2w
    c2w_4x4[3, 3] = 1.0
    
    w2c = np.linalg.inv(c2w_4x4)
    
    camera.world_view_transform = torch.from_numpy(w2c).cuda().float().transpose(0, 1)
    camera.full_proj_transform = (
        camera.world_view_transform.unsqueeze(0).bmm(
            camera.projection_matrix.unsqueeze(0)
        ).squeeze(0)
    )
    camera.camera_center = camera.world_view_transform.inverse()[3, :3]
    
    return camera


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_path_3d(gaussians, cameras, path_data, output_path, subsample=10000, 
                     visible_volume_points=None, waypoints=None):
    """Create 3D visualization of camera path overlaid on point cloud."""
    print("\nGenerating 3D path visualization...")
    
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    
    try:
        sh = gaussians.get_features[:, 0, :3].detach().cpu().numpy()
        colors = (sh - sh.min()) / (sh.max() - sh.min() + 1e-8)
        colors = np.clip(colors, 0, 1)
    except:
        colors = np.ones((len(xyz), 3)) * 0.5
    
    if len(xyz) > subsample:
        indices = np.random.choice(len(xyz), subsample, replace=False)
        xyz_sub = xyz[indices]
        colors_sub = colors[indices]
    else:
        xyz_sub = xyz
        colors_sub = colors
    
    path_positions = np.array([pos for pos, _ in path_data])
    
    train_positions = []
    for cam in cameras:
        w2c = cam.world_view_transform.transpose(0, 1).cpu().numpy()
        c2w = np.linalg.inv(w2c)
        train_positions.append(c2w[:3, 3])
    train_positions = np.array(train_positions)
    
    fig = plt.figure(figsize=(20, 10))
    
    # 3D view
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(xyz_sub[:, 0], xyz_sub[:, 1], xyz_sub[:, 2], 
                c=colors_sub, s=0.1, alpha=0.3, label='Point Cloud')
    
    if visible_volume_points is not None and len(visible_volume_points) > 0:
        if len(visible_volume_points) > 100000:
            idx = np.random.choice(len(visible_volume_points), 100000, replace=False)
            vis_pts = visible_volume_points[idx]
        else:
            vis_pts = visible_volume_points
        ax1.scatter(vis_pts[:, 0], vis_pts[:, 1], vis_pts[:, 2],
                   c='cyan', s=1, alpha=0.05, label='Safe Volume', zorder=0)

    ax1.plot(path_positions[:, 0], path_positions[:, 1], path_positions[:, 2], 
             'r-', linewidth=2, label='Camera Path')
    ax1.scatter(*path_positions[0], c='green', s=100, marker='o', label='Path Start', zorder=5)
    ax1.scatter(*path_positions[-1], c='blue', s=100, marker='s', label='Path End', zorder=5)
    
    arrow_interval = max(1, len(path_data) // 20)
    for i in range(0, len(path_data), arrow_interval):
        pos, c2w = path_data[i]
        forward = c2w[:3, 2] * 0.3
        ax1.quiver(pos[0], pos[1], pos[2], forward[0], forward[1], forward[2],
                   color='orange', arrow_length_ratio=0.3, linewidth=1)
    
    ax1.scatter(train_positions[:, 0], train_positions[:, 1], train_positions[:, 2],
                c='purple', s=20, alpha=0.5, marker='^', label='Training Cameras')
    
    if waypoints is not None and len(waypoints) > 0:
        try:
            wp = np.array(waypoints)
            ax1.scatter(wp[:, 0], wp[:, 1], wp[:, 2], c='yellow', s=80, marker='X', 
                       label='Waypoints', zorder=10)
        except:
            pass

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D View: Camera Path on Point Cloud')
    ax1.legend(loc='upper left', fontsize=8)
    
    # Top-down view
    ax2 = fig.add_subplot(122)
    ax2.scatter(xyz_sub[:, 0], xyz_sub[:, 1], c=colors_sub, s=0.1, alpha=0.3)
    
    if visible_volume_points is not None and len(visible_volume_points) > 0:
        if len(visible_volume_points) > 20000:
            idx = np.random.choice(len(visible_volume_points), 20000, replace=False)
            vis_pts = visible_volume_points[idx]
        else:
            vis_pts = visible_volume_points
        ax2.scatter(vis_pts[:, 0], vis_pts[:, 1], c='cyan', s=1, alpha=0.05)

    ax2.plot(path_positions[:, 0], path_positions[:, 1], 'r-', linewidth=2, label='Camera Path')
    ax2.scatter(path_positions[0, 0], path_positions[0, 1], c='green', s=100, marker='o', 
               label='Start', zorder=5)
    ax2.scatter(path_positions[-1, 0], path_positions[-1, 1], c='blue', s=100, marker='s', 
               label='End', zorder=5)
    ax2.scatter(train_positions[:, 0], train_positions[:, 1], c='purple', s=20, alpha=0.5, 
               marker='^', label='Training Cams')
    
    for i in range(0, len(path_data), arrow_interval):
        pos, c2w = path_data[i]
        forward = c2w[:3, 2] * 0.2
        ax2.arrow(pos[0], pos[1], forward[0], forward[1], 
                  head_width=0.05, head_length=0.02, fc='orange', ec='orange')
    
    if waypoints is not None and len(waypoints) > 0:
        try:
            wp = np.array(waypoints)
            ax2.scatter(wp[:, 0], wp[:, 1], c='yellow', s=80, marker='X', 
                       label='Waypoints', zorder=12)
        except:
            pass
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Top-Down View (XY Plane)')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved path visualization to: {output_path}")


# ============================================================================
# RENDERING
# ============================================================================

def render_path_with_outputs(gaussians, scene, path_data, output_dir, 
                             pipeline, background, kernel_size, nth_frame=1,
                             coverage_threshold=0.05):
    """Render camera path with RGB, optical flow, and coverage outputs."""
    # Create output directories
    img_dir = os.path.join(output_dir, "img")
    flow_dir = os.path.join(output_dir, "flow")
    flow_vis_dir = os.path.join(output_dir, "flow_vis")
    coverage_mask_dir = os.path.join(output_dir, "coverage_masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(flow_dir, exist_ok=True)
    os.makedirs(flow_vis_dir, exist_ok=True)
    os.makedirs(coverage_mask_dir, exist_ok=True)
    
    template_cam = scene.getTrainCameras()[0]
    n_frames = len(path_data)
    
    print(f"\nRendering {n_frames} frames (RGB + Coverage)...")
    
    coverage_file = os.path.join(output_dir, "coverage.txt")
    bg_black = torch.zeros((3), dtype=torch.float32, device="cuda")
    override_color = torch.ones((gaussians.get_xyz.shape[0], 3), dtype=torch.float32, device="cuda")
    
    with open(coverage_file, "w") as f_cov, torch.no_grad():
        f_cov.write("frame_idx,empty_percentage,mean_splat_radius,max_splat_radius,top10pct_avg_splat\n")
        
        for i in tqdm(range(n_frames), desc="Rendering"):
            pos, c2w = path_data[i]
            template_cam = update_camera_pose(template_cam, c2w)

            with torch.no_grad():
                render_out = render(template_cam, gaussians, pipeline, background, kernel_size)
                rgb = render_out['render']
                coverage_map = render_out['render'].mean(dim=0)
            
            empty_mask = coverage_map < coverage_threshold
            empty_percent = empty_mask.float().mean().item() * 100.0

            # Compute mean gaussian splat scale and additional stats for gaussians that are visible (occlusion culled)
            mean_scale, max_scale, avg_top10 = calculate_mean_scale(template_cam, gaussians, render_out)

            # Write metrics to validation/metrics file
            f_cov.write(f"{i:05d},{empty_percent:.4f},{mean_scale:.6f},{max_scale:.6f},{avg_top10:.6f}\n")
            
            # Save coverage mask
            cov_np = coverage_map.cpu().numpy()
            cov_vis = (cov_np * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(coverage_mask_dir, f"{i:05d}.png"), cov_vis)
            
            # Save RGB with overlay
            rgb_np = rgb.permute(1, 2, 0).cpu().numpy()
            rgb_np = (rgb_np * 255).clip(0, 255).astype(np.uint8)
            rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
            
            text = f"Empty: {empty_percent:.1f}%"
            cv2.putText(rgb_bgr, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
            cv2.putText(rgb_bgr, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Overlay mean splat radius (scale) seen by the camera
            scale_text = f"MeanScale: {mean_scale:.3f}"
            cv2.putText(rgb_bgr, scale_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
            cv2.putText(rgb_bgr, scale_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

            # Overlay max splat radius
            max_text = f"MaxScale: {max_scale:.3f}"
            cv2.putText(rgb_bgr, max_text, (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(rgb_bgr, max_text, (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            # Overlay average of top-10% biggest gaussians
            top10_text = f"Top10Avg: {avg_top10:.3f}"
            cv2.putText(rgb_bgr, top10_text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(rgb_bgr, top10_text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            output_path = os.path.join(img_dir, f'{i+1:05d}.png')
            cv2.imwrite(output_path, rgb_bgr)
    
    print(f"Saved {n_frames} images to {img_dir}")
    print(f"Saved coverage stats to {coverage_file}")
    
    # Compute optical flow
    print(f"\nComputing optical flow for {n_frames - nth_frame} pairs...")
    
    for i in tqdm(range(0, n_frames - nth_frame), desc="Computing flow"):
        # Render frame i (reference)
        pos1, c2w1 = path_data[i]
        template_cam = update_camera_pose(template_cam, c2w1)
        
        ref_w2c = template_cam.world_view_transform.transpose(0, 1)
        ref_K = template_cam.K
        
        with torch.no_grad():
            out1 = render(template_cam, gaussians, pipeline, background, kernel_size)
        
        rgb1 = out1['render']
        depth1 = out1['depthmid']
        rgb1_np = rgb1.permute(1, 2, 0).detach().cpu().numpy()
        
        # Render frame i + nth_frame (source)
        pos2, c2w2 = path_data[i + nth_frame]
        template_cam = update_camera_pose(template_cam, c2w2)
        
        src_w2c = template_cam.world_view_transform.transpose(0, 1)
        src_K = template_cam.K
        
        with torch.no_grad():
            out2 = render(template_cam, gaussians, pipeline, background, kernel_size)
        
        rgb2 = out2['render']
        depth2 = out2['depthmid']
        rgb2_np = rgb2.permute(1, 2, 0).detach().cpu().numpy()
        
        # Compute optical flow using geometric consistency
        try:
            mask1, _, _, _, _, _, flowf1, _, _, _, _ = g_utils.check_geometric_consistency(
                depth1.unsqueeze(0), ref_K.unsqueeze(0), ref_w2c.unsqueeze(0), rgb1_np,
                depth2.unsqueeze(0), src_K.unsqueeze(0), src_w2c.unsqueeze(0), rgb2_np,
                thre1=2, thre2=0.01
            )
            
            flow_np = flowf1.permute(1, 2, 0).detach().cpu().numpy()
            mask_np = mask1[0].detach().cpu().numpy()
            flow_np[~mask_np] = 0.0
            
            # Save KITTI format
            flow_filename = os.path.join(flow_dir, f"{i:05d}.png")
            writeFlowKITTI(flow_filename, flow_np)
            
            # Save visualization
            flow_vis = flow_to_image(flow_np)
            flow_vis_bgr = cv2.cvtColor(flow_vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(flow_vis_dir, f"flow_vis_{i:05d}.png"), flow_vis_bgr)
            
        except Exception as e:
            print(f"\nError computing flow for frame {i}: {e}")
    
    print(f"Saved {n_frames - nth_frame} flows to {flow_dir}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = ArgumentParser(description="Path Renderer for 3DGS Models")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--num_path_steps", type=int, default=500)
    parser.add_argument("--nth_frame", type=int, default=1)
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--noise_rot_deg", type=float, default=40.0)
    parser.add_argument("--coverage_threshold", type=float, default=0.1)
    parser.add_argument("--skip_safety", action="store_true")
    parser.add_argument("--max_safe_dist", type=float, default=20.0)
    parser.add_argument("--safety_threshold", type=float, default=1.0)
    parser.add_argument("--num_rays_per_cam", type=int, default=800)

    args = get_combined_args(parser)
    
    print(f"\n=== Path Renderer for 3DGS ===")
    print(f"Model: {args.model_path}")
    print(f"Iteration: {args.iteration}")
    print(f"Num steps: {args.num_path_steps}")
    
    safe_state(args.quiet)
    
    if args.random_seed is not None:
        import random
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        random.seed(args.random_seed)
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_path, 'rendered')
    
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    print("\nLoading scene...")
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)
    
    print(f"  Loaded {gaussians.get_xyz.shape[0]} Gaussians")
    
    cameras = scene.getTrainCameras()
    print(f"  Loaded {len(cameras)} cameras")
    
    # Compute safe volume
    safe_v_tree = None
    safe_points = None
    if not args.skip_safety:
        safe_points = compute_global_visible_volume(
            scene, gaussians, pipeline.extract(args), background, 
            kernel_size=args.kernel_size,
            max_dist=args.max_safe_dist,
            num_rays_per_cam=args.num_rays_per_cam,
            seed=args.random_seed
        )
        print(f"  Generated safe volume with {len(safe_points)} points")
        safe_v_tree = KDTree(safe_points)


    
    # Generate path
    print("\nGenerating camera path...")
    path_result = plan_safe_random_path(
        cameras, 
        safe_points if safe_points is not None else np.array([[0, 0, 0]]), 
        safe_v_tree if safe_v_tree is not None else KDTree(np.array([[0, 0, 0]])),
        num_steps=args.num_path_steps,
        safety_threshold=args.safety_threshold if not args.skip_safety else 1e9,
        noise_scale=args.noise_scale,
        noise_rot_deg=args.noise_rot_deg,
        seed=args.random_seed
    )

    
    path_data = [(path_result['positions'][i], path_result['c2ws'][i]) 
                 for i in range(len(path_result['positions']))]
    
    print(f"  Path from camera {path_result['indices'][0]} to {path_result['indices'][1]}")
    
    # Correct trajectory for coverage
    path_data = correct_trajectory_for_coverage(
        path_data, gaussians, scene, pipeline.extract(args), background, args.kernel_size,
        max_empty_percent=args.coverage_threshold,
        max_mean_scale=0.03,
        coverage_threshold=args.coverage_threshold,
        num_rotation_samples=5,
        max_rotation_perturbation_deg=45.0,
        smoothing_sigma=4.0,
        seed=args.random_seed
    )
    
    # Visualize path
    viz_path = os.path.join(args.output_dir, 'path_visualization.png')
    os.makedirs(args.output_dir, exist_ok=True)
    visualize_path_3d(gaussians, cameras, path_data, viz_path, 
                     visible_volume_points=safe_points,
                     waypoints=path_result.get('sampled_waypoints', None))
    
    # Render outputs
    render_path_with_outputs(
        gaussians, scene, path_data, args.output_dir, 
        pipeline.extract(args), background, args.kernel_size, args.nth_frame,
        coverage_threshold=args.coverage_threshold
    )
    
    # Generate video from rendered images
    img_dir = os.path.join(args.output_dir, "img")
    video_path = os.path.join(args.output_dir, "output_video.mp4")
    print(f"\nGenerating video from {img_dir}...")
    ffmpeg_cmd = (
        f'ffmpeg -y -framerate 30 -pattern_type glob -i \'{img_dir}/*.png\' '
        f'-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v libx264 -pix_fmt yuv420p '
        f'"{video_path}"'
    )
    os.system(ffmpeg_cmd)
    print(f"Saved video to: {video_path}")

    print(f"\n=== Done! ===")
    print(f"Output: {args.output_dir}")


if __name__ == '__main__':
    main()
