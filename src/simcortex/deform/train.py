import os
import gc
import math
import logging
from datetime import timedelta
from contextlib import nullcontext
from typing import Dict, List, Tuple
import random
import numpy as np

import hydra
import torch
import torch.nn.functional as F
import torch.distributed as dist

from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd

from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_normal_consistency
from pytorch3d.loss.point_mesh_distance import _PointFaceDistance
from pytorch3d.ops import knn_points

from simcortex.deform.data.dataloader import CSRDeformDataset, collate_csr_deform
from simcortex.deform.utils.coords import voxel_to_world
from simcortex.deform.models.surfdeform import SurfDeform

import trimesh

try:
    from trimesh.collision import CollisionManager
    _ = CollisionManager()
    HAS_FCL = True
except Exception:
    HAS_FCL = False


log = logging.getLogger(__name__)



def vertex_normals_from_mesh(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    verts: (V, 3)
    faces: (F, 3)
    returns vertex normals: (V, 3)
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
    fn = F.normalize(fn, dim=-1, eps=1e-8)

    vn = torch.zeros_like(verts)
    vn.index_add_(0, faces[:, 0], fn)
    vn.index_add_(0, faces[:, 1], fn)
    vn.index_add_(0, faces[:, 2], fn)
    vn = F.normalize(vn, dim=-1, eps=1e-8)
    return vn

def signed_nested_surface_penalty(
    white_v: torch.Tensor,
    white_f: torch.Tensor,
    pial_v: torch.Tensor,
    pial_f: torch.Tensor,
    margin_mm: float = 0.5,
    n_points: int = 40000,
):
    """
    Enforces:
      - white should stay inside pial
      - pial should stay outside white

    Sign convention (assuming outward normals):
      - point inside a closed surface  -> signed distance < 0
      - point outside a closed surface -> signed distance > 0
    """
    device = white_v.device

    mesh_w = Meshes(verts=[white_v], faces=[white_f])
    mesh_p = Meshes(verts=[pial_v], faces=[pial_f])

    # surface samples instead of vertex-only samples
    w_pts = sample_points_from_meshes(mesh_w, num_samples=n_points).squeeze(0)
    p_pts = sample_points_from_meshes(mesh_p, num_samples=n_points).squeeze(0)

    pial_normals = vertex_normals_from_mesh(pial_v, pial_f)
    white_normals = vertex_normals_from_mesh(white_v, white_f)

    # ---------------------------------------------------
    # 1) white samples relative to pial
    # white should be INSIDE pial => signed_w should be <= -margin
    # ---------------------------------------------------
    knn_wp = knn_points(w_pts[None], pial_v[None], K=1, return_nn=False)
    idx_pial = knn_wp.idx[0, :, 0]
    nearest_pial = pial_v[idx_pial]
    nearest_pial_n = pial_normals[idx_pial]

    signed_w = ((w_pts - nearest_pial) * nearest_pial_n).sum(dim=-1)
    loss_w = F.relu(signed_w + margin_mm).mean()

    # ---------------------------------------------------
    # 2) pial samples relative to white
    # pial should be OUTSIDE white => signed_p should be >= +margin
    # ---------------------------------------------------
    knn_pw = knn_points(p_pts[None], white_v[None], K=1, return_nn=False)
    idx_white = knn_pw.idx[0, :, 0]
    nearest_white = white_v[idx_white]
    nearest_white_n = white_normals[idx_white]

    signed_p = ((p_pts - nearest_white) * nearest_white_n).sum(dim=-1)
    loss_p = F.relu(margin_mm - signed_p).mean()

    loss = 0.5 * (loss_w + loss_p)

    with torch.no_grad():
        bad_white_pct = (signed_w > -margin_mm).float().mean().item() * 100.0
        bad_pial_pct = (signed_p < margin_mm).float().mean().item() * 100.0
        mean_signed_w = signed_w.mean().item()
        mean_signed_p = signed_p.mean().item()

    return loss, bad_white_pct, bad_pial_pct, mean_signed_w, mean_signed_p

def spatial_gradients_3d(u: torch.Tensor):
    """
    u: (B, 3, D, H, W), displacement field in voxel units.
    returns central-ish finite differences along D,H,W.
    """
    dz = u[:, :, 1:, :, :] - u[:, :, :-1, :, :]
    dy = u[:, :, :, 1:, :] - u[:, :, :, :-1, :]
    dx = u[:, :, :, :, 1:] - u[:, :, :, :, :-1]
    return dz, dy, dx


def jacobian_det_3d(phi: torch.Tensor):
    """
    phi: (B, 3, D, H, W), displacement field in voxel coordinates.
    Mapping is x -> x + phi(x).
    Returns detJ on inner grid: (B, D-1, H-1, W-1)
    """
    B, C, D, H, W = phi.shape
    assert C == 3

    # forward differences, cropped to common shape
    dD = phi[:, :, 1:, :-1, :-1] - phi[:, :, :-1, :-1, :-1]
    dH = phi[:, :, :-1, 1:, :-1] - phi[:, :, :-1, :-1, :-1]
    dW = phi[:, :, :-1, :-1, 1:] - phi[:, :, :-1, :-1, :-1]

    # phi components are in I,J,K order:
    # component 0 changes D/I, component 1 changes H/J, component 2 changes W/K
    J00 = 1.0 + dD[:, 0]
    J01 = dH[:, 0]
    J02 = dW[:, 0]

    J10 = dD[:, 1]
    J11 = 1.0 + dH[:, 1]
    J12 = dW[:, 1]

    J20 = dD[:, 2]
    J21 = dH[:, 2]
    J22 = 1.0 + dW[:, 2]

    det = (
        J00 * (J11 * J22 - J12 * J21)
        - J01 * (J10 * J22 - J12 * J20)
        + J02 * (J10 * J21 - J11 * J20)
    )
    return det


def folding_penalty_from_phis(phis, margin: float = 0.05, topk_frac: float = 0.001):
    device = phis[0].device
    loss = torch.zeros((), device=device)

    negpct_vals = []
    min_det_vals = []
    p001_vals = []   # 0.01 percentile = q=0.0001
    p01_vals = []    # 0.1 percentile  = q=0.001
    p1_vals = []     # 1 percentile    = q=0.01

    for phi in phis:
        det = jacobian_det_3d(phi)                 # [B, D-1, H-1, W-1]
        det_flat = det.flatten(1)                  # [B, N]

        bad = F.relu(margin - det_flat)
        k = max(1, int(float(topk_frac) * bad.shape[1]))
        loss = loss + bad.topk(k, dim=1).values.mean()

        with torch.no_grad():
            negpct_vals.append((det_flat < 0).float().mean().item() * 100.0)
            min_det_vals.append(det_flat.min().item())

            flat_all = det_flat.reshape(-1)
            p001_vals.append(torch.quantile(flat_all, 0.0001).item())
            p01_vals.append(torch.quantile(flat_all, 0.001).item())
            p1_vals.append(torch.quantile(flat_all, 0.01).item())

    loss = loss / max(len(phis), 1)

    neg_pct = float(sum(negpct_vals) / max(len(negpct_vals), 1))
    min_det = float(min(min_det_vals)) if min_det_vals else 0.0
    jac_p001 = float(sum(p001_vals) / max(len(p001_vals), 1))
    jac_p01  = float(sum(p01_vals) / max(len(p01_vals), 1))
    jac_p1   = float(sum(p1_vals) / max(len(p1_vals), 1))

    return loss, neg_pct, min_det, jac_p001, jac_p01, jac_p1

def count_collisions_inmemory(
    vA_mm: torch.Tensor, fA: torch.Tensor,
    vB_mm: torch.Tensor, fB: torch.Tensor
):
    """
    vA_mm, vB_mm: (V,3) torch float in mm-space (GPU/CPU)
    fA, fB: (F,3) torch long
    Returns: (is_col: bool or None, n_contacts: int or None)
    """
    if not HAS_FCL:
        return None, None

    vA = vA_mm.detach().float().cpu().numpy()
    vB = vB_mm.detach().float().cpu().numpy()
    fA_np = fA.detach().long().cpu().numpy()
    fB_np = fB.detach().long().cpu().numpy()

    if vA.shape[0] == 0 or vB.shape[0] == 0 or fA_np.shape[0] == 0 or fB_np.shape[0] == 0:
        return False, 0

    mA = trimesh.Trimesh(vertices=vA, faces=fA_np, process=False)
    mB = trimesh.Trimesh(vertices=vB, faces=fB_np, process=False)

    cm = CollisionManager()
    cm.add_object("A", mA)
    cm.add_object("B", mB)

    is_col, contacts = cm.in_collision_internal(return_names=False, return_data=True)
    if (not is_col) or (contacts is None):
        return False, 0
    return True, int(len(contacts))


# -----------------------
# DDP helpers
# -----------------------
def setup_ddp() -> Tuple[int, int, int, bool]:
    """Return (rank, world_size, local_rank, is_distributed)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=timedelta(hours=6),
        )
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True

    return 0, 1, 0, False


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def seed_all(seed: int, rank: int = 0):
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_map(cfg_node, keys: Tuple[str, ...]) -> Dict[str, str] | None:
    for k in keys:
        v = getattr(cfg_node, k, None)
        if v is not None and hasattr(v, "items"):
            return {str(kk): str(vv) for kk, vv in v.items()}
    return None


def normalize_subject_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "subject" not in df.columns:
        raise ValueError(f"split_file must contain column 'subject'. Got: {list(df.columns)}")
    df["subject"] = df["subject"].astype(str)
    df["subject"] = df["subject"].apply(lambda x: x if x.startswith("sub-") else f"sub-{x}")
    return df


def _validate_single_split_df(df: pd.DataFrame) -> None:
    req = {"subject", "split"}
    if not req.issubset(set(df.columns)):
        raise ValueError(f"Single-dataset split_file must contain columns {sorted(req)}. Got: {list(df.columns)}")

    if "dataset" in df.columns:
        vals = sorted(df["dataset"].dropna().astype(str).str.strip().unique().tolist())
        if len(vals) > 1:
            raise ValueError(
                "Single-dataset deform train received a split_file with multiple dataset values. "
                "Please provide a split CSV for one dataset only."
            )


def _validate_multi_split_df(df: pd.DataFrame) -> None:
    req = {"subject", "split", "dataset"}
    if not req.issubset(set(df.columns)):
        raise ValueError(f"Multi-dataset split_file must contain columns {sorted(req)}. Got: {list(df.columns)}")


def _detect_deform_train_mode(cfg):
    single_preproc_root = OmegaConf.select(cfg, "dataset.path", default=None)
    single_initsurf_root = OmegaConf.select(cfg, "dataset.initsurf_root", default=None)

    single_preproc_root = None if single_preproc_root in (None, "") else str(single_preproc_root)
    single_initsurf_root = None if single_initsurf_root in (None, "") else str(single_initsurf_root)

    roots_map = _get_map(cfg.dataset, ("roots",))
    initsurf_roots_map = _get_map(cfg.dataset, ("initsurf_roots",))

    if single_preproc_root is not None:
        log.info("Deform training mode: SINGLE-DATASET")
        log.info(f"dataset.path = {single_preproc_root}")
        log.info(f"dataset.initsurf_root = {single_initsurf_root}")

        if single_initsurf_root is None:
            raise ValueError("Single-dataset deform train requires dataset.initsurf_root")

        if roots_map is not None:
            log.warning(
                "Both dataset.path and dataset.roots are present. "
                "Using SINGLE-DATASET mode and ignoring dataset.roots."
            )
        if initsurf_roots_map is not None:
            log.warning(
                "Both dataset.initsurf_root and dataset.initsurf_roots are present. "
                "Using SINGLE-DATASET mode and ignoring dataset.initsurf_roots."
            )

        return "single", single_preproc_root, single_initsurf_root, None, None

    if roots_map is not None:
        log.info("Deform training mode: MULTI-DATASET")
        log.info(f"dataset.roots keys = {list(roots_map.keys())}")

        if initsurf_roots_map is None:
            raise ValueError("Multi-dataset deform train requires dataset.initsurf_roots")

        missing = sorted(set(roots_map.keys()) - set(initsurf_roots_map.keys()))
        if missing:
            raise KeyError(f"dataset.initsurf_roots missing keys required by dataset.roots: {missing}")

        extra = sorted(set(initsurf_roots_map.keys()) - set(roots_map.keys()))
        if extra:
            log.warning(f"dataset.initsurf_roots has extra keys not present in dataset.roots: {extra}")

        return "multi", None, None, roots_map, initsurf_roots_map

    raise ValueError(
        "Could not determine deform training mode. Provide either:\n"
        "  - dataset.path + dataset.initsurf_root   (single-dataset)\n"
        "or\n"
        "  - dataset.roots + dataset.initsurf_roots (multi-dataset)"
    )


# -----------------------
# Geometry helpers
# -----------------------
def mesh_is_valid(verts: torch.Tensor, faces: torch.Tensor) -> bool:
    if verts is None or faces is None:
        return False
    if verts.ndim != 2 or faces.ndim != 2:
        return False
    if verts.shape[1] != 3 or faces.shape[1] != 3:
        return False
    if verts.numel() == 0 or faces.numel() == 0:
        return False
    if torch.isnan(verts).any() or torch.isinf(verts).any():
        return False
    f = faces.long()
    if f.min().item() < 0:
        return False
    if f.max().item() >= verts.shape[0]:
        return False
    return True


# -----------------------
# Low-quantile separation penalty
# -----------------------
_PointFaceDistanceOP = _PointFaceDistance.apply


def point_to_mesh_dist_p3d(points: torch.Tensor, mesh: Meshes) -> torch.Tensor:
    """
    points: (N,3) float on device
    mesh: Meshes (batch size 1)
    returns: (N,) distances in same units as verts (here mm)
    """
    pts = points
    first_idx = torch.zeros((1,), device=pts.device, dtype=torch.int64)  # batch size 1
    max_pts = int(pts.shape[0])

    tris = mesh.verts_packed()[mesh.faces_packed()]  # (F,3,3)
    tri_first = mesh.mesh_to_faces_packed_first_idx()  # (1,)

    d2 = _PointFaceDistanceOP(pts, first_idx, tris, tri_first, max_pts)  # squared
    return d2.sqrt()


def partial_hd_penalty(mesh_a: Meshes, mesh_b: Meshes, p: float, lam: float, n_pts: int):
    """
    Low-quantile symmetric separation penalty.

    This is not classical Hausdorff distance. It samples points on both meshes,
    computes symmetric point-to-surface distances, takes a LOW quantile, and
    penalizes it if the separation is below lam.

    Returns:
      sep_q_mm: scalar tensor (mm)
      penalty: scalar tensor = relu(lam - sep_q_mm)
    """
    pa = sample_points_from_meshes(mesh_a, num_samples=n_pts).squeeze(0)
    pb = sample_points_from_meshes(mesh_b, num_samples=n_pts).squeeze(0)

    da = point_to_mesh_dist_p3d(pa, mesh_b)
    db = point_to_mesh_dist_p3d(pb, mesh_a)

    d_all = torch.cat([da, db], dim=0)  # (2n,)
    sep_q_mm = torch.quantile(d_all, q=float(p))

    lam_t = sep_q_mm.new_tensor(float(lam))
    penalty = F.relu(lam_t - sep_q_mm)
    return sep_q_mm, penalty


# -----------------------
# Random affine augmentation in NDC (volume + verts)
# -----------------------
def voxel_sizes_xyz_from_affine(A: torch.Tensor) -> torch.Tensor:
    A3 = A[:3, :3]
    vsize_ijk = torch.linalg.norm(A3, dim=0).clamp(min=1e-6)
    return vsize_ijk[[2, 1, 0]]  # xyz


def ijk_to_xyz(v_ijk: torch.Tensor) -> torch.Tensor:
    return torch.stack([v_ijk[..., 2], v_ijk[..., 1], v_ijk[..., 0]], dim=-1)


def xyz_to_ijk(v_xyz: torch.Tensor) -> torch.Tensor:
    return torch.stack([v_xyz[..., 2], v_xyz[..., 1], v_xyz[..., 0]], dim=-1)


def voxel_to_ndc_xyz(v_xyz: torch.Tensor, D: int, H: int, W: int) -> torch.Tensor:
    den = torch.tensor([W - 1, H - 1, D - 1], device=v_xyz.device, dtype=v_xyz.dtype).clamp(min=1.0)
    return 2.0 * (v_xyz / den) - 1.0


def ndc_to_voxel_xyz(u_xyz: torch.Tensor, D: int, H: int, W: int) -> torch.Tensor:
    den = torch.tensor([W - 1, H - 1, D - 1], device=u_xyz.device, dtype=u_xyz.dtype).clamp(min=1.0)
    return 0.5 * (u_xyz + 1.0) * den


def random_affine_ndc_xyz(B: int, rot_deg: float, scale_range: float, trans_ndc_xyz: torch.Tensor, device, dtype):
    ang = (torch.rand(B, 3, device=device, dtype=dtype) * 2 - 1) * (rot_deg * math.pi / 180.0)
    cx, sx = torch.cos(ang[:, 0]), torch.sin(ang[:, 0])
    cy, sy = torch.cos(ang[:, 1]), torch.sin(ang[:, 1])
    cz, sz = torch.cos(ang[:, 2]), torch.sin(ang[:, 2])

    Rx = torch.stack([
        torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx),
        torch.zeros_like(cx), cx, -sx,
        torch.zeros_like(cx), sx, cx
    ], dim=-1).view(-1, 3, 3)

    Ry = torch.stack([
        cy, torch.zeros_like(cy), sy,
        torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy),
        -sy, torch.zeros_like(cy), cy
    ], dim=-1).view(-1, 3, 3)

    Rz = torch.stack([
        cz, -sz, torch.zeros_like(cz),
        sz, cz, torch.zeros_like(cz),
        torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)
    ], dim=-1).view(-1, 3, 3)

    R = Rz @ Ry @ Rx

    ds = (torch.rand(B, 1, device=device, dtype=dtype) * 2 - 1) * scale_range
    s = 1.0 + ds
    A = R * s.view(B, 1, 1)

    t = (torch.rand(B, 3, device=device, dtype=dtype) * 2 - 1) * trans_ndc_xyz
    b = t
    return A, b


def apply_aug(vol, padded_init_ijk, lengths, gt_verts_dict_list, affines, cfg, surface_names):
    prob = float(getattr(cfg.dataset, "aug_prob", 0.0))
    if prob <= 0.0:
        return vol, padded_init_ijk, gt_verts_dict_list

    B, C, D, H, W = vol.shape
    device = vol.device
    dtype = vol.dtype

    mask = (torch.rand(B, device=device) < prob)
    if mask.sum().item() == 0:
        return vol, padded_init_ijk, gt_verts_dict_list

    rot_deg = float(getattr(cfg.dataset, "aug_rot_range_deg", 0.0))
    scale_range = float(getattr(cfg.dataset, "aug_scale_range", 0.0))
    trans_mm = float(getattr(cfg.dataset, "aug_trans_range_mm", 0.0))

    trans_ndc_xyz = torch.zeros((B, 3), device=device, dtype=dtype)
    den_xyz = torch.tensor([W - 1, H - 1, D - 1], device=device, dtype=dtype).clamp(min=1.0)

    for i in range(B):
        vsize_xyz = voxel_sizes_xyz_from_affine(affines[i].to(device=device, dtype=dtype))
        trans_vox_xyz = (trans_mm / vsize_xyz)
        trans_ndc_xyz[i] = 2.0 * (trans_vox_xyz / den_xyz)

    A_fwd, b_fwd = random_affine_ndc_xyz(B, rot_deg, scale_range, trans_ndc_xyz, device, dtype)

    I = torch.eye(3, device=device, dtype=dtype).view(1, 3, 3).repeat(B, 1, 1)
    Z = torch.zeros((B, 3), device=device, dtype=dtype)
    A_fwd = torch.where(mask.view(B, 1, 1), A_fwd, I)
    b_fwd = torch.where(mask.view(B, 1), b_fwd, Z)

    A_inv = torch.linalg.inv(A_fwd)
    b_inv = -(A_inv @ b_fwd.unsqueeze(-1)).squeeze(-1)

    theta = torch.zeros((B, 3, 4), device=device, dtype=dtype)
    theta[:, :, :3] = A_inv
    theta[:, :, 3] = b_inv

    grid = F.affine_grid(theta, size=vol.size(), align_corners=True)
    vol = F.grid_sample(vol, grid, mode="bilinear", padding_mode="border", align_corners=True)

    for i in range(B):
        if not mask[i].item():
            continue

        L = int(lengths[i].item())

        v_ijk = padded_init_ijk[i, :L]
        v_xyz = ijk_to_xyz(v_ijk)
        u = voxel_to_ndc_xyz(v_xyz, D, H, W)
        u2 = (A_fwd[i] @ u.t()).t() + b_fwd[i].view(1, 3)
        v_xyz2 = ndc_to_voxel_xyz(u2, D, H, W)
        padded_init_ijk[i, :L] = xyz_to_ijk(v_xyz2)

        gdict = gt_verts_dict_list[i]
        for s in surface_names:
            gv_ijk = gdict[s]
            gv_xyz = ijk_to_xyz(gv_ijk)
            ug = voxel_to_ndc_xyz(gv_xyz, D, H, W)
            ug2 = (A_fwd[i] @ ug.t()).t() + b_fwd[i].view(1, 3)
            gv_xyz2 = ndc_to_voxel_xyz(ug2, D, H, W)
            gdict[s] = xyz_to_ijk(gv_xyz2)
        gt_verts_dict_list[i] = gdict

    return vol, padded_init_ijk, gt_verts_dict_list


def apply_intensity_aug(vol, cfg):
    """
    MRI-appearance augmentation applied ONLY to the MRI channel (vol[:, 0:1]).
    The geometry/probability channel(s) (vol[:, 1:]) are left untouched.

    All operations are safe on z-score-normalized MRI (values ~N(0,1), may be
    negative). This is the main regularizer for closing the train/val gap, since
    the affine aug in apply_aug() never perturbs intensity/appearance.

    Config (under cfg.dataset, all default to 0 = disabled):
      aug_intensity_prob : per-sample probability of applying intensity aug
      aug_bias_strength  : std of the smooth multiplicative bias field (e.g. 0.3)
      aug_gain_range     : +/- multiplicative contrast gain (e.g. 0.1 -> x*[0.9,1.1])
      aug_bright_range   : +/- additive brightness shift in z-units (e.g. 0.1)
      aug_noise_std      : std of additive Gaussian noise in z-units (e.g. 0.05)
    """
    prob = float(getattr(cfg.dataset, "aug_intensity_prob", 0.0))
    if prob <= 0.0:
        return vol

    B, C, D, H, W = vol.shape
    device = vol.device
    dtype = vol.dtype

    mask = (torch.rand(B, device=device) < prob)
    if mask.sum().item() == 0:
        return vol

    bias_strength = float(getattr(cfg.dataset, "aug_bias_strength", 0.0))
    gain_range    = float(getattr(cfg.dataset, "aug_gain_range", 0.0))
    bright_range  = float(getattr(cfg.dataset, "aug_bright_range", 0.0))
    noise_std     = float(getattr(cfg.dataset, "aug_noise_std", 0.0))

    mri = vol[:, 0:1].clone()  # (B,1,D,H,W)

    for i in range(B):
        if not mask[i].item():
            continue

        x = mri[i:i + 1]  # (1,1,D,H,W)

        # 1) smooth multiplicative bias field (low-frequency -> upsampled -> exp)
        if bias_strength > 0.0:
            lo = torch.randn(1, 1, 4, 5, 4, device=device, dtype=dtype) * bias_strength
            field = F.interpolate(lo, size=(D, H, W), mode="trilinear", align_corners=True)
            x = x * torch.exp(field)

        # 2) global contrast gain
        if gain_range > 0.0:
            g = 1.0 + (torch.rand(1, device=device, dtype=dtype) * 2 - 1) * gain_range
            x = x * g

        # 3) global brightness shift
        if bright_range > 0.0:
            b = (torch.rand(1, device=device, dtype=dtype) * 2 - 1) * bright_range
            x = x + b

        # 4) additive Gaussian noise
        if noise_std > 0.0:
            x = x + torch.randn_like(x) * noise_std

        mri[i:i + 1] = x

    if C > 1:
        vol = torch.cat([mri, vol[:, 1:]], dim=1)
    else:
        vol = mri

    return vol


# -----------------------
# Utilities for building padded init verts
# -----------------------
def build_merged_init_and_metadata(batch, device, surface_names):
    B = len(batch["init_verts_vox"])

    per_counts_init: List[List[int]] = []
    merged_init_list: List[torch.Tensor] = []
    init_faces_list: List[Dict[str, torch.Tensor]] = []
    gt_verts_list: List[Dict[str, torch.Tensor]] = []
    gt_faces_list: List[Dict[str, torch.Tensor]] = []

    for i in range(B):
        counts = []
        v_all = []
        f_init_dict = {}
        gv_dict = {}
        gf_dict = {}

        for s in surface_names:
            v = batch["init_verts_vox"][i][s].to(device)
            f = batch["init_faces"][i][s].to(device).long()
            gv = batch["gt_verts_vox"][i][s].to(device)
            gf = batch["gt_faces"][i][s].to(device).long()

            counts.append(int(v.shape[0]))
            v_all.append(v)
            f_init_dict[s] = f
            gv_dict[s] = gv
            gf_dict[s] = gf

        per_counts_init.append(counts)
        merged_init_list.append(torch.cat(v_all, dim=0))
        init_faces_list.append(f_init_dict)
        gt_verts_list.append(gv_dict)
        gt_faces_list.append(gf_dict)

    lengths = torch.tensor([v.shape[0] for v in merged_init_list], device=device, dtype=torch.long)
    Vmax = int(lengths.max().item())

    padded_init = torch.zeros((B, Vmax, 3), device=device, dtype=merged_init_list[0].dtype)
    for i in range(B):
        padded_init[i, :lengths[i]] = merged_init_list[i]

    return lengths, padded_init, per_counts_init, init_faces_list, gt_verts_list, gt_faces_list


def save_model_state(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    net = model.module if hasattr(model, "module") else model
    torch.save(net.state_dict(), path)


def extract_model_state_dict(checkpoint):
    """Return a model state_dict from either a raw state_dict or a full checkpoint."""
    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            value = checkpoint.get(key, None)
            if isinstance(value, dict):
                state = value
                break

    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint does not contain a state_dict-like object: {type(state)}")

    # Be tolerant of checkpoints saved from DataParallel/DDP wrappers.
    if state and all(isinstance(k, str) and k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}
    return state


def move_optimizer_state_to_device(optimizer, device):
    """Move optimizer state tensors after loading a CPU checkpoint for CUDA training."""
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device, non_blocking=True)


def save_full_checkpoint(
    model,
    optimizer,
    scheduler,
    path: str,
    epoch: int,
    best_score: float,
    best_rmse_seen: float,
    best_model_epoch: int,
    best_rmse_epoch: int,
    cfg,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    net = model.module if hasattr(model, "module") else model
    ckpt = {
        "epoch": epoch,
        "model": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_score": best_score,
        "best_rmse_seen": best_rmse_seen,
        "best_model_epoch": best_model_epoch,
        "best_rmse_epoch": best_rmse_epoch,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    torch.save(ckpt, path)


def fmt_collision_stats(total, hit, csum):
    if total <= 0:
        return "NA"
    pct = 100.0 * (hit / total)
    mean_all = csum / total
    mean_hit = csum / max(hit, 1.0)
    return f"{hit:.0f}/{total:.0f} ({pct:.2f}%) | MeanContacts(all)={mean_all:.2f} | MeanContacts(hit)={mean_hit:.2f}"


def compute_collision_percentages(
    lh_total: float,
    lh_hit: float,
    rh_total: float,
    rh_hit: float,
    lr_total: float,
    lr_hit: float,
) -> Tuple[float, float]:
    wp_total = lh_total + rh_total
    wp_hit = lh_hit + rh_hit
    wp_pct = 100.0 * wp_hit / max(wp_total, 1.0)
    lr_pct = 100.0 * lr_hit / max(lr_total, 1.0)
    return float(wp_pct), float(lr_pct)


# -----------------------
# Main
# -----------------------
@hydra.main(version_base=None, config_path="pkg://simcortex.configs.deform", config_name="train")
def main(cfg: DictConfig):
    rank, world_size, local_rank, is_distributed = setup_ddp()

    try:
        if cfg.user_config:
            cfg = OmegaConf.merge(cfg, OmegaConf.load(cfg.user_config))

        level = getattr(logging, str(getattr(cfg.trainer, "log_level", "INFO")).upper(), logging.INFO)
        logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        if rank == 0:
            log.info("world_size=%d, local_rank=%d", world_size, local_rank)
            print(OmegaConf.to_yaml(cfg))

        seed_all(int(cfg.trainer.seed), rank=rank)
        torch.backends.cudnn.benchmark = True

        surface_names = list(cfg.dataset.surface_name)
        inshape = tuple(int(x) for x in cfg.model.inshape)

        split_file = str(cfg.dataset.split_file)
        train_split = str(getattr(cfg.dataset, "train_split_name", "train"))
        val_split = str(getattr(cfg.dataset, "val_split_name", "val"))

        session_label = str(getattr(cfg.dataset, "session_label", "01"))
        space = str(getattr(cfg.dataset, "space", "MNI152"))

        mode, single_preproc_root, single_initsurf_root, roots_map, initsurf_roots_map = _detect_deform_train_mode(cfg)

        df = pd.read_csv(split_file)
        df = normalize_subject_column(df)

        # ---- Multi-dataset mode ----
        if mode == "multi":
            _validate_multi_split_df(df)
            df["dataset"] = df["dataset"].astype(str).str.strip()

            train_sets = []
            val_sets = []

            for ds_key, ds_df in df.groupby("dataset"):
                if ds_key not in roots_map or ds_key not in initsurf_roots_map:
                    raise KeyError(f"Missing dataset key in config: {ds_key}")

                preproc_root = str(roots_map[ds_key])
                initsurf_root = str(initsurf_roots_map[ds_key])

                tr_subs = ds_df[ds_df["split"].astype(str).str.strip() == train_split]["subject"].astype(str).tolist()
                va_subs = ds_df[ds_df["split"].astype(str).str.strip() == val_split]["subject"].astype(str).tolist()

                if len(tr_subs) > 0:
                    train_sets.append(
                        CSRDeformDataset(
                            preproc_root=preproc_root,
                            initsurf_root=initsurf_root,
                            subjects=tr_subs,
                            session_label=session_label,
                            space=space,
                            surface_names=surface_names,
                            inshape_dhw=inshape,
                            prob_clip_min=cfg.dataset.prob_clip_min,
                            prob_clip_max=cfg.dataset.prob_clip_max,
                            prob_gamma=cfg.dataset.prob_gamma,
                            aug=False,
                        )
                    )

                if len(va_subs) > 0:
                    val_sets.append(
                        CSRDeformDataset(
                            preproc_root=preproc_root,
                            initsurf_root=initsurf_root,
                            subjects=va_subs,
                            session_label=session_label,
                            space=space,
                            surface_names=surface_names,
                            inshape_dhw=inshape,
                            prob_clip_min=cfg.dataset.prob_clip_min,
                            prob_clip_max=cfg.dataset.prob_clip_max,
                            prob_gamma=cfg.dataset.prob_gamma,
                            aug=False,
                        )
                    )

            if len(train_sets) == 0:
                raise RuntimeError("No training subjects found (multi-dataset). Check split_file and train_split_name.")
            if len(val_sets) == 0:
                raise RuntimeError("No validation subjects found (multi-dataset). Check split_file and val_split_name.")

            train_ds = ConcatDataset(train_sets) if len(train_sets) > 1 else train_sets[0]
            val_ds = ConcatDataset(val_sets) if len(val_sets) > 1 else val_sets[0]

        # ---- Single-dataset mode ----
        else:
            _validate_single_split_df(df)

            tr_subs = df[df["split"].astype(str).str.strip() == train_split]["subject"].astype(str).tolist()
            va_subs = df[df["split"].astype(str).str.strip() == val_split]["subject"].astype(str).tolist()

            if len(tr_subs) == 0:
                raise RuntimeError("No training subjects found (single-dataset). Check split_file and train_split_name.")
            if len(va_subs) == 0:
                raise RuntimeError("No validation subjects found (single-dataset). Check split_file and val_split_name.")

            train_ds = CSRDeformDataset(
                preproc_root=str(single_preproc_root),
                initsurf_root=str(single_initsurf_root),
                subjects=tr_subs,
                session_label=session_label,
                space=space,
                surface_names=surface_names,
                inshape_dhw=inshape,
                prob_clip_min=cfg.dataset.prob_clip_min,
                prob_clip_max=cfg.dataset.prob_clip_max,
                prob_gamma=cfg.dataset.prob_gamma,
                aug=False,
            )

            val_ds = CSRDeformDataset(
                preproc_root=str(single_preproc_root),
                initsurf_root=str(single_initsurf_root),
                subjects=va_subs,
                session_label=session_label,
                space=space,
                surface_names=surface_names,
                inshape_dhw=inshape,
                prob_clip_min=cfg.dataset.prob_clip_min,
                prob_clip_max=cfg.dataset.prob_clip_max,
                prob_gamma=cfg.dataset.prob_gamma,
                aug=False,
            )

        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if is_distributed else None

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=int(cfg.trainer.img_batch_size),
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=int(cfg.trainer.num_workers),
            pin_memory=True,
            collate_fn=collate_csr_deform,
        )

        # IMPORTANT: validation loader is NOT distributed to avoid sampler padding (77 -> 78)
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=int(cfg.trainer.img_batch_size),
            shuffle=False,
            num_workers=int(cfg.trainer.num_workers),
            pin_memory=True,
            collate_fn=collate_csr_deform,
        )

        if rank == 0:
            log.info("Loaded %d training subjects", len(train_ds))
            log.info("Loaded %d validation subjects", len(val_ds))

        # model
        # This deformation trainer uses exactly two volumetric channels: MRI + ribbon probability.
        # The optional prob-gradient channel is intentionally not supported in this final trainer.
        if int(cfg.model.c_in) != 2:
            raise ValueError(
                f"model.c_in={cfg.model.c_in}, but this trainer expects exactly 2 input channels "
                "[MRI, ribbon_prob]. Set model.c_in=2."
            )

        model = SurfDeform(
            C_hid=cfg.model.c_hid,
            C_in=int(cfg.model.c_in),
            inshape=inshape,
            sigma=float(cfg.model.sigma),
            geom_ratio=float(getattr(cfg.model, "geom_ratio", 0.5)),
            geom_depth=int(getattr(cfg.model, "geom_depth", 4)),
            gn_groups=int(getattr(cfg.model, "gn_groups", 8)),
            gate_init=float(getattr(cfg.model, "gate_init", -3.0)),
            dropout=float(getattr(cfg.model, "dropout", 0.0)),
        ).to(device)

        # optional initialization checkpoint: model-only or full checkpoint are both supported.
        init_ckpt = str(getattr(cfg.model, "init_ckpt", "") or "")
        if init_ckpt:
            if rank == 0:
                log.info("Loading init_ckpt: %s", init_ckpt)
            raw_ckpt = torch.load(init_ckpt, map_location="cpu")
            sd = extract_model_state_dict(raw_ckpt)
            missing, unexpected = model.load_state_dict(
                sd,
                strict=bool(getattr(cfg.model, "init_strict", True)),
            )
            if rank == 0:
                log.info("Init load done. missing=%d unexpected=%d", len(missing), len(unexpected))

        if is_distributed:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

        # optim
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.trainer.learning_rate),
            weight_decay=float(cfg.trainer.weight_decay),
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(cfg.trainer.scheduler_factor),
            patience=int(cfg.trainer.scheduler_patience),
            threshold=float(cfg.trainer.scheduler_threshold_mm),
            threshold_mode=str(cfg.trainer.scheduler_threshold_mode),
            cooldown=int(cfg.trainer.scheduler_cooldown),
            min_lr=float(cfg.trainer.scheduler_min_lr),
            verbose=(rank == 0),
        )

        # Logging & Config Saving
        out_root = str(getattr(cfg.outputs, "root", getattr(cfg.outputs, "output_dir", "")))

        tb_writer = None
        if rank == 0:
            os.makedirs(out_root, exist_ok=True)

            log_path = os.path.join(out_root, "train.log")
            root_logger = logging.getLogger()
            root_logger.setLevel(level)

            for h in list(root_logger.handlers):
                if isinstance(h, logging.FileHandler):
                    root_logger.removeHandler(h)

            file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
            )
            root_logger.addHandler(file_handler)

            tb_dir = os.path.join(out_root, "tb_logs")
            os.makedirs(tb_dir, exist_ok=True)

            log.info("TensorBoard logging to %s", tb_dir)
            log.info("Log file writing to %s", log_path)

            resolved_conf_yaml = OmegaConf.to_yaml(cfg, resolve=True)
            config_path = os.path.join(out_root, "config_resolved.yaml")
            with open(config_path, "w") as f:
                f.write(resolved_conf_yaml)
            log.info("Resolved config saved to %s", config_path)

            file_handler.flush()
            tb_writer = SummaryWriter(tb_dir)
            formatted_config = resolved_conf_yaml.replace("\n", "  \n")
            tb_writer.add_text(
                "Hyperparameters",
                f"### Training Configuration\n```yaml\n{formatted_config}\n```",
                0,
            )

        # weights
        chamfer_w = float(cfg.objective.chamfer_weight)
        chamfer_scale = float(getattr(cfg.objective, "chamfer_scale", 1.0))
        edge_w_base = float(cfg.objective.edge_loss_weight)
        normal_w_base = float(cfg.objective.normal_weight)
        reg_warmup = int(getattr(cfg.objective, "reg_warmup_epochs", 0))

        # separation weights/settings: white vs pial per hemisphere
        hd_w_base = float(getattr(cfg.objective, "hd_weight", 0.0))
        hd_p = float(getattr(cfg.objective, "hd_p", 0.05))
        hd_lam = float(getattr(cfg.objective, "hd_lambda_mm", 0.5))
        Phd = int(getattr(cfg.objective, "hd_points", 30000))

        # Pial-LR separation: lh_pial vs rh_pial
        pial_lr_w_base = float(getattr(cfg.objective, "pial_lr_hd_weight", 0.0))
        pial_lr_p = float(getattr(cfg.objective, "pial_lr_hd_p", hd_p))
        pial_lr_lam = float(getattr(cfg.objective, "pial_lr_hd_lambda_mm", hd_lam))
        pial_lr_pts = int(getattr(cfg.objective, "pial_lr_hd_points", Phd))

        # Signed nested white-pial loss
        signed_w_base = float(getattr(cfg.objective, "signed_nested_weight", 0.0))
        signed_margin = float(getattr(cfg.objective, "signed_margin_mm", 0.5))
        signed_points = int(getattr(cfg.objective, "signed_points", 40000))

        # Jacobian folding penalty
        jac_w = float(getattr(cfg.objective, "jacobian_weight", 0.0))
        jac_margin = float(getattr(cfg.objective, "jacobian_margin", 0.05))
        jac_topk_frac = float(getattr(cfg.objective, "jacobian_topk_frac", 0.001))

        # train setup
        num_epochs = int(cfg.trainer.num_epochs)
        accum_steps = max(1, int(cfg.trainer.grad_accum_steps))
        grad_clip = float(cfg.trainer.grad_clip_norm)
        mesh_chunk = max(1, int(cfg.trainer.mesh_chunk))
        Ptrain = int(cfg.trainer.points_per_image)
        Pval = int(cfg.trainer.val_points_per_image)
        val_interval = max(1, int(cfg.trainer.validation_interval))
        col_interval = int(getattr(cfg.trainer, "collision_interval", val_interval))

        # Collision-aware checkpoint settings.
        # deform_best_model.pth is the final recommended model.
        alpha_wp = float(OmegaConf.select(cfg, "checkpoint.alpha_wp", default=0.006))
        alpha_lr = float(OmegaConf.select(cfg, "checkpoint.alpha_lr", default=0.002))
        rmse_guardrail_rel = float(OmegaConf.select(cfg, "checkpoint.rmse_guardrail_rel", default=1.06))
        score_delta = float(OmegaConf.select(cfg, "checkpoint.min_delta_score", default=1e-4))
        require_collision_for_best = bool(OmegaConf.select(cfg, "checkpoint.require_collision_for_best", default=True))

        if rank == 0:
            log.info(
                "Collision-aware checkpointing: alpha_wp=%.4f alpha_lr=%.4f "
                "rmse_guardrail_rel=%.4f min_delta_score=%.6f require_collision_for_best=%s",
                alpha_wp, alpha_lr, rmse_guardrail_rel, score_delta, require_collision_for_best,
            )
            if col_interval != val_interval:
                log.warning(
                    "collision_interval (%d) != validation_interval (%d). "
                    "Collision-aware model selection can only update on epochs with collision metrics. "
                    "Recommended: set collision_interval == validation_interval.",
                    col_interval, val_interval,
                )

        # Diagnostic best RMSE checkpoint.
        best_rmse_seen = float("inf")
        best_rmse_epoch = -1

        # Final model selection checkpoint.
        best_score = float("inf")
        best_model_epoch = -1

        no_improve = 0
        no_improve_rmse = 0
        early_patience = int(getattr(cfg.trainer, "early_stop_patience", 0))
        # RMSE delta is used only for diagnostic best-rmse checkpoint.
        early_rmse_delta = float(getattr(cfg.trainer, "early_stop_min_delta_mm", 0.0))

        start_epoch = 1
        resume_from = str(getattr(cfg.trainer, "resume_from", "") or "")
        if resume_from:
            if rank == 0:
                log.info("Resuming from full checkpoint: %s", resume_from)
            ckpt = torch.load(resume_from, map_location="cpu")
            if not isinstance(ckpt, dict) or "model" not in ckpt or "optimizer" not in ckpt:
                raise ValueError(
                    "trainer.resume_from must point to a full checkpoint containing at least "
                    "'model' and 'optimizer'. Use model.init_ckpt for model-only initialization."
                )

            net = model.module if hasattr(model, "module") else model
            net.load_state_dict(extract_model_state_dict(ckpt), strict=True)
            optimizer.load_state_dict(ckpt["optimizer"])
            move_optimizer_state_to_device(optimizer, device)

            if scheduler is not None and ckpt.get("scheduler", None) is not None:
                scheduler.load_state_dict(ckpt["scheduler"])

            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_score = float(ckpt.get("best_score", best_score))
            best_rmse_seen = float(ckpt.get("best_rmse_seen", best_rmse_seen))
            best_model_epoch = int(ckpt.get("best_model_epoch", best_model_epoch))
            best_rmse_epoch = int(ckpt.get("best_rmse_epoch", best_rmse_epoch))

            if rank == 0:
                log.info(
                    "Resume state loaded: start_epoch=%d best_score=%.6f best_rmse=%.6f "
                    "best_model_epoch=%d best_rmse_epoch=%d",
                    start_epoch, best_score, best_rmse_seen, best_model_epoch, best_rmse_epoch,
                )

        # -----------------------
        # Training loop
        # -----------------------
        for epoch in range(start_epoch, num_epochs + 1):
            if is_distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            if rank == 0:
                log.info("Epoch %d/%d", epoch, num_epochs)

            # warmup for regularizers (including separation penalties)
            t = 1.0
            if reg_warmup > 0:
                t = min(1.0, epoch / float(reg_warmup))
            edge_w = edge_w_base * t
            normal_w = normal_w_base * t
            hd_w_eff = hd_w_base * t
            pial_lr_w_eff = pial_lr_w_base * t
            signed_w_eff = signed_w_base * t
            jac_w_eff = jac_w * t

            model.train()
            optimizer.zero_grad(set_to_none=True)

            # epoch stats (sum over meshes)
            csq_sum = 0.0
            edge_sum = 0.0
            normal_sum = 0.0
            mesh_count = 0.0

            total_obj_sum = 0.0
            total_obj_count = 0.0

            # separation stats (sum over pairs)
            sep_pen_sum = 0.0
            sep_q_sum = 0.0
            sep_count = 0.0

            # Pial-LR stats (sum over pairs)
            pial_lr_pen_sum = 0.0
            pial_lr_sep_q_sum = 0.0
            pial_lr_count = 0.0

            jac_pen_sum = 0.0
            jac_neg_sum = 0.0
            jac_min_global = float("inf")
            jac_count = 0.0
            jac_p001_sum = 0.0
            jac_p01_sum = 0.0
            jac_p1_sum = 0.0

            # Signed nested stats
            signed_pen_sum = 0.0
            signed_badw_sum = 0.0
            signed_badp_sum = 0.0
            signed_wmean_sum = 0.0
            signed_pmean_sum = 0.0
            signed_count = 0.0


            surf_stats = {s: {"csq": 0.0, "count": 0.0} for s in surface_names}

            num_train_batches = len(train_loader)

            for batch_idx, batch in enumerate(tqdm(train_loader, disable=(rank != 0), desc=f"Train {epoch} [r{rank}]")):
                window_start = (batch_idx // accum_steps) * accum_steps
                current_accum_size = min(accum_steps, num_train_batches - window_start)
                is_last_micro_in_window = ((batch_idx + 1) == num_train_batches) or (((batch_idx + 1) % accum_steps) == 0)

                vol = batch["vol"].to(device, non_blocking=True)
                aff = batch["affine"].to(device, non_blocking=True)
                shift = batch["shift_ijk"].to(device, non_blocking=True)

                B, _, D, H, W = vol.shape

                lengths, padded_init, per_counts_init, init_faces_list, gt_verts_list, gt_faces_list = \
                    build_merged_init_and_metadata(batch, device, surface_names)

                # augmentation
                vol, padded_init, gt_verts_list = apply_aug(
                    vol=vol,
                    padded_init_ijk=padded_init,
                    lengths=lengths,
                    gt_verts_dict_list=gt_verts_list,
                    affines=aff,
                    cfg=cfg,
                    surface_names=surface_names,
                )

                # MRI-appearance augmentation (intensity/bias/noise), MRI channel only
                vol = apply_intensity_aug(vol, cfg)


                need_phis = (jac_w_eff > 0.0)

                if need_phis:
                    pred_vox, aux = model(
                        padded_init,
                        vol,
                        int(cfg.model.n_steps),
                        return_phis=True,
                    )
                else:
                    pred_vox = model(
                        padded_init,
                        vol,
                        int(cfg.model.n_steps),
                        return_phis=False,
                    )
                    aux = None

                loss_jac = torch.zeros((), device=device)
                jac_neg_pct = 0.0
                jac_min_det = 0.0
                jac_p001 = 0.0
                jac_p01 = 0.0
                jac_p1 = 0.0

                if need_phis:
                    loss_jac, jac_neg_pct, jac_min_det, jac_p001, jac_p01, jac_p1 = folding_penalty_from_phis(
                        aux["phis"],
                        margin=jac_margin,
                        topk_frac=jac_topk_frac,
                    )

                if jac_w_eff > 0.0:
                    jac_pen_sum += float(loss_jac.detach().item())
                    jac_neg_sum += float(jac_neg_pct)
                    jac_min_global = min(jac_min_global, float(jac_min_det))
                    jac_p001_sum += float(jac_p001)
                    jac_p01_sum += float(jac_p01)
                    jac_p1_sum += float(jac_p1)
                    jac_count += 1.0

                # Build mesh lists in WORLD(mm) for Chamfer/edge/normal
                pred_verts_mm, pred_faces = [], []
                gt_verts_mm, gt_faces = [], []
                surf_of_mesh = []

                # store pred meshes per sample for separation losses
                pred_mesh_mm_per_sample = [dict() for _ in range(B)]

                for i in range(B):
                    pred_i = pred_vox[i, :lengths[i]]
                    splits = torch.split(pred_i, per_counts_init[i], dim=0)

                    A = aff[i]
                    sh = shift[i].view(1, 3)

                    for j, s in enumerate(surface_names):
                        pv = splits[j]
                        gv = gt_verts_list[i][s]

                        f = init_faces_list[i][s]
                        gf = gt_faces_list[i][s]

                        pv_mm = voxel_to_world(pv - sh, A)
                        gv_mm = voxel_to_world(gv - sh, A)

                        # store pred mesh for separation losses if pred is valid
                        if mesh_is_valid(pv_mm, f):
                            pred_mesh_mm_per_sample[i][s] = (pv_mm, f)

                        # for chamfer/regularizers, need both pred and gt valid
                        if (not mesh_is_valid(pv_mm, f)) or (not mesh_is_valid(gv_mm, gf)):
                            continue

                        pred_verts_mm.append(pv_mm)
                        pred_faces.append(f)
                        gt_verts_mm.append(gv_mm)
                        gt_faces.append(gf)
                        surf_of_mesh.append(s)

                M = len(pred_verts_mm)
                if M == 0:
                    zero = sum(p.sum() * 0.0 for p in (model.module.parameters() if hasattr(model, "module") else model.parameters()))
                    loss_to_back = zero / float(current_accum_size)

                    sync_ctx = nullcontext()
                    if is_distributed and hasattr(model, "no_sync") and (not is_last_micro_in_window):
                        sync_ctx = model.no_sync()

                    with sync_ctx:
                        loss_to_back.backward()

                    if is_last_micro_in_window:
                        if grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    continue

                # -----------------------
                # White-pial separation penalty
                # -----------------------
                loss_sep = torch.zeros((), device=device)
                pair_count = 0
                sep_q_sum_batch = 0.0

                if hd_w_eff > 0.0:
                    for i in range(B):
                        md = pred_mesh_mm_per_sample[i]

                        if ("lh_white" in md) and ("lh_pial" in md):
                            vw, fw = md["lh_white"]
                            vp, fp = md["lh_pial"]
                            mw = Meshes(verts=[vw], faces=[fw])
                            mpial = Meshes(verts=[vp], faces=[fp])
                            sep_q, pen = partial_hd_penalty(mw, mpial, p=hd_p, lam=hd_lam, n_pts=Phd)
                            loss_sep = loss_sep + pen
                            sep_q_sum_batch += float(sep_q.detach().item())
                            pair_count += 1

                        if ("rh_white" in md) and ("rh_pial" in md):
                            vw, fw = md["rh_white"]
                            vp, fp = md["rh_pial"]
                            mw = Meshes(verts=[vw], faces=[fw])
                            mpial = Meshes(verts=[vp], faces=[fp])
                            sep_q, pen = partial_hd_penalty(mw, mpial, p=hd_p, lam=hd_lam, n_pts=Phd)
                            loss_sep = loss_sep + pen
                            sep_q_sum_batch += float(sep_q.detach().item())
                            pair_count += 1

                    if pair_count > 0:
                        loss_sep = loss_sep / float(pair_count)
       
                # -----------------------
                # Signed nested white-pial penalty
                # -----------------------
                loss_signed = torch.zeros((), device=device)
                signed_pair_count = 0
                signed_badw_batch_sum = 0.0
                signed_badp_batch_sum = 0.0
                signed_wmean_batch_sum = 0.0
                signed_pmean_batch_sum = 0.0

                if signed_w_eff > 0.0:
                    for i in range(B):
                        md = pred_mesh_mm_per_sample[i]

                        if ("lh_white" in md) and ("lh_pial" in md):
                            vw, fw = md["lh_white"]
                            vp, fp = md["lh_pial"]

                            lsgn, badw, badp, meanw, meanp = signed_nested_surface_penalty(
                                vw, fw, vp, fp,
                                margin_mm=signed_margin,
                                n_points=signed_points,
                            )
                            loss_signed = loss_signed + lsgn
                            signed_badw_batch_sum += badw
                            signed_badp_batch_sum += badp
                            signed_wmean_batch_sum += meanw
                            signed_pmean_batch_sum += meanp
                            signed_pair_count += 1

                        if ("rh_white" in md) and ("rh_pial" in md):
                            vw, fw = md["rh_white"]
                            vp, fp = md["rh_pial"]

                            lsgn, badw, badp, meanw, meanp = signed_nested_surface_penalty(
                                vw, fw, vp, fp,
                                margin_mm=signed_margin,
                                n_points=signed_points,
                            )
                            loss_signed = loss_signed + lsgn
                            signed_badw_batch_sum += badw
                            signed_badp_batch_sum += badp
                            signed_wmean_batch_sum += meanw
                            signed_pmean_batch_sum += meanp
                            signed_pair_count += 1

                    if signed_pair_count > 0:
                        loss_signed = loss_signed / float(signed_pair_count)
                # -----------------------
                # Pial-LR separation: lh_pial vs rh_pial
                # -----------------------
                loss_pial_lr = torch.zeros((), device=device)
                pial_lr_pair_count = 0
                pial_lr_sep_q_sum_batch = 0.0

                if pial_lr_w_eff > 0.0:
                    for i in range(B):
                        md = pred_mesh_mm_per_sample[i]
                        if ("lh_pial" in md) and ("rh_pial" in md):
                            vl, fl = md["lh_pial"]
                            vr, fr = md["rh_pial"]
                            ml = Meshes(verts=[vl], faces=[fl])
                            mr = Meshes(verts=[vr], faces=[fr])

                            sep_q_lr, pen_lr = partial_hd_penalty(
                                ml, mr, p=pial_lr_p, lam=pial_lr_lam, n_pts=pial_lr_pts
                            )
                            loss_pial_lr = loss_pial_lr + pen_lr
                            pial_lr_sep_q_sum_batch += float(sep_q_lr.detach().item())
                            pial_lr_pair_count += 1

                    if pial_lr_pair_count > 0:
                        loss_pial_lr = loss_pial_lr / float(pial_lr_pair_count)

                # -----------------------
                # Chamfer/edge/normal losses (chunked)
                # -----------------------
                loss_csq = torch.zeros((), device=device)
                loss_edge = torch.zeros((), device=device)
                loss_norm = torch.zeros((), device=device)

                csq_det_sum = 0.0

                for start in range(0, M, mesh_chunk):
                    end = min(M, start + mesh_chunk)

                    mpred = Meshes(verts=pred_verts_mm[start:end], faces=pred_faces[start:end])
                    mgt = Meshes(verts=gt_verts_mm[start:end], faces=gt_faces[start:end])

                    pp = sample_points_from_meshes(mpred, num_samples=Ptrain)
                    pg = sample_points_from_meshes(mgt, num_samples=Ptrain)

                    csq_per, _ = chamfer_distance(pp, pg, batch_reduction=None)
                    e = mesh_edge_loss(mpred)
                    n = mesh_normal_consistency(mpred)

                    mchunk = (end - start)

                    loss_csq = loss_csq + csq_per.mean() * mchunk
                    loss_edge = loss_edge + e * mchunk
                    loss_norm = loss_norm + n * mchunk

                    csq_det_sum += float(csq_per.detach().sum().item())
                    for k in range(mchunk):
                        ss = surf_of_mesh[start + k]
                        surf_stats[ss]["csq"] += float(csq_per[k].detach().item())
                        surf_stats[ss]["count"] += 1.0

                loss_csq = loss_csq / M
                loss_edge = loss_edge / M
                loss_norm = loss_norm / M

                # total loss
                total_loss = (
                    chamfer_w * (chamfer_scale * loss_csq)
                    + edge_w * loss_edge
                    + normal_w * loss_norm
                    + hd_w_eff * loss_sep
                    + pial_lr_w_eff * loss_pial_lr
                    + jac_w_eff * loss_jac
                    + signed_w_eff * loss_signed
                )
                
                loss_to_back = total_loss / float(current_accum_size)

                sync_ctx = nullcontext()
                if is_distributed and hasattr(model, "no_sync") and (not is_last_micro_in_window):
                    sync_ctx = model.no_sync()

                with sync_ctx:
                    loss_to_back.backward()

                if is_last_micro_in_window:
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                # stats
                csq_sum += csq_det_sum
                edge_sum += float((loss_edge.detach() * M).item())
                normal_sum += float((loss_norm.detach() * M).item())
                mesh_count += float(M)

                total_obj_sum += float(total_loss.detach().item())
                total_obj_count += 1.0

                if pair_count > 0:
                    sep_pen_sum += float((loss_sep.detach() * pair_count).item())
                    sep_q_sum += float(sep_q_sum_batch)
                    sep_count += float(pair_count)

                if pial_lr_pair_count > 0:
                    pial_lr_pen_sum += float((loss_pial_lr.detach() * pial_lr_pair_count).item())
                    pial_lr_sep_q_sum += float(pial_lr_sep_q_sum_batch)
                    pial_lr_count += float(pial_lr_pair_count)

                if signed_pair_count > 0:
                    signed_pen_sum += float((loss_signed.detach() * signed_pair_count).item())
                    signed_badw_sum += float(signed_badw_batch_sum)
                    signed_badp_sum += float(signed_badp_batch_sum)
                    signed_wmean_sum += float(signed_wmean_batch_sum)
                    signed_pmean_sum += float(signed_pmean_batch_sum)
                    signed_count += float(signed_pair_count)

    
            # reduce train stats
            if is_distributed:
                tstat = torch.tensor(
                    [
                        csq_sum, edge_sum, normal_sum, mesh_count,
                        total_obj_sum, total_obj_count,
                        sep_pen_sum, sep_q_sum, sep_count,
                        pial_lr_pen_sum, pial_lr_sep_q_sum, pial_lr_count,
                        signed_pen_sum, signed_badw_sum, signed_badp_sum,
                        signed_wmean_sum, signed_pmean_sum, signed_count,
                        jac_pen_sum, jac_neg_sum, jac_count,
                        jac_p001_sum, jac_p01_sum, jac_p1_sum,
                    ],
                    device=device, dtype=torch.float64,
                )
                dist.all_reduce(tstat, op=dist.ReduceOp.SUM)

                (
                    csq_sum, edge_sum, normal_sum, mesh_count,
                    total_obj_sum, total_obj_count,
                    sep_pen_sum, sep_q_sum, sep_count,
                    pial_lr_pen_sum, pial_lr_sep_q_sum, pial_lr_count,
                    signed_pen_sum, signed_badw_sum, signed_badp_sum,
                    signed_wmean_sum, signed_pmean_sum, signed_count,
                    jac_pen_sum, jac_neg_sum, jac_count,
                    jac_p001_sum, jac_p01_sum, jac_p1_sum,
                ) = tstat.tolist()

                jac_min_tensor = torch.tensor(jac_min_global, device=device, dtype=torch.float64)
                dist.all_reduce(jac_min_tensor, op=dist.ReduceOp.MIN)
                jac_min_global = float(jac_min_tensor.item())

                surf_tensor = torch.zeros((len(surface_names), 2), device=device, dtype=torch.float64)
                for i, s in enumerate(surface_names):
                    surf_tensor[i, 0] = surf_stats[s]["csq"]
                    surf_tensor[i, 1] = surf_stats[s]["count"]

                dist.all_reduce(surf_tensor, op=dist.ReduceOp.SUM)

                surf_global = {
                    s: {"csq": surf_tensor[i, 0].item(), "count": surf_tensor[i, 1].item()}
                    for i, s in enumerate(surface_names)
                }
            else:
                surf_global = surf_stats

            # log train
            if rank == 0 and mesh_count > 0:
                csq_mean = csq_sum / mesh_count
                rmse_mm_train = math.sqrt(max(csq_mean, 0.0))
                edge_mean = edge_sum / mesh_count
                norm_mean = normal_sum / mesh_count
                total_mean = total_obj_sum / max(total_obj_count, 1.0)

                if sep_count > 0:
                    sep_pen_mean = sep_pen_sum / sep_count
                    sep_q_mean_mm = sep_q_sum / sep_count
                else:
                    sep_pen_mean = 0.0
                    sep_q_mean_mm = 0.0

                if pial_lr_count > 0:
                    pial_lr_pen_mean = pial_lr_pen_sum / pial_lr_count
                    pial_lr_sep_q_mean = pial_lr_sep_q_sum / pial_lr_count
                else:
                    pial_lr_pen_mean = 0.0
                    pial_lr_sep_q_mean = 0.0

                if signed_count > 0:
                    signed_pen_mean = signed_pen_sum / signed_count
                    signed_badw_mean = signed_badw_sum / signed_count
                    signed_badp_mean = signed_badp_sum / signed_count
                    signed_wmean_mean = signed_wmean_sum / signed_count
                    signed_pmean_mean = signed_pmean_sum / signed_count
                else:
                    signed_pen_mean = 0.0
                    signed_badw_mean = 0.0
                    signed_badp_mean = 0.0
                    signed_wmean_mean = 0.0
                    signed_pmean_mean = 0.0
                    
                if jac_count > 0:
                    jac_pen_mean = jac_pen_sum / jac_count
                    jac_neg_mean = jac_neg_sum / jac_count
                    jac_min_mean = jac_min_global
                    jac_p001_mean = jac_p001_sum / jac_count
                    jac_p01_mean = jac_p01_sum / jac_count
                    jac_p1_mean = jac_p1_sum / jac_count
                else:
                    jac_pen_mean = 0.0
                    jac_neg_mean = 0.0
                    jac_min_mean = 0.0
                    jac_p001_mean = 0.0
                    jac_p01_mean = 0.0
                    jac_p1_mean = 0.0

                surf_str = ", ".join(
                    f"{s}={math.sqrt(max(surf_global[s]['csq']/max(surf_global[s]['count'],1.0),0.0)):.4f}mm"
                    for s in surface_names
                )

                log.info(
                    "Epoch %d [Train] | ChamferRMSE=%.4f mm | Edge=%.6f | Normal=%.6f | "
                    "SepPen=%.6f | SepQ=%.4f mm | wSep=%.4f | "
                    "PialLRSepPen=%.6f | PialLRSepQ=%.4f mm | wPialLR=%.4f | "
                    "SignedPen=%.6f | SignedBadW=%.2f%% | SignedBadP=%.2f%% | "
                    "SignedWMean=%.4f mm | SignedPMean=%.4f mm | wSigned=%.4f | "
                    "JacPen=%.6f | JacNeg=%.6f%% | JacMin=%.6f | "
                    "JacP0.01=%.6f | JacP0.1=%.6f | JacP1=%.6f | wJac=%.4f | "
                    "TotalObj=%.6f | Surfaces: %s",
                    epoch,
                    rmse_mm_train,
                    edge_mean,
                    norm_mean,
                    sep_pen_mean,
                    sep_q_mean_mm,
                    hd_w_eff,
                    pial_lr_pen_mean,
                    pial_lr_sep_q_mean,
                    pial_lr_w_eff,
                    signed_pen_mean,
                    signed_badw_mean,
                    signed_badp_mean,
                    signed_wmean_mean,
                    signed_pmean_mean,
                    signed_w_eff,
                    jac_pen_mean,
                    jac_neg_mean,
                    jac_min_mean,
                    jac_p001_mean,
                    jac_p01_mean,
                    jac_p1_mean,
                    jac_w_eff,
                    total_mean,
                    surf_str,
                )

                if tb_writer is not None:
                    net0 = model.module if hasattr(model, "module") else model
                    if hasattr(net0, "munet"):
                        for name, module in net0.munet.named_modules():
                            if hasattr(module, "gate_logit"):
                                gate_value = torch.sigmoid(module.gate_logit.detach()).item()
                                tb_writer.add_scalar(f"gates/{name}", gate_value, epoch)

                    tb_writer.add_scalar("train/rmse_mm", rmse_mm_train, epoch)
                    tb_writer.add_scalar("train/edge", edge_mean, epoch)
                    tb_writer.add_scalar("train/normal", norm_mean, epoch)
                    tb_writer.add_scalar("train/total_obj", total_mean, epoch)
                    tb_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

                    tb_writer.add_scalar("train/sep_penalty", sep_pen_mean, epoch)
                    tb_writer.add_scalar("train/sep_q_mean_mm", sep_q_mean_mm, epoch)
                    tb_writer.add_scalar("train/sep_weight_eff", hd_w_eff, epoch)

                    tb_writer.add_scalar("train/pial_lr_sep_penalty", pial_lr_pen_mean, epoch)
                    tb_writer.add_scalar("train/pial_lr_sep_q_mean_mm", pial_lr_sep_q_mean, epoch)
                    tb_writer.add_scalar("train/pial_lr_weight_eff", pial_lr_w_eff, epoch)

                    tb_writer.add_scalar("train/signed_penalty", signed_pen_mean, epoch)
                    tb_writer.add_scalar("train/signed_bad_white_pct", signed_badw_mean, epoch)
                    tb_writer.add_scalar("train/signed_bad_pial_pct", signed_badp_mean, epoch)
                    tb_writer.add_scalar("train/signed_white_mean_mm", signed_wmean_mean, epoch)
                    tb_writer.add_scalar("train/signed_pial_mean_mm", signed_pmean_mean, epoch)
                    tb_writer.add_scalar("train/signed_weight_eff", signed_w_eff, epoch)

                    tb_writer.add_scalar("train/jac_penalty", jac_pen_mean, epoch)
                    tb_writer.add_scalar("train/jac_neg_pct", jac_neg_mean, epoch)
                    tb_writer.add_scalar("train/jac_min_det", jac_min_mean, epoch)
                    tb_writer.add_scalar("train/jac_p001_det", jac_p001_mean, epoch)
                    tb_writer.add_scalar("train/jac_p01_det", jac_p01_mean, epoch)
                    tb_writer.add_scalar("train/jac_p1_det", jac_p1_mean, epoch)
                    tb_writer.add_scalar("train/jac_weight_eff", jac_w_eff, epoch)

            # -----------------------
            # Validation (rank0 only) + collisions
            # -----------------------
            stop_tensor = torch.tensor(0, device=device, dtype=torch.int64)
            collision_error_tensor = torch.tensor(0, device=device, dtype=torch.int64)

            if (epoch % val_interval) == 0:
                # Use underlying module to avoid DDP collectives in forward
                net = model.module if hasattr(model, "module") else model
                net.eval()

                do_collision_check = (epoch % col_interval == 0)
                rmse_tensor = torch.tensor(float("inf"), device=device, dtype=torch.float64)

                val_csq_sum = 0.0
                val_count = 0.0
                val_surf = {s: {"csq": 0.0, "count": 0.0} for s in surface_names}

                lh_total = lh_hit = lh_contacts_sum = 0.0
                rh_total = rh_hit = rh_contacts_sum = 0.0
                lr_total = lr_hit = lr_contacts_sum = 0.0

                if rank == 0:
                    with torch.no_grad():
                        for batch in tqdm(val_loader, disable=False, desc=f"Val {epoch} [rank0]"):
                            vol = batch["vol"].to(device, non_blocking=True)
                            aff = batch["affine"].to(device, non_blocking=True)
                            shift = batch["shift_ijk"].to(device, non_blocking=True)

                            B = vol.shape[0]

                            per_counts_init = []
                            merged_init_list = []
                            for i in range(B):
                                v_all = []
                                counts = []
                                for s in surface_names:
                                    v = batch["init_verts_vox"][i][s].to(device)
                                    v_all.append(v)
                                    counts.append(int(v.shape[0]))
                                per_counts_init.append(counts)
                                merged_init_list.append(torch.cat(v_all, dim=0))

                            lengths = torch.tensor([v.shape[0] for v in merged_init_list], device=device, dtype=torch.long)
                            Vmax = int(lengths.max().item())
                            padded_init = torch.zeros((B, Vmax, 3), device=device, dtype=merged_init_list[0].dtype)
                            for i in range(B):
                                padded_init[i, :lengths[i]] = merged_init_list[i]

                            pred_vox = net(padded_init, vol, int(cfg.model.n_steps))

                            for i in range(B):
                                A = aff[i]
                                sh = shift[i].view(1, 3)

                                pred_i = pred_vox[i, :lengths[i]]
                                splits = torch.split(pred_i, per_counts_init[i], dim=0)

                                pred_mm = {}
                                pred_f = {}

                                for j, s in enumerate(surface_names):
                                    pv = splits[j]
                                    gv = batch["gt_verts_vox"][i][s].to(device)

                                    pv_mm = voxel_to_world(pv - sh, A)
                                    gv_mm = voxel_to_world(gv - sh, A)

                                    f = batch["init_faces"][i][s].to(device).long()
                                    gf = batch["gt_faces"][i][s].to(device).long()

                                    if mesh_is_valid(pv_mm, f):
                                        pred_mm[s] = pv_mm
                                        pred_f[s] = f

                                    if (not mesh_is_valid(pv_mm, f)) or (not mesh_is_valid(gv_mm, gf)):
                                        continue

                                    mpred = Meshes(verts=[pv_mm], faces=[f])
                                    mgt = Meshes(verts=[gv_mm], faces=[gf])

                                    pp = sample_points_from_meshes(mpred, num_samples=Pval)
                                    pg = sample_points_from_meshes(mgt, num_samples=Pval)

                                    csq, _ = chamfer_distance(pp, pg)

                                    val_csq_sum += float(csq.item())
                                    val_count += 1.0
                                    val_surf[s]["csq"] += float(csq.item())
                                    val_surf[s]["count"] += 1.0

                                # collision checks
                                if do_collision_check and HAS_FCL:
                                    if ("lh_white" in pred_mm) and ("lh_pial" in pred_mm):
                                        is_col, ncon = count_collisions_inmemory(
                                            pred_mm["lh_white"], pred_f["lh_white"],
                                            pred_mm["lh_pial"], pred_f["lh_pial"],
                                        )
                                        if is_col is not None:
                                            lh_total += 1.0
                                            lh_hit += 1.0 if is_col else 0.0
                                            lh_contacts_sum += float(ncon)

                                    if ("rh_white" in pred_mm) and ("rh_pial" in pred_mm):
                                        is_col, ncon = count_collisions_inmemory(
                                            pred_mm["rh_white"], pred_f["rh_white"],
                                            pred_mm["rh_pial"], pred_f["rh_pial"],
                                        )
                                        if is_col is not None:
                                            rh_total += 1.0
                                            rh_hit += 1.0 if is_col else 0.0
                                            rh_contacts_sum += float(ncon)

                                    if ("lh_pial" in pred_mm) and ("rh_pial" in pred_mm):
                                        is_col, ncon = count_collisions_inmemory(
                                            pred_mm["lh_pial"], pred_f["lh_pial"],
                                            pred_mm["rh_pial"], pred_f["rh_pial"],
                                        )
                                        if is_col is not None:
                                            lr_total += 1.0
                                            lr_hit += 1.0 if is_col else 0.0
                                            lr_contacts_sum += float(ncon)

                    # log val + checkpoint
                    if val_count > 0:
                        csq_mean = val_csq_sum / val_count
                        rmse_mm = math.sqrt(max(csq_mean, 0.0))
                        rmse_tensor.fill_(rmse_mm)

                        surf_str = ", ".join(
                            f"{s}={math.sqrt(max(val_surf[s]['csq']/max(val_surf[s]['count'],1.0),0.0)):.4f}mm"
                            for s in surface_names
                        )
                        log.info("Epoch %d [Val] | ChamferRMSE=%.4f mm | Surfaces: %s", epoch, rmse_mm, surf_str)

                        collision_available = bool(do_collision_check and HAS_FCL and ((lh_total + rh_total) > 0 or lr_total > 0))
                        wp_pct = 0.0
                        lr_pct = 0.0
                        score = rmse_mm

                        if do_collision_check:
                            if not HAS_FCL:
                                log.info("Epoch %d [Val] | Collision check skipped (python-fcl not available).", epoch)
                            else:
                                log.info("Epoch %d [Val] | White–Pial Collisions LH: %s", epoch, fmt_collision_stats(lh_total, lh_hit, lh_contacts_sum))
                                log.info("Epoch %d [Val] | White–Pial Collisions RH: %s", epoch, fmt_collision_stats(rh_total, rh_hit, rh_contacts_sum))
                                log.info("Epoch %d [Val] | Pial–Pial Collisions LR: %s", epoch, fmt_collision_stats(lr_total, lr_hit, lr_contacts_sum))

                                wp_pct, lr_pct = compute_collision_percentages(
                                    lh_total, lh_hit, rh_total, rh_hit, lr_total, lr_hit
                                )
                                score = rmse_mm + alpha_wp * wp_pct + alpha_lr * lr_pct

                                log.info(
                                    "Epoch %d [ValScore] | Score=%.4f | RMSE=%.4f mm | "
                                    "WhitePial=%.2f%% | PialLR=%.2f%% | alpha_wp=%.4f | alpha_lr=%.4f",
                                    epoch, score, rmse_mm, wp_pct, lr_pct, alpha_wp, alpha_lr,
                                )
                        else:
                            log.info(
                                "Epoch %d [ValScore] | Collision metrics not scheduled this epoch. "
                                "RMSE-only score=%.4f. Recommended: collision_interval == validation_interval.",
                                epoch, score,
                            )

                        if tb_writer is not None:
                            tb_writer.add_scalar("val/rmse_mm", rmse_mm, epoch)

                            if collision_available:
                                total = lh_total + rh_total
                                hit = lh_hit + rh_hit
                                csum = lh_contacts_sum + rh_contacts_sum

                                if total > 0:
                                    tb_writer.add_scalar("collisions/whitepial_pct_pairs_colliding_total", wp_pct, epoch)
                                    tb_writer.add_scalar("collisions/whitepial_num_pairs_colliding_total", hit, epoch)
                                    tb_writer.add_scalar("collisions/whitepial_mean_contacts_all_total", csum / total, epoch)
                                    tb_writer.add_scalar("collisions/whitepial_mean_contacts_hit_total", csum / max(hit, 1.0), epoch)

                                if lr_total > 0:
                                    tb_writer.add_scalar("collisions/piallr_pct_pairs_colliding", lr_pct, epoch)
                                    tb_writer.add_scalar("collisions/piallr_num_pairs_colliding", lr_hit, epoch)
                                    tb_writer.add_scalar("collisions/piallr_mean_contacts_all", lr_contacts_sum / lr_total, epoch)
                                    tb_writer.add_scalar("collisions/piallr_mean_contacts_hit", lr_contacts_sum / max(lr_hit, 1.0), epoch)

                                tb_writer.add_scalar("val/collision_aware_score", score, epoch)
                                tb_writer.add_scalar("val/wp_collision_pct", wp_pct, epoch)
                                tb_writer.add_scalar("val/pial_lr_collision_pct", lr_pct, epoch)
                                tb_writer.add_scalar("val/best_collision_aware_score", best_score, epoch)

                        # Diagnostic best-RMSE checkpoint.
                        if rmse_mm < (best_rmse_seen - early_rmse_delta):
                            best_rmse_seen = rmse_mm
                            best_rmse_epoch = epoch
                            no_improve_rmse = 0
                            ckpt_rmse = os.path.join(out_root, "checkpoints", "deform_best_rmse.pth")
                            ckpt_rmse_full = os.path.join(out_root, "checkpoints", "deform_best_rmse_full.pth")

                            save_model_state(model, ckpt_rmse)
                            save_full_checkpoint(
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                path=ckpt_rmse_full,
                                epoch=epoch,
                                best_score=best_score,
                                best_rmse_seen=best_rmse_seen,
                                best_model_epoch=best_model_epoch,
                                best_rmse_epoch=best_rmse_epoch,
                                cfg=cfg,
                            )


                            log.info(
                                "🌟 Best RMSE checkpoint updated at epoch %d | RMSE=%.4f mm -> %s",
                                epoch, rmse_mm, ckpt_rmse,
                            )
                        else:
                            no_improve_rmse += 1

                        reasonable = rmse_mm <= best_rmse_seen * rmse_guardrail_rel
                        ckpt_model = os.path.join(out_root, "checkpoints", "deform_best_model.pth")
                        ckpt_model_full = os.path.join(out_root, "checkpoints", "deform_best_model_full.pth")

                        if collision_available:
                            if reasonable and score < (best_score - score_delta):
                                best_score = score
                                best_model_epoch = epoch
                                no_improve = 0
                                save_model_state(model, ckpt_model)
                                save_full_checkpoint(
                                    model=model,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    path=ckpt_model_full,
                                    epoch=epoch,
                                    best_score=best_score,
                                    best_rmse_seen=best_rmse_seen,
                                    best_model_epoch=best_model_epoch,
                                    best_rmse_epoch=best_rmse_epoch,
                                    cfg=cfg,
                                )

                                log.info(
                                    "🌟 Best collision-aware model updated at epoch %d | "
                                    "Score=%.4f | RMSE=%.4f mm | WP=%.2f%% | PialLR=%.2f%% | "
                                    "Guardrail=%.4f mm | BestRMSE=%.4f mm -> %s",
                                    epoch, score, rmse_mm, wp_pct, lr_pct,
                                    best_rmse_seen * rmse_guardrail_rel, best_rmse_seen, ckpt_model,
                                )
                            else:
                                no_improve += 1
                                log.info(
                                    "Epoch %d [ValScore] | No score improvement | "
                                    "Score=%.4f | BestScore=%.4f | RMSE=%.4f mm | "
                                    "Reasonable=%s | BestModelEpoch=%d | no_improve=%d",
                                    epoch, score, best_score, rmse_mm,
                                    reasonable, best_model_epoch, no_improve,
                                )

                        else:
                            if require_collision_for_best:
                                collision_error_tensor.fill_(1)
                                log.error(
                                    "Epoch %d [ValScore] | Collision metrics are unavailable, but "
                                    "checkpoint.require_collision_for_best=True. Install/enable collision checking "
                                    "or set checkpoint.require_collision_for_best=False.",
                                    epoch,
                                )
                            else:
                                fallback_score = rmse_mm

                                if fallback_score < (best_score - score_delta):
                                    best_score = fallback_score
                                    best_model_epoch = epoch
                                    no_improve = 0
                                    save_model_state(model, ckpt_model)
                                    save_full_checkpoint(
                                        model=model,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        path=ckpt_model_full,
                                        epoch=epoch,
                                        best_score=best_score,
                                        best_rmse_seen=best_rmse_seen,
                                        best_model_epoch=best_model_epoch,
                                        best_rmse_epoch=best_rmse_epoch,
                                        cfg=cfg,
                                    )
                                    log.info(
                                        "🌟 Best RMSE-fallback model updated at epoch %d | "
                                        "RMSE=%.4f mm | collision_available=False | "
                                        "require_collision_for_best=False -> %s",
                                        epoch, rmse_mm, ckpt_model,
                                    )
                                else:
                                    no_improve += 1
                                    log.info(
                                        "Epoch %d [ValScore] | No RMSE improvement "
                                        "(collision unavailable fallback) | RMSE=%.4f mm | "
                                        "BestScore=%.4f | BestModelEpoch=%d | no_improve=%d",
                                        epoch, rmse_mm, best_score, best_model_epoch, no_improve,
                                    )
                        # Early stopping follows the RMSE plateau (decoupled from the
                        # collision-aware score). The score still selects best_model, but
                        # training continues as long as val RMSE keeps improving, so a noisy
                        # WP/LR score (single-subject quantization) can no longer cut training
                        # short while accuracy is still improving.
                        if early_patience > 0 and no_improve_rmse >= early_patience:
                            log.info(
                                "🛑 Early stopping after %d validation checks without RMSE improvement. "
                                "BestScore=%.4f at epoch %d | BestRMSE=%.4f at epoch %d | "
                                "no_improve(score)=%d no_improve(rmse)=%d",
                                early_patience, best_score, best_model_epoch, best_rmse_seen, best_rmse_epoch,
                                no_improve, no_improve_rmse,
                            )
                            stop_tensor.fill_(1)


                if is_distributed:
                    dist.broadcast(rmse_tensor, src=0)
                    dist.broadcast(collision_error_tensor, src=0)

                shared_rmse_mm = float(rmse_tensor.item())
                if math.isfinite(shared_rmse_mm):
                    scheduler.step(shared_rmse_mm)

                if rank == 0:
                    ckpt_last_full = os.path.join(out_root, "checkpoints", "deform_last_full.pth")
                    save_full_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        path=ckpt_last_full,
                        epoch=epoch,
                        best_score=best_score,
                        best_rmse_seen=best_rmse_seen,
                        best_model_epoch=best_model_epoch,
                        best_rmse_epoch=best_rmse_epoch,
                        cfg=cfg,
                    )

                if collision_error_tensor.item() == 1:
                    raise RuntimeError(
                        "Collision metrics are unavailable while "
                        "checkpoint.require_collision_for_best=True."
                    )

                net.train()

            # Sync early-stop decision across ranks
            if is_distributed:
                dist.broadcast(stop_tensor, src=0)

            if stop_tensor.item() == 1:
                break

        if tb_writer is not None:
            tb_writer.close()

    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
