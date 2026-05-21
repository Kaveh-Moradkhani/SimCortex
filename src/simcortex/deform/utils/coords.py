# util/coords.py
from __future__ import annotations
import torch


def world_to_voxel(verts_mm: torch.Tensor, affine_vox2world: torch.Tensor) -> torch.Tensor:
    """
    verts_mm: (V,3) in world mm (RAS/MNI)
    affine_vox2world: (4,4)
    returns verts_vox: (V,3) in voxel index coords (I,J,K) matching volume array indexing (D,H,W)
    """
    if verts_mm.numel() == 0:
        return verts_mm

    A = affine_vox2world.to(device=verts_mm.device, dtype=verts_mm.dtype)
    invA = torch.linalg.inv(A)

    V = verts_mm.shape[0]
    ones = torch.ones((V, 1), device=verts_mm.device, dtype=verts_mm.dtype)
    homog = torch.cat([verts_mm, ones], dim=1)  # (V,4)
    vox = (invA @ homog.t()).t()[:, :3]
    return vox


def voxel_to_world(verts_vox: torch.Tensor, affine_vox2world: torch.Tensor) -> torch.Tensor:
    """
    verts_vox: (V,3) voxel indices (I,J,K)
    affine_vox2world: (4,4)
    returns verts_mm: (V,3)
    """
    if verts_vox.numel() == 0:
        return verts_vox

    A = affine_vox2world.to(device=verts_vox.device, dtype=verts_vox.dtype)
    V = verts_vox.shape[0]
    ones = torch.ones((V, 1), device=verts_vox.device, dtype=verts_vox.dtype)
    homog = torch.cat([verts_vox, ones], dim=1)
    mm = (A @ homog.t()).t()[:, :3]
    return mm


def voxel_to_ndc_ijk(verts_vox: torch.Tensor, inshape_dhw: torch.Tensor) -> torch.Tensor:
    """
    verts_vox in IJK order matching volume (D,H,W).
    inshape_dhw is (3,) tensor [D,H,W]
    returns NDC verts in same IJK axis order, values in [-1,1] using align_corners=True convention.
    """
    den = (inshape_dhw.to(verts_vox.device, verts_vox.dtype) - 1.0).clamp(min=1.0)
    return 2.0 * (verts_vox / den) - 1.0


def ndc_to_voxel_ijk(verts_ndc: torch.Tensor, inshape_dhw: torch.Tensor) -> torch.Tensor:
    den = (inshape_dhw.to(verts_ndc.device, verts_ndc.dtype) - 1.0).clamp(min=1.0)
    return 0.5 * (verts_ndc + 1.0) * den


def make_center_crop_pad_slices(src_shape_dhw, tgt_shape_dhw):
    """
    Compute crop slices and pad (before, after) for center crop/pad to target shape.
    Returns:
      crop_slices: tuple(slice_d, slice_h, slice_w) to apply on src
      pad_before: (3,) ints for (D,H,W)
      pad_after:  (3,) ints for (D,H,W)
      crop_before: (3,) ints for (D,H,W) removed from the low end
    """
    D0, H0, W0 = src_shape_dhw
    D1, H1, W1 = tgt_shape_dhw

    def one_axis(n0, n1):
        if n0 >= n1:
            cb = (n0 - n1) // 2
            ca = (n0 - n1) - cb
            pb, pa = 0, 0
            sl = slice(cb, n0 - ca)
        else:
            pb = (n1 - n0) // 2
            pa = (n1 - n0) - pb
            cb, ca = 0, 0
            sl = slice(0, n0)
        return sl, pb, pa, cb

    sd, pbd, pad, cbd = one_axis(D0, D1)
    sh, pbh, pah, cbh = one_axis(H0, H1)
    sw, pbw, paw, cbw = one_axis(W0, W1)

    crop_slices = (sd, sh, sw)
    pad_before = (pbd, pbh, pbw)
    pad_after  = (pad, pah, paw)
    crop_before = (cbd, cbh, cbw)
    return crop_slices, pad_before, pad_after, crop_before
