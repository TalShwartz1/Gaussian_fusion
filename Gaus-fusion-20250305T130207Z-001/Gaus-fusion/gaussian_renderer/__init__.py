# yaniv - modified to change the camera position -----------------------------------------------------------------------
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, separate_sh=False, override_color=None, use_trained_exp=False):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create a tensor for screen-space points (for computing gradients).
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype,
                                          requires_grad=True, device="cuda")
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    # Compute tangents of half field-of-view angles.
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # --- MODIFYING CAMERA POSITION ---
    # Instead of just modifying camera_center, update the view transformation.
    # Clone the original view matrix.
    modified_viewmatrix = viewpoint_camera.world_view_transform.clone()
    # For a 4x4 transform (assumed row-major with translation in the 4th row),

    # scale the translation part (row index 3, first 3 columns) to move the camera closer.
    factor_x = 1  # scale x by 0.5
    factor_y = 1  # scale y by 0.75
    factor_z = 1  # scale z by 0.5

    modified_viewmatrix[3, 0] = modified_viewmatrix[3, 0] * factor_x
    modified_viewmatrix[3, 1] = modified_viewmatrix[3, 1] * factor_y
    modified_viewmatrix[3, 2] = modified_viewmatrix[3, 2] * factor_z


    # Recompute the camera center from the modified view matrix.
    modified_camera_center = modified_viewmatrix.inverse()[3, :3]
    # Recompute the full projection transform using the modified view matrix.
    modified_full_proj_transform = (
        modified_viewmatrix.unsqueeze(0).bmm(viewpoint_camera.projection_matrix.unsqueeze(0))
    ).squeeze(0)
    # --- END MODIFYING CAMERA POSITION ---

    # Set up rasterization configuration using the modified matrices.
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=modified_viewmatrix,
        projmatrix=modified_full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=modified_camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3D covariance is provided, use it; otherwise use scaling/rotation.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # Prepare color information (using spherical harmonics or override).
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - modified_camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize the visible Gaussians.
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            dc=dc,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)

    # Apply exposure if needed.
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]) \
                             .permute(2, 0, 1) + exposure[:3, 3, None, None]

    # Clamp rendered image values and prepare output.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "depth": depth_image
    }

    return out
