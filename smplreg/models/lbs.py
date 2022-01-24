""" Modified lbs functions to animate a registered mesh of any number of vertices.
"""

from smplx.lbs import (
    Tensor,
    Tuple,
    blend_shapes,
    vertices2joints,
    batch_rodrigues,
    batch_rigid_transform,
)


def joint_transform(
    betas: Tensor,
    pose: Tensor,
    v_template: Tensor,
    shapedirs: Tensor,
    J_regressor: Tensor,
    parents: Tensor,
    pose2rot: bool = True,
    detail: Tensor = None,
) -> Tuple[Tensor, Tensor]:
    """Obtain Joint transformations the given shape and pose parameters
       Modified from the original lbs function of smplx
    Parameters
    ----------
    betas : torch.tensor BxNB
        The tensor of shape parameters
    pose : torch.tensor Bx(J + 1) * 3
        The pose parameters in axis-angle format
    v_template torch.tensor BxVx3
        The template mesh that will be deformed
    shapedirs : torch.tensor 1xNB
        The tensor of PCA shape displacements
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from
        the position of the vertices
    parents: torch.tensor J
        The array that describes the kinematic tree for the model
    pose2rot: bool, optional
        Flag on whether to convert the input pose tensor to rotation
        matrices. The default value is True. If False, then the pose tensor
        should already contain rotation matrices and have a size of
        Bx(J + 1)x9
    detail torch.tensor BxVx3
        Per vertex offset added to the template mesh that will be deformed
    dtype: torch.dtype, optional

    Returns
    -------
    A: torch.tensor BxJx3
        Rotation transformation of joints
    """

    batch_size = max(betas.shape[0], pose.shape[0])

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)
    # Add per vertex offset
    if detail is not None:
        v_shaped = v_shaped + detail

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view([batch_size, -1, 3, 3])
    else:
        rot_mats = pose.view(batch_size, -1, 3, 3)

    # 4. Get the global joint location
    _, A = batch_rigid_transform(rot_mats, J, parents, dtype=betas.dtype)

    return A
