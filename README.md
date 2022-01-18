A Pytorch3D-based registration method between a reconstructed point cloud (e.g., the output of PIFuHD) and an estimated SMPL mesh (e.g., ProHMR).

## Install

```
python3 setup.py develop
```

## Point cloud and SMPL estimation results

0. Run PIFu code to estimate the point cloud.

0. Run ProHMR code to generate an initial SMPL estimation.

## Running

```
python3 demo.py
```

The results will be saved in the ```outputs``` folder.
