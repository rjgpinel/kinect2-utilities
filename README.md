# kintools

## Calibration

Calibrate a camera using recorded images of a Charuco board (e.g. 50 images):
```
python -m kintools.calibration path_to_images/
```

## Pose Estimation

Markers pose estimation using camera calibration parameters:
```
python -m kintools.pose_estimation cam_params.pkl path_to_image
```

## Reconstruction

Convert depth image from kinect2 to point cloud:
```
python -m kintools.reconstruct -i depth/depth.npy -o cloud/res.npy
```

Display point cloud:
```
python -m kintools.show cloud/res.npy
```
