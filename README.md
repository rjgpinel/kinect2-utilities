# kintools

Convert depth image from kinect2 to point cloud:
```
python -m kintools.process -i depth/depth.npy -o cloud/res.npy
```

Display point cloud:
```
python -m kintools.show cloud/res.npy
```
