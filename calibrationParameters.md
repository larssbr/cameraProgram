# Calibration is done in agisoft using cameratype: fisheye
Agisoft Lens software uses a pinhole camera model for lens calibration. The distortions are modeled using
Brown's distortion model.

The camera model specifies the transformation from point coordinates in the local camera coordinate
system to the pixel coordinates in the image frame.


## Parameters
(fx, fy) - focal lengths,
(cx, cy) - principal point coordinates --> optical centers expressed in pixels coordinates

K1, K2, K3 - radial distortion coefficients, --> from fish eye effect
P1, P2 - tangential distortion coefficients --> Tangential distortion occurs because the image taking lenses are not perfectly parallel to the imaging plane.
skew - skew coefficient between the x and the y axis.


## Left camera calibration parameters:
(from 267 images)

fx=1792.64
fy=1792.64

k1=0.523977
k2=0.963942
k3=1.87865
k4=0

skew=0

cx=694.684
cy=492.203

p1=o
p2=0
p3=0
p4=0

## Right camera calibration parameters:
(from 267 images)

fx=1806.11
fy=1806.11

k1=0.521372
k2=0.817687
k3=2.79161
k4=0

skew=0

cx=662.218
cy=464.514

p1=o
p2=0
p3=0
p4=0


## intrinsic_matrix, distCoeff


intrinsic_matrix = [fx skew x0;
                    0 fy y0;
                    0 0 1]
                    
 distCoeff =[k1 k2 p1 p2 k3]