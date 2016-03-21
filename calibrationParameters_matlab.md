
## Stereo calibration parameters:


Intrinsic parameters of left camera:

Focal Length:          fc_left = [ 2222.72426   2190.48031 ] ± [ 21.51931   45.24327 ]
Principal point:       cc_left = [ 681.42537   -22.08306 ] ± [ 0.00000   0.00000 ]
Skew:             alpha_c_left = [ 0.00000 ] ± [ 0.00000  ]   => angle of pixel axes = 90.00000 ± 0.00000 degrees
Distortion:            kc_left = [ 0.27724   0.28163   -0.06867   0.00358  0.00000 ] ± [ 0.02982   0.11206   0.00436   0.00097  0.00000 ]

# 
fxL = 2222.72426
fyL = 2190.48031

cxL = 681.42537
cyL = -22.08306

k1L =  0.27724
k2L = 0.28163
k3L = -0.06867
k4L = 0.00358
k5L = 0.00000 


Intrinsic parameters of right camera:

Focal Length:          fc_right = [ 2226.10095   2195.17250 ] ± [ 23.92448   48.37895 ]
Principal point:       cc_right = [ 637.64260   -33.60849 ] ± [ 12.12391   22.50961 ]
Skew:             alpha_c_right = [ 0.00000 ] ± [ 0.00000  ]   => angle of pixel axes = 90.00000 ± 0.00000 degrees
Distortion:            kc_right = [ 0.29407   0.29892   -0.08315   -0.01218  0.00000 ] ± [ 0.02722   0.09309   0.00715   0.00366  0.00000 ]

# 
fxR = 2226.10095
fyR = 2195.17250

cxR = 637.64260
cyR = -33.60849


k1R =  0.29407
k2R = 0.29892
k3R = 0-0.08315
k4R = -0.01218
k5R = 0.00000


Extrinsic parameters (position of right camera wrt left camera):

Rotation vector:             om = [ -0.04129   0.01292  -0.09670 ] ± [ 0.00969   0.00552  0.00071 ]
Translation vector:           T = [ 303.48269   -19.19528  29.06076 ] ± [ 2.17912   5.09096  18.22290 ]


Note: The numerical errors are approximately three times the standard deviations (for reference).


## intrinsic_matrix, distCoeff


intrinsic_matrix = [fx skew x0;
                    0 fy y0;
                    0 0 1]
                    
 distCoeff =[k1 k2 p1 p2 k3]