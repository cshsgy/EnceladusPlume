deltas = [0.2,0.3,0.5];
depths = [1000,1200,1400,1600,1800,2000];

generate_r_parallel(deltas,depths);

[r,phi,rho,phi0] = interp_r_function(273.15,1500,0.25)
% Expected:r~0.7447, phi<phi0
