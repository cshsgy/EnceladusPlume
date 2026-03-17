function [zs, vel_profile] = vel_now(v0,h,L,w,dwdt)
% Takes v at entry, width, and current level h, time t.
npts = 1000;
zs = linspace(-L,h,npts);
vel_profile = zeros(npts,1);
dvdz = -dwdt/w;
% Using trapezoidal rule to integrate...
vel_profile(1) = v0;
for i=2:npts
    vel_profile(i) = vel_profile(i-1) + dvdz*(zs(i)-zs(i-1));
end
end

