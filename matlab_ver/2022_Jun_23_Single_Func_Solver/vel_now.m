function [zs, vel_profile] = vel_now(v0,h,L,t,dt,wmaxmin,wmin)
% Takes v at entry, width, and current level h, time t.
npts = 1000;
zs = linspace(-L,h,npts);
vel_profile = zeros(npts,1);
dwdt = zeros(npts,1);
ws = zeros(npts,1);
for i=1:npts
    dwdt(i) = (width_new(zs(i),t+dt,wmaxmin,wmin) - width_new(zs(i),t-dt,wmaxmin,wmin))/(dt*2);
    ws(i) = width_new(zs(i), t,wmaxmin,wmin);
end
dvdz = -dwdt./ws;
% Using trapezoidal rule to integrate...
vel_profile(1) = v0;
for i=2:npts
    vel_profile(i) = vel_profile(i-1) + 0.5*(dvdz(i-1)+dvdz(i))*(zs(i)-zs(i-1));
end
end

