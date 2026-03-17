function [result] = pvpt(zs, t, dt,wmaxmin,wmin)
% Essentially partial v partial t peeling off dv0/dt
npts = length(zs);
integrand = zeros(npts,1);
result = 0;    
w = width_new(0,t,wmaxmin,wmin);
dwdt = (width_new(0, t+dt,wmaxmin,wmin)-width_new(0, t-dt,wmaxmin,wmin))/(2*dt);
dwdt2 = (width_new(0,t+dt,wmaxmin,wmin)-2*width_new(0,t,wmaxmin,wmin)+width_new(0,t-dt,wmaxmin,wmin))/(dt^2);
integrand = dwdt^2/w^2-dwdt2/w;
result = integrand*(zs(end)-zs(1));
end
