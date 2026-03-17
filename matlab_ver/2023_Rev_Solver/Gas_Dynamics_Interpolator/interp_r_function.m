function [r,phi,rho,phi0] = interp_r_function(T,d,w)
load('r_rec.mat','Tb','depth','delta','phi_rec','phi0_rec','r_rec','rho_rec');
% Assume that only interpolation exists. Any extrapolation would cause
% error.
iT = sum(Tb<T);
id = sum(depth<d);
iw = sum(delta<=w);
if iw==0
        iw = 1; % extrap for smaller width...
end
if id*iT*iw==0 || id==length(depth) || iw==length(delta)
	% disp('Interpolation function: out of range!');
        r = nan;
        phi = nan;
	phi0 = nan;
	rho = nan;
        return;
end
Td = (T-Tb(iT))/(Tb(iT+1)-Tb(iT));
dd = (d-depth(id))/(depth(id+1)-depth(id));
wd = (w-delta(iw))/(delta(iw+1)-delta(iw));

% Find the r
c00 = r_rec(iw,id,iT)*(1-Td)+r_rec(iw,id,iT+1)*Td;
c01 = r_rec(iw+1,id,iT)*(1-Td)+r_rec(iw+1,id,iT+1)*Td;
c10 = r_rec(iw,id+1,iT)*(1-Td)+r_rec(iw,id+1,iT+1)*Td;
c11 = r_rec(iw+1,id+1,iT)*(1-Td)+r_rec(iw+1,id+1,iT+1)*Td;
c0 = c00*(1-dd)+c10*dd;
c1 = c01*(1-dd)+c11*dd;
r = c0*(1-wd)+c1*wd;

% Find the top evap
c00 = phi_rec(iw,id,iT)*(1-Td)+phi_rec(iw,id,iT+1)*Td;
c01 = phi_rec(iw+1,id,iT)*(1-Td)+phi_rec(iw+1,id,iT+1)*Td;
c10 = phi_rec(iw,id+1,iT)*(1-Td)+phi_rec(iw,id+1,iT+1)*Td;
c11 = phi_rec(iw+1,id+1,iT)*(1-Td)+phi_rec(iw+1,id+1,iT+1)*Td;
c0 = c00*(1-dd)+c10*dd;
c1 = c01*(1-dd)+c11*dd;
phi = c0*(1-wd)+c1*wd;

% Find the top rho
c00 = rho_rec(iw,id,iT)*(1-Td)+rho_rec(iw,id,iT+1)*Td;
c01 = rho_rec(iw+1,id,iT)*(1-Td)+rho_rec(iw+1,id,iT+1)*Td;
c10 = rho_rec(iw,id+1,iT)*(1-Td)+rho_rec(iw,id+1,iT+1)*Td;
c11 = rho_rec(iw+1,id+1,iT)*(1-Td)+rho_rec(iw+1,id+1,iT+1)*Td;
c0 = c00*(1-dd)+c10*dd;
c1 = c01*(1-dd)+c11*dd;
rho = c0*(1-wd)+c1*wd;

% Find the equivalent T
c00 = phi0_rec(iw,id,iT)*(1-Td)+phi0_rec(iw,id,iT+1)*Td;
c01 = phi0_rec(iw+1,id,iT)*(1-Td)+phi0_rec(iw+1,id,iT+1)*Td;
c10 = phi0_rec(iw,id+1,iT)*(1-Td)+phi0_rec(iw,id+1,iT+1)*Td;
c11 = phi0_rec(iw+1,id+1,iT)*(1-Td)+phi0_rec(iw+1,id+1,iT+1)*Td;
c0 = c00*(1-dd)+c10*dd;
c1 = c01*(1-dd)+c11*dd;
phi0 = c0*(1-wd)+c1*wd;
end
