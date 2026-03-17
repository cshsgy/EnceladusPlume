function [friction] = friction(zs, v, t,wmaxmin,wmin)
% Integrate the friction term in the RHS. Note that minus sign NOT
% included.
npts = length(zs);
dfdt = zeros(npts,1);
dz = 0.1;
Cf = 0.004; % Can also be in the form of a function...
friction = zeros(npts,1);
for i=1:npts
      dfdt(i) = 2*Cf/width_new(zs(i),t,wmaxmin,wmin)*v(i)^2;
      dwdz = 0; % (width_new(zs(i)+dz,t,wmaxmin,wmin)-width_new(zs(i)-dz,t,wmaxmin,wmin))/(2*dz);
      dfdt(i) = dfdt(i)/sqrt(1+0.25*dwdz^2);
      if v(i)<0
        dfdt(i) = dfdt(i)*-1;
      end
      if i>1
        friction(i) = friction(i-1)+0.5*(dfdt(i-1)+dfdt(i))*(zs(i)-zs(i-1));
      end
end
friction = friction(end);
end

