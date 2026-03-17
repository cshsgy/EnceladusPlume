function [Tw,Ev,MTop,PhiTop] = MikiModelFull(r,width,zs,T2,K,Lv,G,Cd,Dx,hmax)
% r is the value to be specified
% Notice that G is the gravity in SI units
% Fixed constants
% Gas constant of vapor. Universally constant so is not an input
% *** NOTE: the output of mass flux is of unit length & unit width...
rg = 8.341/0.018;
Tb = 273.15;
dz = zs(2)-zs(1); % Consider uniform grid here

% Calculate the boundaries
phi0 = (1-r)*VaporPressure(Tb)/sqrt(2*pi*rg*Tb);
rho0 = VaporPressure(Tb)/(rg*Tb)*r;

% More initializations
rho = zeros(length(zs),1); % Density
M = zeros(length(zs),1); % Mach Number
T = zeros(length(zs),1); % Gas Temperature
Tw = zeros(length(zs),1); % Wall Temperature
Ev = zeros(length(zs),1); % Wall Evaporation (one side)
rho(1) = rho0;
M(1) = phi0/rho0/sqrt(1.33*rg*Tb);
T(1) = Tb;
phi = phi0;
Tw(1) = Tb;

% Main solution process. runge kutta method applied
for i=2:length(zs)
    dn = rho(i-1);
    pb = dn*rg*T(i-1);
    [Tw(i),Ev(i)] = MikiFindEvap(T2(i),T(i-1),pb,Dx,K,Lv);
    if zs(i)>hmax && Ev(i)>0
      Ev(i) = 0;
    end
    if Ev(i)>0
        T(i) = (phi*T(i-1)+2*dz*Ev(i)*Tw(i))/(phi+2*Ev(i));
    else
        T(i) = T(i-1);
    end
    pbt = dn*rg*T(i);
    if i<length(T2)
        [~,Evt] = MikiFindEvap(T2(i+1),T(i),pbt,Dx,K,Lv);
    else
        [~,Evt] = MikiFindEvap(T2(i),T(i),pbt,Dx,K,Lv);
    end
    if zs(i)>hmax && Evt>0
      Evt = 0;
    end
    v = (phi+2*dz*Ev(i))/dn;
    M(i) = v/sqrt(1.33*rg*T(i-1));
    k1 = (2*Cd*dn*v^2/width+dn*G)/(v^2-rg*T(i-1));

    dn = rho(i-1)+(zs(i)-zs(i-1))*0.5*k1;
    v = (phi+dz*(Ev(i)+Evt))/dn;
    k2 = (2*Cd*dn*v^2/width+dn*G)/(v^2-rg*(T(i-1)+T(i))/2);

    dn = rho(i-1)+(zs(i)-zs(i-1))*0.5*k2;
    v = (phi+dz*(Ev(i)+Evt))/dn;
    k3 = (2*Cd*dn*v^2/width+dn*G)/(v^2-rg*(T(i-1)+T(i))/2);

    dn = rho(i-1)+(zs(i)-zs(i-1))*k3;
    v = (phi+2*dz*Evt)/dn;
    k4 = (2*Cd*dn*v^2/width+dn*G)/(v^2-rg*T(i));
    rho(i) = rho(i-1)+(zs(i)-zs(i-1))*1/6*(k1+k4+2*k2+2*k3);
    if rho(i)-rho(i-1)>0
        % rho too small, force to be negative
        rho(i) = -1;
    end
    if rho(i)<0
        % rho becomes too small
        break;
    end
    if M(i)>1.6
        break;
    end
    phi = phi+dz*Ev(i);
    if phi<0
        break;
    end
end

Ev(1) = Ev(2); % To keep the interpolation right

if((rho(i)>0)&&(phi>0))
    MTop = M(i);
    PhiTop = phi;
else
    MTop = -1;
    PhiTop = phi; 
    % NOTE: when MTop<0, PhiTop>0 then increase r, otherwise reduce r.
end
end

