function [phi_to_zero,rho_to_zero,mach_top,phi_top,rho_top,phi0] = solve_function(Tb,depth,width,r)
% Fixed constants
kt = 2.4; % Thermal conductivity
lv = 2.8e6; % Latent heat of vaporization
rg = 8.341/0.018; % Gas constant of vapor
g = 0.113; % Gravitational acceleration
ec = vapor_press(273.15)/(2*pi*rg)^0.5;
bv = lv/rg;
Cd = 0.002;

% Calculate the boundaries
phi = (1-r)*evaporation(ec,bv,Tb);
phi0 = phi;
rho0 = vapor_press(Tb)/(rg*Tb)*r;

% integrate the f(z)
z_f = 0:0.1:depth;
f = zeros(length(z_f),1);
dfdz = zeros(length(z_f),1);
f_now = 0;
phi_to_zero = depth;
flag = 0;
for i=2:length(f)
    ev_now = find_evap(Tb,68,5.67e-8,depth-z_f(i),kt,lv)/width;
    f_now = f_now+ev_now*(z_f(i)-z_f(i-1));
    f(i) = f_now;
    dfdz(i) = ev_now;
    if (phi+f_now<0)&&(flag==0)
        phi_to_zero = z_f(i);
        flag = 1;
    end
end

% Integrate for rho
rho = zeros(length(z_f),1);
M = zeros(length(z_f),1);
rho(1) = rho0;
rho_to_zero = depth;
M(1) = phi/rho0/sqrt(1.33*rg*Tb);
for i=2:length(f)
    % runge kutta method used here
    dn = rho(i-1);
    v = (phi+f(i))/dn;
    M(i) = v/sqrt(1.33*rg*Tb);
    k1 = (2*Cd*dn*v^2/width+dn*g)/(v^2-rg*Tb);
    if i<length(f)
        dn = rho(i-1)+(z_f(i)-z_f(i-1))*0.5*k1;
        v = (phi+(f(i)+f(i+1))/2)/dn;
        k2 = (2*Cd*dn*v^2/width+dn*g)/(v^2-rg*Tb);

        dn = rho(i-1)+(z_f(i)-z_f(i-1))*0.5*k2;
        v = (phi+(f(i)+f(i+1))/2)/dn;
        k3 = (2*Cd*dn*v^2/width+dn*g)/(v^2-rg*Tb);

        dn = rho(i-1)+(z_f(i)-z_f(i-1))*k3;
        v = (phi+f(i+1))/dn;
        k4 = (2*Cd*dn*v^2/width+dn*g)/(v^2-rg*Tb);
    else
        k2 = k1;
        k3 = k2;
        k4 = k3;
    end
    rho(i) = rho(i-1)+(z_f(i)-z_f(i-1))*1/6*(k1+k4+2*k2+2*k3);
    
    if (rho(i)<0)||(rho(i)-rho(i-1)>0) 
        % rho increases? this happens when rho is so small that numerical
        % problems are caused
        rho_to_zero = z_f(i);
        break;
    end
end
if((rho(end)>0)&&(phi+f(end)>0))
    mach_top = M(end);
    phi_top = phi+f(end);
    rho_top = rho(end);
else
    mach_top = 0;
    phi_top = 0;
    rho_top = 0;
end

function [pw] = vapor_press(Tw)
pw = 3.63e12*exp(-6147/Tw); % Vapor pressure at wall temperature
end

function [E] = find_evap(Tw,Te,sig,d,k,L)
c1 = 2*sig;
c2 = 4*k/pi/d;
% Use dichotomy here, more stable
T_l = 1;
T_r = Tw;
while(T_r-T_l>1e-8)
    v_l = c1*(T_l^4-Te^4)+c2*(T_l-Tw);
    T_m = (T_l+T_r)/2;
    v_m = c1*(T_m^4-Te^4)+c2*(T_m-Tw);
    if(v_l*v_m<0)
        T_r = T_m;
    else
        T_l = T_m;
    end
end
Ts = (T_l+T_r)/2;
E = -sig/L*(Ts^4-Te^4);
end

end

