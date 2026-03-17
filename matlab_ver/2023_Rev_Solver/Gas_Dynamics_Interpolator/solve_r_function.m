function [r,phi_top,rho_top,phi0] = solve_r_function(Tb,depth,width)
% For stability here we use dichotomy
r_l = 1e-5;
r_r = 1-1e-5;
iter = 0;
phi_l = 0;
phi_r = 0;
rho_l = 0;
rho_r = 0;
phi0_l = 0;
phi0_r = 0;

while(abs(r_l-r_r)>1e-4)
    iter = iter+1;
    display(iter);
    r_m = (r_l+r_r)/2;
    [phi_to_zero,rho_to_zero,mach_top,phi_top,rho_top,phi0] = solve_function(Tb,depth,width,r_m);
    if(mach_top==0) % has not reached top
        if(rho_to_zero)<(phi_to_zero)
            % rho becomes zero first.
            % rho_0 increases with r, but phi_0 decreases with r
            % So need to increase r
            r_l = r_m;
            phi_l = phi_top;
	    rho_l = rho_top;
	    phi0_l = phi0;
        else
            r_r = r_m;
            phi_r = phi_top;
	    rho_r = rho_top;
	    phi0_r = phi0;
        end
    else
        if(mach_top>1.6)
            % increase r, increase rho, decrease v
            r_l = r_m;
            phi_l = phi_top;
	    rho_l = rho_top;
	    phi0_l = phi0;
        else
            % decrease r, decrease rho, increase v
            r_r = r_m;
            phi_r = phi_top;
	    rho_r = rho_top;
	    phi0_r = phi0;
        end
    end
end

r = (r_l+r_r)/2;
phi_top = (phi_l+phi_r)/2;
rho_top = (rho_l+rho_r)/2;
phi0 = (phi0_l+phi0_r)/2;
end

