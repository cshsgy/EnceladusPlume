function [PhiTop,Tw,Ev,r] = GasDynamicsMarchInTimeConstrained(width,zs,T2,K,Lv,G,Cd,Dx,r_last,hmax)
% Search for correct r value
Max_Change_Allowed = 0.005;
% Initial r borders
r_l = r_last-Max_Change_Allowed;
r_r = r_last+Max_Change_Allowed;

while r_r-r_l>1e-4
    rs = (r_r+r_l)/2;
    [Tw,Ev,M,PhiTop] = MikiModelFull(rs,width,zs,T2,K,Lv,G,Cd,Dx,hmax);
        if PhiTop<0
            % Decrease r to increase Phi
            r_r = rs;
        else
            if M<0
                % Rho too small. Increase r
                r_l = rs;
            else
                if M>1.6
                    % Increase r to slow down
                    r_l = rs; 
                else
                    % Decrease r to speed up
                    r_r = rs;
                end
            end
        end
end

r = rs;
Tw(Tw==0) = T2(Tw==0); % Finish up and avoid error (?)

end

