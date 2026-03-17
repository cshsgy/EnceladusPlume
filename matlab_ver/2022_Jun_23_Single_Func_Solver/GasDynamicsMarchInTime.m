function [PhiTop,Tw,Ev,r] = GasDynamicsMarchInTime(width,zs,T2,K,Lv,G,Cd,Dx)
% Search for correct r value
Max_Par = 12;
% Initial r borders
r_l = 0.001;
r_r = 0.999;
% Cannot use Newton-Raphson, use parallelized random search

while r_r-r_l>1e-3
    rs = linspace(r_l,r_r,Max_Par);
    Cmd = zeros(Max_Par,1);
    parfor i=1:length(rs)
        [~,~,M,Phi] = MikiModelFull(rs(i),width,zs,T2,K,Lv,G,Cd,Dx);
        % fprintf('r=%d,M=%.2f\n',i,M);
        % Determine increase or decrease
        if Phi<0
            % Decrease r to increase Phi
            Cmd(i) = -1;
        else
            if M<0
                % Rho too small. Increase r
                Cmd(i) = 1;
            else
                if M>1
                    % Increase r to slow down
                    Cmd(i) = 1; 
                else
                    % Decrease r to speed up
                    Cmd(i) = -1;
                end
            end
        end
    end
    % Find the first decrease cmd and last increase cmd
    r_l = rs(1);
    r_r = rs(end);
    for i=2:length(Cmd)
        if Cmd(i)<0
            r_l = rs(i-1);
            break;
        end
    end
    flag = 0;
    for i=length(Cmd):-1:1
        if (Cmd(i)>0)&&(flag==0)
            continue;
        end
        if Cmd(i)<0
            flag = 1;
        end
        if Cmd(i)>0
            r_r = rs(i+1);
            break;
        end
    end
    if flag==0
        warning('Parallelism too small, not capturing the correct region');
        Max_Par = Max_Par*2;
	if Max_Par>100
	    % We consider no hope of finding it elsewhere
	    r_l = rs(1)+0.9*(1-rs(1));
	    r_r = rs(2);
	else
            r_l = rs(1);
            r_r = rs(end);
	end
        continue;
    end
    if (r_l==rs(1))&&(r_r==rs(end))
        % No change, force shrink in size
        r_l = 0.9*r_l+0.1*r_r;
        r_r = 0.9*r_r+0.1*r_l;
    end
end

r = (r_r+r_l)/2;

[Tw,Ev,~,PhiTop] = MikiModelFull(r,width,zs,T2,K,Lv,G,Cd,Dx);
Tw(Tw==0) = T2(Tw==0); % Finish up and avoid error
end

