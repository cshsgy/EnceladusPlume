function composed_main_func(wmin,wmaxmin,depth)
    % Note: everything to be ensured in SI BASIC UNITS.
    % Runge-kutta is tricky due to the iterative step
    % v = (3-sqrt(2))/sqrt(2);
    % h = 1;
    v = 0.00;
    h = 0.00;
    L = depth;
    g = 0.11; % g=0.11 for enceladus
    v_rec = [];
    vs_rec = [];
    h_rec = [];
    t_rec = [];
    w_rec = [];
    dt = 0.05; % Time step is limited
    t_stop = 12500000; % nper = 100, % For un-normalized results, period=118370, use 500000
    iter = 0;
    t = 0;
    last_dvdt = 0;
    last_dhdt = 0;
    a = tic;
    flag = 0;
    while t<t_stop
        iter = iter+1;
        if mod(iter,100)==0
            disp("Progress (pct):");
            disp(t/t_stop*100);
            disp("Estimated remaining time (s):");
            b = toc(a);
            est_time = b/t*t_stop;
            disp(est_time);
        end
        t_rec(end+1) = t;
        v_rec(end+1) = v;
        h_rec(end+1) = h;
        w_rec(end+1) = width_new(0,t,wmaxmin,wmin);
        [zs,v_now] = vel_now(v, h, L, t, dt,wmaxmin,wmin);
        vs_rec(end+1,:) = v_now;
        % Runge Kutta: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method
        [k1,s1] = derivative(v,h,L,t,g,wmaxmin,wmin);
        [k2,s2] = derivative(v+2/9*k1,h+2/9*s1,L,t+dt*2/9,g,wmaxmin,wmin);
        [k3,s3] = derivative(v+1/12*k1+1/4*k2,h+1/12*s1+1/4*s2,L,t+dt*1/3,g,wmaxmin,wmin);
        [k4,s4] = derivative(v+69/128*k1-243/128*k2+135/64*k3,h+69/128*s1-243/128*s2+135/64*s3,L,t+dt*3/4,g,wmaxmin,wmin);
        [k5,s5] = derivative(v-17/12*k1+27/4*k2-27/5*k3+16/15*k4,h-17/12*s1+27/4*s2-27/5*s3+16/15*s4,L,t+dt,g,wmaxmin,wmin);
        [k6,s6] = derivative(v+65/432*k1-5/16*k2+13/16*k3+4/27*k4+5/144*k5,h+65/432*s1-5/16*s2+13/16*s3+4/27*s4+5/144*s5,L,t+dt*5/6,g,wmaxmin,wmin);
        
        t = t + dt;
        v = v + (47/450*k1+12/25*k3+32/225*k4+1/30*k5+6/25*k6)*dt;
        h = h + (47/450*s1+12/25*s3+32/225*s4+1/30*s5+6/25*s6)*dt;
        tek = -1/150*k1+3/100*k3-16/75*k4-1/20*k5+6/25*k6;
        tes = -1/150*s1+3/100*s3-16/75*s4-1/20*s5+6/25*s6;
        % Originally ^0.2, but still unstable. Here use ^1 for fast
        % response?
        dtk = 0.9*dt*(1e-8/abs(tek)); % Error tolerance for velocity is smaller
        dts = 0.9*dt*(1e-6/abs(tek));
        dt = min(10,max(0.01,min(dtk,dts))); % Max 10, min 0.01
        if isnan(v)
            break; % Opt out when the solution blows up...
        end
        % Check the last half a cycle, if more than 4 zero crossings, then skip
        if mod(iter,5000)==0
            ct = 0;
            for ss=1:1000 % Check at most half a day...
                if h_rec(end-ss)*h_rec(end-ss-1)<=0
                    ct = ct+1;
                end
            end
            if ct>=4
                flag = 1; break;
            end
        end
        if h>L/10 % Overflow, break to save time and space
            flag = 1; break;
        end
    end

    % Heat conduction discretization
    Dx = 0.05;
    % Heat conduction solve up to. Note: characteristic length~0.34m
    Max_x = 0.4;
    % Number of Vertical Levels 
    Nz = 100;
    % Effective surface temperature
    Te = 68;
    % Periods to run. sqrt(Kappa*T)~3m here, good enough to reach eq
    Periods_Run = 60;
    
    % physical parameters
    Kappa = 1e-6; % Diffusivity
    Period = 118800;
    Lv = 2.84E6; % Sublimation latent heat
    K = 3; % Conductivity
    G = 0.113; % Gravity
    Cd = 0.002; % Drag Coefficient
    
    % Derived Parameters
    [t_rec,h_rec,w_rec] = SimplifyCrackDynamicsData(t_rec,Period,h_rec,w_rec);
    D = L/10; % Crack depth taken as 10% of the full length
    if max(h_rec)>D
        disp(strcat('Encounters water to the surface, PROGRAM TERMINATED WITHOUT SAVING.'));
        return;
    end
    Dt = 0.1/Kappa*Dx^2;
    
    %% Variables initialization
    % Solve up to 1 grid from the surface to protect the heat conduction val
    z_wet = linspace(min(h_rec),D,Nz+1);
    z_wet = z_wet(1:end-1);
    dz = (max(z_wet)-min(z_wet))/(Nz-1);
    
    % Solve at most 1.5m into the wall
    xs = 0.0:Dx:Max_x;
    nx = length(xs);
    % Initialize to close to water temp
    T = ones(Nz,nx)*273.15;
    % Not wetted walls initialized to be colder
    T(z_wet>max(h_rec),:) = T(z_wet>max(h_rec),:)-5;
    for i=1:Nz
        Ts = SurfaceTemperature(Te,T(i,1),D-z_wet(i)+max(xs),K);
        if Ts<0
            error('NUMERICAL ERROR: No Solution to the surface temperature.');
        end
        T(i,:) = T(i,:)-xs*(T(i,1)-Ts)*2/pi/(D-z_wet(i)+max(xs));
    end
    
    % Recording variables
    phi_rec = [];
    r_rec = [0.6]; % just some random number put up here
    time_rec = [];
    
    Ev_rec = {};
    Tw_rec = {};
    
    % For debugging purposes
    T_rec = {};
    width_rec = [];
    zs_rec = {};
    
    %% Run Main Solver 
    t = 0; iter = 1; tic;
    while(t<Periods_Run*Period)
        time_rec(end+1) = t;
        if mod(iter,50)==0
            % Printing diagnostics every several steps
            disp('Number of iteration:');
            disp(iter);
            tmp = toc;
            fprintf('Estimated time left (h): %.2f\n',t/(Periods_Run*Period));
        end
        % Interpolate to the above-liquid grid
        height = interp1(t_rec,h_rec,mod(t,Period));
        depth = D-height;
        width = interp1(t_rec,w_rec,mod(t,Period));
        zs = linspace(0,depth,Nz);
        T2_o = T(:,2);
        T2 = interp1(z_wet,T2_o,zs+height,'linear','extrap');
        
        % Calculate the gas
        [phi_rec(end+1),Tw,Ev,r_rec(end+1)] = GasDynamicsMarchInTimeConstrained(width,zs,T2,K,Lv,G,Cd,Dx,r_rec(end),max(h_rec));
        T_rec{end+1} = T;
        width_rec(end+1) = width;
        zs_rec{end+1} = zs;
        
        % Interpolate back
        Tw = interp1(zs+height,Tw,z_wet,'linear','extrap');
        Ev = interp1(zs+height,Ev,z_wet,'linear','extrap');
        Tw(z_wet<height) = 273.15;
        Ev(z_wet<height) = 0;
        
        % Take record
        Ev_rec{end+1} = Ev;
        Tw_rec{end+1} = Tw;
        
        % Calculate heat diffusion, parallelized
        for ii=1:Nz
            T(ii,:) = HeatDiffusionMarchInTime(z_wet(ii),xs,height,depth,...
                T(ii,:),Tw(ii),Kappa,K,Te,Dx,Dt);
        end
        
        t = t+Dt; iter = iter+1;
    end
    save('run_results.mat','Period','time_rec','phi_rec','width_rec','r_rec','t_rec','h_rec');
    
end
    
    
