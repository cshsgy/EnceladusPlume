function [w_rec,h_rec,t_rec] = liquid_dynamics_func(w_in,t_in,L)
% Note: everything to be ensured in SI BASIC UNITS.
v = 0.00;
h = 0.00;
g = 0.11; % g=0.11 for enceladus

% Assumed here: w_in and t_in are both n*1, column vectors
w_in = [w_in; w_in; w_in];
t_in = [t_in-118800; t_in; t_in+118800];
dwdt_in = gradient(w_in,t_in);
dwdt2_in = gradient(dwdt_in,t_in);

v_rec = [];
vs_rec = [];
h_rec = [];
t_rec = [];
w_rec = [];
dt = 0.1;
P = 118800;
t_stop = P*4; % nper = 4, t_stop = 1110.4 % For un-normalized results, period=118370, use 500000
iter = 0;
t = 0;
last_dvdt = 0;
last_dhdt = 0;
a = tic;
while t<t_stop
    iter = iter+1;
    if mod(iter,100)==0
	disp("Current dt(s):")
	disp(dt)
        disp("Estimated remaining time (s):");
        b = toc(a);
        est_time = b/t*t_stop-b;
        disp(est_time);
    end
    t_rec(end+1) = t;
    v_rec(end+1) = v;
    h_rec(end+1) = h;
    w = interp1(t_in,w_in,mod(t,P));
    w_rec(end+1) = w;
    dwdt = interp1(t_in,dwdt_in,mod(t,P));
    [zs,v_now] = vel_now(v, h, L, w, dwdt);
    vs_rec(end+1,:) = v_now;
    % Runge Kutta: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method
    t_now = t;
    w = interp1(t_in,w_in,mod(t_now,P));
    dwdt = interp1(t_in,dwdt_in,mod(t_now,P));
    dwdt2 = interp1(t_in,dwdt2_in,mod(t_now,P));
    [k1,s1] = derivative(v,h,L,g,w,dwdt,dwdt2);

    t_now = t+dt*2/9;
    w = interp1(t_in,w_in,mod(t_now,P));
    dwdt = interp1(t_in,dwdt_in,mod(t_now,P));
    dwdt2 = interp1(t_in,dwdt2_in,mod(t_now,P));
    [k2,s2] = derivative(v+2/9*k1,h+2/9*s1,L,g,w,dwdt,dwdt2);

    t_now = t+dt*1/3;
    w = interp1(t_in,w_in,mod(t_now,P));
    dwdt = interp1(t_in,dwdt_in,mod(t_now,P));
    dwdt2 = interp1(t_in,dwdt2_in,mod(t_now,P));
    [k3,s3] = derivative(v+1/12*k1+1/4*k2,h+1/12*s1+1/4*s2,L,g,w,dwdt,dwdt2);

    t_now = t+dt*3/4;
    w = interp1(t_in,w_in,mod(t_now,P));
    dwdt = interp1(t_in,dwdt_in,mod(t_now,P));
    dwdt2 = interp1(t_in,dwdt2_in,mod(t_now,P));    
    [k4,s4] = derivative(v+69/128*k1-243/128*k2+135/64*k3,h+69/128*s1-243/128*s2+135/64*s3,L,g,w,dwdt,dwdt2);
    
    t_now = t+dt;
    w = interp1(t_in,w_in,mod(t_now,P));
    dwdt = interp1(t_in,dwdt_in,mod(t_now,P));
    dwdt2 = interp1(t_in,dwdt2_in,mod(t_now,P));
    [k5,s5] = derivative(v-17/12*k1+27/4*k2-27/5*k3+16/15*k4,h-17/12*s1+27/4*s2-27/5*s3+16/15*s4,L,g,w,dwdt,dwdt2);
    
    t_now = t+dt*5/6;
    w = interp1(t_in,w_in,mod(t_now,P));
    dwdt = interp1(t_in,dwdt_in,mod(t_now,P));
    dwdt2 = interp1(t_in,dwdt2_in,mod(t_now,P));
    [k6,s6] = derivative(v+65/432*k1-5/16*k2+13/16*k3+4/27*k4+5/144*k5,h+65/432*s1-5/16*s2+13/16*s3+4/27*s4+5/144*s5,L,g,w,dwdt,dwdt2);
    
    t = t + dt;
    v = v + (47/450*k1+12/25*k3+32/225*k4+1/30*k5+6/25*k6)*dt;
    h = h + (47/450*s1+12/25*s3+32/225*s4+1/30*s5+6/25*s6)*dt;
    tek = -1/150*k1+3/100*k3-16/75*k4-1/20*k5+6/25*k6;
    tes = -1/150*s1+3/100*s3-16/75*s4-1/20*s5+6/25*s6;
    % Originally ^0.2, but still unstable. Here use ^1 for fast
    % response?
    dtk = 0.9*dt*(1e-8/abs(tek)); % Error tolerance for velocity is smaller
    dts = 0.9*dt*(1e-6/abs(tek));
    dt = min(50,max(0.005,min(dtk,dts))); % Max 50, min 0.005. This may require 10+ hours to run for very abrupt changes
    if isnan(v)
        break; % Opt out when the solution blows up...
    end
end
end

