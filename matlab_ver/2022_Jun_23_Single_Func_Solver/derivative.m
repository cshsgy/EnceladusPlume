function [RHS,dhdt] = derivative(v,h,L,t,g,wmaxmin,wmin)
dt = 1;
[zs,v_now] = vel_now(v, h, L, t, dt,wmaxmin,wmin);
fric = friction(zs,v_now,t,wmaxmin,wmin);
RHS = -0.5*(-v_now(1)^2+v_now(end)^2)-g*h-fric;
add_term = additional_term(zs,t, dt,wmaxmin,wmin);
RHS = RHS-add_term;
RHS = RHS/(h+L);
dhdt = v_now(end);
end

