function [RHS,dhdt] = derivative(v,h,L,g,w,dwdt,dwdt2)
[zs,v_now] = vel_now(v, h, L, w, dwdt);
fric = friction(zs,v_now,w);
RHS = -0.5*(v_now(1)^2+v_now(end)^2)-g*h-fric;
add_term = additional_term(zs,w,dwdt,dwdt2);
RHS = RHS-add_term;
RHS = RHS/(h+L);
dhdt = v_now(end);
end

