function [evap_rate] = evaporation(ec,bv,tm)
  t0 = 273.15;
  evap_rate = ec/(tm^0.5)*exp(bv*(1/t0-1/tm));
end

