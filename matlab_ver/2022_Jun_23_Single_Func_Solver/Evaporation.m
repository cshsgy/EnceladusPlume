function [E] = Evaporation(Pg,Pw,Tg,Tw)
    % Not multiplied by 2 since we are working on one side only for
    % conduction.
    rg = 8.341/0.018;
    E = -Pg/sqrt(2*pi*rg*Tg)+Pw/sqrt(2*pi*rg*Tw);
end