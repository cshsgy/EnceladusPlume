function [Tw,Ev] = MikiFindEvap(T2,Tg,Pb,Dx,K,Lv)
% find evaporation from T2, Pb(Back Pres), Tg(Tgas), and Dx.
% Initial guesses
Tw1 = 273.15;
Tw2 = T2;
v1 = FunctionToSolve(Evaporation(Pb,VaporPressure(Tw1),Tg,Tw1),T2,Tw1,Dx,K,Lv);
v2 = FunctionToSolve(Evaporation(Pb,VaporPressure(Tw2),Tg,Tw2),T2,Tw2,Dx,K,Lv);
while abs(Tw1-Tw2)>1e-3
    Tw3 = Tw1-v1*(Tw1-Tw2)/(v1-v2);
    Tw1 = Tw2;
    v1 = v2;
    Tw2 = Tw3;
    v2 = FunctionToSolve(Evaporation(Pb,VaporPressure(Tw2),Tg,Tw2),T2,Tw2,Dx,K,Lv);
end
Tw = Tw2;
Ev = Evaporation(Pb,VaporPressure(Tw),Tg,Tw);
end

