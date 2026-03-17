function [v] = FunctionToSolve(Ev,T2,Tw,Dx,K,Lv)
    % This is the function that we aim to have = 0
    v = K*(T2-Tw)/Dx-Lv*Ev;
end