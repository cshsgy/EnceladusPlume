function [val,t_day] = accum_last_days(P,t_rec,val_rec,n_days)
    t_rec = t_rec/P;
    t_day = linspace(0,1,1000);
    while max(t_rec)>1
        t_rec = t_rec-1;
    end
    t_rec = t_rec+1;
    val = interp1(t_rec,val_rec,t_day);
    for i=2:n_days
        t_rec = t_rec+1;
        val = val+interp1(t_rec,val_rec,t_day);
    end
    val = val/n_days;
end