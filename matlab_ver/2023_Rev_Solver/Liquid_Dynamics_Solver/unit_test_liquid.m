t_in = 100:100:118800;
t_in = t_in';
L = 20000;
w_in = 0.5+0.2*sin(t_in/118800*2*pi);

[w_rec,h_rec,t_rec] = main_func(w_in,t_in,L);

