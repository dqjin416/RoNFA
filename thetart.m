%阈值函数theta_rt
function y=thetart(x)
x(x<0)=0;
tmp=1-exp(-x);
y=tmp;