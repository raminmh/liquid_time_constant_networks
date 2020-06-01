function ode_test
%ODE_TEST  Run non-adaptive ODE solvers of different orders.
%   ODE_TEST compares the orders of accuracy of several explicit Runge-Kutta
%   methods. The non-adaptive ODE solvers are tested on a problem used in
%   E. Hairer, S.P. Norsett, and G. Wanner, Solving Ordinary Differential
%   Equations I, Nonstiff Problems, Springer-Verlag, 1987. 
%   The errors of numerical solutions obtained with several step sizes is
%   plotted against the step size used. For a given solver, the slope of that 
%   line corresponds to the order of the integration method used.  
%

solvers = {'ode1','ode2','ode3','ode4','ode5'};
Nsteps = [800 400 200 100 75 50 20 11];

x0 = 0;
y0 = [1;1;1;1];
xend = 1;
h = (xend-x0)./Nsteps;

err = zeros(length(solvers),length(h));

for i = 1:length(solvers)  
  solver = solvers{i};
  for j = 1:length(Nsteps)
    N = Nsteps(j);
    x = linspace(x0,xend,N+1);
    y = feval(solver,@f,x,y0);
    err(i,j) = max(max(abs(y - yexact(x))));
  end
end  

figure
loglog(h,err);
title('Different slopes for methods of different orders...');
xlabel('log(h)');
ylabel('log(err)');
legend(solvers{:},4);

% -------------------------
function yp = f(x,y)
yp = [  2*x*y(4)*y(1)
       10*x*y(4)*y(1)^5
        2*x*y(4)
       -2*x*(y(3)-1) ];

% -------------------------
function y = yexact(x)
x2 = x(:).^2;
y = zeros(length(x),4);
y(:,1) = exp(sin(x2));
y(:,2) = exp(5*sin(x2));
y(:,3) = sin(x2)+1;
y(:,4) = cos(x2);

