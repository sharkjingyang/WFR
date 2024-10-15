Nx = 40;
Nt = 30;
alpha = 1;
tao = 0.001;
X = [-5, 5];
T = [0, 1];
x_space = linspace(X(1), X(2), Nx);
t_space = linspace(T(1), T(2), Nt + 1);
mu = rand(Nt + 1, Nx)/Nx;
mu(1, :) = rho_0(x_space);
mu(end, :) = rho_1(x_space);
dx = (X(2)-X(1)) / (Nx - 1);
dt = 1 / Nt;

[X, Y] = meshgrid(x_space, t_space);
lam = 0;
k_max = 200000;


for k = 1:k_max
    disp(k);
    lam = 0.5 * (1 + sqrt(1 + 4 * lam^2));
    lam_nex = 0.5 * (1 + sqrt(1 + 4 * lam^2));
    gamma = (1 - lam) / lam_nex;
    mu_half = mu;
    
    for t = 2:Nt
        mu_half(t,:) = mu(t,:) - tao * dE_dmu_n(mu,t,Nx,dx,dt,alpha);
    end
    
    mu_half(mu_half < 0) = 0;
    
    mu = (1 - gamma) * mu_half + gamma * mu;
    
    if mod(k, 1000) == 0
       surf(X, Y, mu);
       xlabel('x');
       ylabel('t');
       zlabel('mu');
       title('3D Plot');
       drawnow;
       pause(0.2);
       clf;
    end
end



function g = Gaussian(x, mu, sigma)
    g = 1 / sqrt(2 * pi * sigma^2) * exp(-(x - mu).^2 / (2 * sigma^2));
end

function r = rho_0(x)
    r = 1/3*Gaussian(x, -2, sqrt(1))+2/3*Gaussian(x, 2, sqrt(1));
end

function r = rho_1(x)
    r = Gaussian(x, 0, sqrt(1)) ;
end






