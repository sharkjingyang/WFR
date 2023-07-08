
function L = Operator_L_u_alpha(mu,Nx,dx,alpha)
    L = zeros(Nx, Nx);
    for i = 2:Nx-1
        L(i, i) = 0.5 / dx^2 * (mu(i-1) + mu(i)) + 0.5 / dx^2 * (mu(i) + mu(i+1));
    end
    L(1, 1) = 0.5 / dx^2 * (mu(1) + mu(2));
    L(Nx, Nx) = 0.5 / dx^2 * (mu(Nx-1) + mu(Nx));
    for i = 1:Nx-1
        L(i, i+1) = -0.5 / dx^2 * (mu(i) + mu(i+1));
        L(i+1, i) = -0.5 / dx^2 * (mu(i) + mu(i+1));
    end
    L = L + eye(Nx) * alpha;
end
