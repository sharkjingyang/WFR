function dE_dmu_n = dE_dmu_n(mu, n,Nx,dx,dt,alpha)
    c=(Operator_L_u_alpha(mu(n,:),Nx,dx,alpha)\(mu(n+1,:) - mu(n,:))')';
    A = -2 * c;
    B = 2 * (Operator_L_u_alpha(mu(n-1,:),Nx,dx,alpha)\(mu(n,:) - mu(n-1,:))')';
    C = zeros(1,Nx);
    for i = 1:Nx
        C(i) = -L_ei(c, i,Nx,dx,alpha);
    
    end
    
    dE_dmu_n = (A + B + C) * dx / dt;
end
