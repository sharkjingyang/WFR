function L_ei_value = L_ei(mu, i,Nx,dx,alpha)
    if i == 1
        L_ei_value = 0.5 * (mu(i+1)-mu(i))^2 / dx^2;
    elseif i == Nx
        L_ei_value = 0.5 * (mu(i)-mu(i-1))^2 / dx^2;
    elseif 2 <= i&& i<= Nx-1
        L_ei_value = 0.5 * (mu(i)-mu(i-1))^2/dx^2+0.5*(mu(i+1) - mu(i))^2/dx^2;
    end
end