function a = A(b, n, e0, e1)
% b- float, beta
% n- integer, exponent in area and moment of inertia
% e0-float, eta0=L0/L1
% e1-float, eta1=1
% returns matrix A; Ax=0; constraint matrix
z0 = z(b, e0);
z1 = z(b, e1);
a = [J(n+1, z0)  Y(n+1, z0)  I(n+1, z0) -K(n+1, z0);...
     J(n+2, z0)  Y(n+2, z0)  I(n+2, z0)  K(n+2, z0);...
     J(n+0, z1)  Y(n+0, z1)  I(n+0, z1)  K(n+0, z1);...
     J(n+1, z1)  Y(n+1, z1) -I(n+1, z1)  K(n+1, z1)];
end