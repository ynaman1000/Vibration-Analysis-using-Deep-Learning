function w = W(e, n, b, c)
% e- float, between [L0/L1, 1], eta=x/L1
% n- integer, exponent in area and moment of inertia
% b- [no_of_modesx1] array, betas
% c- [no_of_modesx4] array; [c1,c2,c3,c4]
% returns mode shape; displacement at each position
z_arr = z(b, e);
w = sum([J(n, z_arr) Y(n, z_arr) I(n, z_arr) K(n, z_arr)].*c, 2).*((e).^(-n/2));
end