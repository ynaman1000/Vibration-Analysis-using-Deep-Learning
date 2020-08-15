function o = omega(b, l1, r1o, r1i, E, rho)
% b-  [no_pointsx1], array of betas
% l1- float, L1
% r1o-float, outer radius at base
% r1i-float, inner radius at base
% E-  float, Young's modulus
% rho-float, density
% returns natural frequency corresponding to b
o = ((b / l1).^2) * (((E * I1(r1o,r1i)) / (rho * A1(r1o,r1i)))^0.5);
end