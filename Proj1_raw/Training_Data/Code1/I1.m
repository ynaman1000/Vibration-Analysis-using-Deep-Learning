function i = I1(ro, ri)
% ro- float, outer radius
% ri- float, inner radius
% returns moment of inertia about flexural axis (i.e. diameter)
i = (pi/4) * (ro^4-ri^4);
end