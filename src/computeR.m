function [r] = computeR(A,G,x1,x2)
    r = x1'*A*x1 + x2'*A*x2 - 2*x1'*G*x2; % eq 4
end