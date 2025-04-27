function [hiss] = hessiana(x, w)
    m = size(x, 2);
    z = zeros(m, 1);
    
    for i = 1:m
        e = log_sig(w' * x(:, i));  
        z(i) = e * (1 - e);  % Derivative of sigmoid
    end
    
    Q = diag(z);
    hiss = (x * Q * x') / m;
end