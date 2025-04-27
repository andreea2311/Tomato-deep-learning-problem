function grad = gradient(X, y, W)
   [N, ~] = size(X);
   z = X * W;
   log_h = log_sig(z);
   h = exp(log_h); % Convert from log space
   error = h - y;
   grad = (X' * error) / N;
end