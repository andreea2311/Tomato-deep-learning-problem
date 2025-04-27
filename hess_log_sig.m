function [hess_log_sig] = hess_log_sig(z)
 % The second derivative of log(sigmoid(z))
 sig_z = 1 ./ (1 + exp(-z));
 hess_log_sig = sig_z .* (1 - sig_z);
end