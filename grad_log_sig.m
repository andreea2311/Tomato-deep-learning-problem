function [grad_log_sig] = grad_log_sig(z)
 % The gradient of log(sigmoid(z)) is sigmoid(-z)
 % This is just the sigmoid function
 grad_log_sig = 1 ./ (1 + exp(-z));
end