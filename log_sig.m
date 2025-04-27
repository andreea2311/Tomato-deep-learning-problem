function [lsig] = log_sig(z)
 % Numerically stable implementation
 lsig = zeros(size(z));
 pos_mask = z >= 0;
 neg_mask = ~pos_mask;
 
 % For positive values use log(sigmoid(z)) = log(1) - log(1 + exp(-z))
 lsig(pos_mask) = -log(1 + exp(-z(pos_mask)));
 
 % For negative values use log(sigmoid(z)) = z - log(1 + exp(z))
 lsig(neg_mask) = z(neg_mask) - log(1 + exp(z(neg_mask)));
end