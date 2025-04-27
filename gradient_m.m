function [weight1, bias1, weight2, bias2, history] = gradient_m(X, y, input_dim, hidden_dim, output_dim, alpha, max_iter)
% Gradient descent training function with log_sigmoid activation
 weight1 = randn(input_dim, hidden_dim) * 0.01; % Reduced initial weights
 bias1 = zeros(1, hidden_dim);
 weight2 = randn(hidden_dim, output_dim) * 0.01; % Reduced initial weights
 bias2 = zeros(1, output_dim);
 eps_val = 1e-10;
 history.loss = [];
 history.grad_norm = [];
% Add a small regularization
 lambda = 0.0001;
for iter = 1:max_iter
% Forward pass with numerical stability
 z1 = X * weight1 + repmat(bias1, size(X, 1), 1);
 log_h1 = log_sig(z1); % Using log_sigmoid activation
 h1 = exp(log_h1); % Convert back for next layer calculation
 z2 = h1 * weight2 + repmat(bias2, size(h1, 1), 1);
 log_h2 = log_sig(z2); % Using log_sigmoid activation
 h2 = exp(log_h2); % Convert back for loss calculation
% Clip values to avoid numerical issues
 h2 = min(max(h2, eps_val), 1-eps_val);
% Compute loss with regularization
 reg_term = (lambda/2) * (sum(sum(weight1.^2)) + sum(sum(weight2.^2)));
 loss = -mean(sum(y .* log(h2) + (1 - y) .* log(1 - h2), 2)) + reg_term;
% Check for NaN and break if found
if isnan(loss)
 fprintf('NaN detected at iteration %d. Stopping training.\n', iter);
break;
end
 history.loss = [history.loss; loss];
% Compute gradients with regularization
 d_h2 = (h2 - y) / size(X, 1);
% Gradient with respect to z2
 d_z2 = d_h2 .* h2 .* (1 - h2);  % Fixed gradient calculation
% Gradients for layer 2
 grad_weight2 = h1' * d_z2 + lambda * weight2;
 grad_bias2 = sum(d_z2, 1);
% Gradient for hidden layer
 d_h1 = (d_z2 * weight2');
% Gradient with respect to z1
 d_z1 = d_h1 .* h1 .* (1 - h1);  % Fixed gradient calculation
% Gradients for layer 1
 grad_weight1 = X' * d_z1 + lambda * weight1;
 grad_bias1 = sum(d_z1, 1);
% Flatten total gradient norm
 total_grad = [grad_weight1(:); grad_bias1(:); grad_weight2(:); grad_bias2(:)];
 grad_norm = norm(total_grad);
 history.grad_norm = [history.grad_norm; grad_norm];
% Use a learning rate decay - modified for more stability
 curr_alpha = alpha / (1 + 0.005 * iter);  % gentler decay
% Gradient descent update with gradient clipping
 max_grad_norm = 5.0;
if grad_norm > max_grad_norm
 scaling_factor = max_grad_norm / grad_norm;
 grad_weight1 = grad_weight1 * scaling_factor;
 grad_bias1 = grad_bias1 * scaling_factor;
 grad_weight2 = grad_weight2 * scaling_factor;
 grad_bias2 = grad_bias2 * scaling_factor;
end
 weight1 = weight1 - curr_alpha * grad_weight1;
 bias1 = bias1 - curr_alpha * grad_bias1;
 weight2 = weight2 - curr_alpha * grad_weight2;
 bias2 = bias2 - curr_alpha * grad_bias2;
% Print progress
if mod(iter, 10) == 0
 fprintf('Iteration %d, Loss: %.6f, Gradient Norm: %.6f\n', iter, loss, grad_norm);
end
% Stop if gradient is small
if grad_norm < eps_val
break;
end
end
end