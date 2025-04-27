function [weight1, bias1, weight2, bias2, history] = adam_m(X, y, input_dim, hidden_dim, output_dim, alpha, max_iter)
% Adam optimization training function with log_sigmoid activation
    weight1 = randn(input_dim, hidden_dim) * 0.01; % Reduced initial weights
    bias1 = zeros(1, hidden_dim);
    weight2 = randn(hidden_dim, output_dim) * 0.01; % Reduced initial weights
    bias2 = zeros(1, output_dim);
    eps_val = 1e-10;
    history.loss = [];
    history.grad_norm = [];

    % Adam hyperparameters
    beta1 = 0.9;    % Exponential decay rate for first moment estimates
    beta2 = 0.999;  % Exponential decay rate for second moment estimates
    epsilon = 1e-8; % Small constant for numerical stability
    
    % Try a lower learning rate
    alpha = alpha * 0.1; % Reduce the learning rate to improve stability
    
    % Initialize moment estimates
    m_weight1 = zeros(size(weight1));
    v_weight1 = zeros(size(weight1));
    m_bias1 = zeros(size(bias1));
    v_bias1 = zeros(size(bias1));
    
    m_weight2 = zeros(size(weight2));
    v_weight2 = zeros(size(weight2));
    m_bias2 = zeros(size(bias2));
    v_bias2 = zeros(size(bias2));
    
    % Add a small regularization
    lambda = 0.0001;
    
    for iter = 1:max_iter
        % Forward pass with numerical stability
        z1 = X * weight1 + repmat(bias1, size(X, 1), 1);
        % Use standard sigmoid instead of log_sigmoid for more stable forward pass
        h1 = 1 ./ (1 + exp(-z1));
        
        % Clip activation values
        h1 = min(max(h1, eps_val), 1-eps_val);
        
        z2 = h1 * weight2 + repmat(bias2, size(h1, 1), 1);
        % Use standard softmax for output layer
        exp_scores = exp(z2 - max(z2, [], 2));  % Subtract max for numerical stability
        h2 = exp_scores ./ sum(exp_scores, 2);
        
        % Clip values to avoid numerical issues
        h2 = min(max(h2, eps_val), 1-eps_val);
        
        % Cross-entropy loss with regularization
        reg_term = (lambda/2) * (sum(sum(weight1.^2)) + sum(sum(weight2.^2)));
        loss = -mean(sum(y .* log(h2), 2)) + reg_term;
        
        % Check for NaN and break if found
        if isnan(loss)
            fprintf('NaN detected at iteration %d. Stopping training.\n', iter);
            break;
        end
        history.loss = [history.loss; loss];
        
        % Gradient clipping threshold
        max_grad_norm = 5.0;
        
        % Compute gradients with regularization
        d_h2 = (h2 - y) / size(X, 1);
        
        % Gradients for layer 2
        grad_weight2 = h1' * d_h2 + lambda * weight2;
        grad_bias2 = sum(d_h2, 1);
        
        % Gradient for hidden layer
        d_h1 = (d_h2 * weight2');
        
        % Gradient with respect to sigmoid activation
        d_z1 = d_h1 .* h1 .* (1 - h1);
        
        % Gradients for layer 1
        grad_weight1 = X' * d_z1 + lambda * weight1;
        grad_bias1 = sum(d_z1, 1);
        
        % Flatten total gradient norm
        total_grad = [grad_weight1(:); grad_bias1(:); grad_weight2(:); grad_bias2(:)];
        grad_norm = norm(total_grad);
        
        % Apply gradient clipping if gradient norm is too large
        if grad_norm > max_grad_norm
            scaling_factor = max_grad_norm / grad_norm;
            grad_weight1 = grad_weight1 * scaling_factor;
            grad_bias1 = grad_bias1 * scaling_factor;
            grad_weight2 = grad_weight2 * scaling_factor;
            grad_bias2 = grad_bias2 * scaling_factor;
            grad_norm = max_grad_norm;
        end
        
        history.grad_norm = [history.grad_norm; grad_norm];
        
        % Adam update rule
        t = iter;
        
        % Update biased first moment estimate (momentum)
        m_weight1 = beta1 * m_weight1 + (1 - beta1) * grad_weight1;
        m_bias1 = beta1 * m_bias1 + (1 - beta1) * grad_bias1;
        m_weight2 = beta1 * m_weight2 + (1 - beta1) * grad_weight2;
        m_bias2 = beta1 * m_bias2 + (1 - beta1) * grad_bias2;
        
        % Update biased second raw moment estimate (RMSprop)
        v_weight1 = beta2 * v_weight1 + (1 - beta2) * (grad_weight1.^2);
        v_bias1 = beta2 * v_bias1 + (1 - beta2) * (grad_bias1.^2);
        v_weight2 = beta2 * v_weight2 + (1 - beta2) * (grad_weight2.^2);
        v_bias2 = beta2 * v_bias2 + (1 - beta2) * (grad_bias2.^2);
        
        % Compute bias-corrected first moment estimate
        m_weight1_hat = m_weight1 / (1 - beta1^t);
        m_bias1_hat = m_bias1 / (1 - beta1^t);
        m_weight2_hat = m_weight2 / (1 - beta1^t);
        m_bias2_hat = m_bias2 / (1 - beta1^t);
        
        % Compute bias-corrected second raw moment estimate
        v_weight1_hat = v_weight1 / (1 - beta2^t);
        v_bias1_hat = v_bias1 / (1 - beta2^t);
        v_weight2_hat = v_weight2 / (1 - beta2^t);
        v_bias2_hat = v_bias2 / (1 - beta2^t);
        
        % Update parameters
        weight1 = weight1 - alpha * m_weight1_hat ./ (sqrt(v_weight1_hat) + epsilon);
        bias1 = bias1 - alpha * m_bias1_hat ./ (sqrt(v_bias1_hat) + epsilon);
        weight2 = weight2 - alpha * m_weight2_hat ./ (sqrt(v_weight2_hat) + epsilon);
        bias2 = bias2 - alpha * m_bias2_hat ./ (sqrt(v_bias2_hat) + epsilon);
        
        % Print progress
        if mod(iter, 50) == 0
            fprintf('Iteration %d, Loss: %.6f, Gradient Norm: %.6f\n', iter, loss, grad_norm);
        end
        
        % Stop if gradient is small
        if grad_norm < eps_val
            break;
        end
    end
end