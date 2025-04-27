function [W1, b1, W2, b2, history] = newton(X, y, input_dim, hidden_dim, output_dim, max_iter)
    % Initialize with small values
    W1 = randn(input_dim, hidden_dim) * 0.01;
    b1 = zeros(1, hidden_dim);
    W2 = randn(hidden_dim, output_dim) * 0.01;
    b2 = zeros(1, output_dim);
    
    eps_val = 1e-10;
    history.loss = [];
    history.grad_norm = [];
    
    % Hyperparameters
    lambda = 0.01;      % Regularization
    lr_init = 0.05;     % Initial learning rate
    lr_decay = 0.99;    % Learning rate decay
    
    % Use a batch approach for Newton method to improve speed
    batch_size = min(1000, size(X, 1)); % Use smaller batches
    
    for iter = 1:max_iter
        % Adjust learning rate
        learning_rate = lr_init * (lr_decay ^ iter);
        
        % Randomly select batch
        batch_idx = randperm(size(X, 1), batch_size);
        X_batch = X(batch_idx, :);
        y_batch = y(batch_idx, :);
        
        % Forward pass
        z1 = X_batch * W1 + repmat(b1, batch_size, 1);
        h1 = 1.0 ./ (1.0 + exp(-z1));
        
        z2 = h1 * W2 + repmat(b2, batch_size, 1);
        h2 = 1.0 ./ (1.0 + exp(-z2));
        
        % Clip values to avoid numerical issues
        h2 = min(max(h2, eps_val), 1-eps_val);
        
        % Compute loss with regularization
        reg_term = (lambda/2) * (sum(sum(W1.^2)) + sum(sum(W2.^2)));
        loss = -mean(sum(y_batch .* log(h2) + (1 - y_batch) .* log(1 - h2), 2)) + reg_term;
        
        % Check for NaN
        if isnan(loss)
            fprintf('NaN detected at iteration %d. Stopping training.\n', iter);
            break;
        end
        
        history.loss = [history.loss; loss];
        
        % Compute gradients with regularization
        d_h2 = (h2 - y_batch) / batch_size;
        grad_W2 = h1' * d_h2 + lambda * W2;
        grad_b2 = sum(d_h2, 1);
        
        d_h1 = (d_h2 * W2') .* h1 .* (1 - h1);
        grad_W1 = X_batch' * d_h1 + lambda * W1;
        grad_b1 = sum(d_h1, 1);
        
        % Total gradient norm
        total_grad = [grad_W1(:); grad_b1(:); grad_W2(:); grad_b2(:)];
        grad_norm = norm(total_grad);
        history.grad_norm = [history.grad_norm; grad_norm];
        
        if mod(iter, 50) == 0
            fprintf('Iteration %d, Loss: %.6f, Gradient Norm: %.6f\n', iter, loss, grad_norm);
        end
        
        % Use approximate Newton method with diagonal Hessian for efficiency
        % Update W1
        for h = 1:hidden_dim
            % Use a diagonal approximation for the Hessian
            diag_h = h1(:, h) .* (1 - h1(:, h));
            H_diag = sum(X_batch.^2 .* repmat(diag_h, 1, input_dim), 1)' / batch_size + lambda;
            W1(:, h) = W1(:, h) - learning_rate * (grad_W1(:, h) ./ H_diag);
        end
        
        % Update W2
        for o = 1:output_dim
            % Use a diagonal approximation for the Hessian
            diag_h = h2(:, o) .* (1 - h2(:, o));
            H_diag = sum(h1.^2 .* repmat(diag_h, 1, hidden_dim), 1)' / batch_size + lambda;
            W2(:, o) = W2(:, o) - learning_rate * (grad_W2(:, o) ./ H_diag);
        end
        
        % Update biases with gradient descent
        b1 = b1 - learning_rate * grad_b1;
        b2 = b2 - learning_rate * grad_b2;
        
        % Stopping condition
        if grad_norm < eps_val
            break;
        end
    end
end
