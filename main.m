%____________________________________

input_dim = size(X_train, 2);
hidden_dim = 60;
output_dim = numClasses;
max_iter = 100;
learning_rate = 0.5;

fprintf('Training with Gradient Descent...\n');
tic;
[W1_gd, b1_gd, W2_gd, b2_gd, history_gd] = gradient_m(X_train, Y_train, input_dim, hidden_dim, output_dim, learning_rate, max_iter);
time_gd = toc;


y_pred_gd = predict(X_test, W1_gd, b1_gd, W2_gd, b2_gd);
fprintf('Evaluating Gradient Descent model...\n');
evaluate(Y_test, y_pred_gd);

fprintf('Training with Alternative Optimisation Method...\n');
tic;
[W1_ao, b1_ao, W2_ao, b2_ao, history_ao] = adam_m(X_train, Y_train, input_dim, hidden_dim, output_dim, learning_rate, max_iter);
time_ao = toc;

y_pred_ao = predict(X_test, W1_ao, b1_ao, W2_ao, b2_ao);
fprintf('Evaluating Newton model...\n');
evaluate(Y_test, y_pred_ao);

% Plotting
figure;
subplot(1,2,1);
plot(history_gd.loss, 'b', 'LineWidth',2); hold on;
plot(history_ao.loss, 'r', 'LineWidth',2);
xlabel('Iteration');
ylabel('Loss');
legend('Gradient Descent', 'Alternative Opt');
title('Loss over Iterations');

subplot(1,2,2);
plot(history_gd.grad_norm, 'b', 'LineWidth',2); hold on;
plot(history_ao.grad_norm, 'r', 'LineWidth',2);
xlabel('Iteration');
ylabel('Gradient Norm');
legend('Gradient Descent', 'Alternative Opt');
title('Gradient Norm over Iterations');

fprintf('Training times: GD = %.2f sec, AO = %.2f sec\n', time_gd, time_ao);