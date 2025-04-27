input_dim = size(X_train, 2);
hidden_dim = 50;
output_dim = numClasses;
max_iter = 300;
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
fprintf('Evaluating Alternative Optimisation model...\n');
evaluate(Y_test, y_pred_ao);

fprintf('Training with Newton...\n');
tic;
[W1_nt, b1_nt, W2_nt, b2_nt, history_nt] = newton(X_train, Y_train, input_dim, hidden_dim, output_dim, max_iter);
time_nt = toc;

y_pred_gd = predict(X_test, W1_nt, b1_nt, W2_nt, b2_nt);
fprintf('Evaluating Newton model...\n');
evaluate(Y_test, y_pred_nt);

