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

subplot(1,2,1);
plot(history_gd.loss, 'b', 'LineWidth',2); hold on;
plot(history_nt.loss, 'g', 'LineWidth',2);
xlabel('Iteration');
ylabel('Loss');
legend('Gradient Descent', 'Newton');
title('Loss over Iterations');

subplot(1,2,2);
plot(history_gd.grad_norm, 'b', 'LineWidth',2); hold on;
plot(history_nt.grad_norm, 'g', 'LineWidth',2);
xlabel('Iteration');
ylabel('Gradient Norm');
legend('Gradient Descent', 'Newton');
title('Gradient Norm over Iterations');

fprintf('Training times: GD = %.2f sec, AO = %.2fsec NT = %.2fsec\n', time_gd, time_ao, time_nt);