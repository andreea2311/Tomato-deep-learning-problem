function [lsig] = log_sig(A)
    sig = 1.0 ./ ( 1.0 + exp(-A));
    lsig=log(sig);
end