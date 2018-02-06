function [g_out, F_out, z_out, nb_grad] = approxgrad(yk,Ytrain,Xtrain,param,xk_past,yk_past)

%   Compute the approximate proximal point at yk: 
%      Warm start the subproblem according to different strategy
%      Solve the subproblem up to required accuracy
%
%   Outputs:  
%      z_out    approximate proximal point of yk
%      g_out    approximate gradient of the Moreau envelope at yk
%      F_out    approximate function value of the Moreau envelope at yk
%      nb_grad  nb of epochs to achieve the approximation
%


if strcmp(param.model,'lasso') || strcmp(param.model,'elasticnet')
    label = 0; % label fo square loss
    
    %%%%%%%%%%%%%%  WARM START  %%%%%%%%%%%%%%%%%%%
    if (param.warm_start == 0) % warm start at x_{k-1}
        z0 = xk_past;
    elseif (param.warm_start == 1)  % warm start at x_{k-1}+ kappa/(kappa+mu)*(y_{k-1}-y_{k-2})
        w0 = xk_past + (param.kappa/(param.kappa+param.mu))*(yk-yk_past);
        z0 = compute_prox(w0,Ytrain,Xtrain,param,yk);
    elseif (param.warm_start == 2) % warm start at y_{k-1}
        z0 = compute_prox(yk,Ytrain,Xtrain,param,yk); 
    elseif (param.warm_start == 3) % warm start at best (x_{k-1}, x_{k-1}+ kappa/(kappa+mu)*(y_{k-1}-y_{k-2})
        w0 = xk_past + (param.kappa/(param.kappa+param.mu))*(yk-yk_past);
        z0 = compute_prox(w0,Ytrain,Xtrain,param,w0);
        value0 = compute_loss(z0,Ytrain,Xtrain,param)+ 0.5*param.kappa*(z0-yk)'*(z0-yk);

        z1 = xk_past;
        value1 = compute_loss(z1,Ytrain,Xtrain,param)+ 0.5*param.kappa*(z1-yk)'*(z1-yk);

       if value1<value0
           z0 = z1;
           fprintf('best warm start = x(k-1) \n');
       end
    end
    
    %%%%%%%%%%%%%%  Solve the subproblem  %%%%%%%%%%%%%%%%%%%
    if strcmp(param.algo,'svrg')
        if strcmp(param.stop_criterion,'onepass')
            [z_out, nb_grad, F_out]=...
            mex_svrg_elasticnet_onepass(Ytrain,Xtrain,label,param.Lips,param.lambda,param.mu,z0,param.m,param.eta,param.kappa,yk); 
        elseif strcmp(param.stop_criterion,'absolute')
            [z_out, nb_grad, F_out]=...
            mex_svrg_elasticnet_eps(Ytrain,Xtrain,label,param.Lips,param.lambda,param.mu,z0,param.m,param.eta,param.kappa,yk,param.epsilon); 
        elseif strcmp(param.stop_criterion,'relative')
            [z_out, nb_grad, F_out]=...
            mex_svrg_elasticnet_delta(Ytrain,Xtrain,label,param.Lips,param.lambda,param.mu,z0,param.m,param.eta,param.kappa,yk,param.delta); 
        end
    end
elseif strcmp(param.model,'logi')
    label = 1; % label fo logistic loss
    
    %%%%%%%%%%%%%%  WARM START  %%%%%%%%%%%%%%%%%%%
    if (param.warm_start == 0) % warm start at x_{k-1}
        z0 = xk_past;
    elseif (param.warm_start == 1)  % warm start at x_{k-1}+ 0.5*(y_{k-1}-y_{k-2})
        z0  = xk_past + (param.kappa/(param.kappa+ param.mu))*(yk-yk_past);
    elseif (param.warm_start == 2)  % warm start at y_{k-1}
        z0 = yk;
    elseif (param.warm_start == 3)  % warm start at best (x_{k-1}, x_{k-1}+ kappa/(kappa+mu)*(y_{k-1}-y_{k-2})
        z0  = xk_past + (param.kappa/(param.kappa+param.mu))*(yk-yk_past);
        value0 = compute_loss(z0,Ytrain,Xtrain,param)+ 0.5*param.kappa*(z0-yk)'*(z0-yk);

        z1 = xk_past;
        value1 = compute_loss(z1,Ytrain,Xtrain,param)+ 0.5*param.kappa*(z1-yk)'*(z1-yk);
        if value1<value0
           z0 = z1;
           fprintf('warm start = x(k-1) \n');
        end
    end
    
    %%%%%%%%%%%%%%  Solve the subproblem  %%%%%%%%%%%%%%%%%%%
    if strcmp(param.algo,'svrg')
        if strcmp(param.stop_criterion,'onepass')
            [z_out, nb_grad,F_out]=...
                mex_svrg_smooth_onepass(Ytrain,Xtrain,label,param.Lips,param.mu,z0,param.m,param.eta,param.kappa,yk); 
        elseif strcmp(param.stop_criterion,'absolute')
            [z_out, nb_grad,F_out]=...
                mex_svrg_smooth_eps(Ytrain,Xtrain,label,param.Lips,param.mu,z0,param.m,param.eta,param.kappa,yk,param.epsilon); 
        elseif strcmp(param.stop_criterion,'relative')
            [z_out, nb_grad,F_out]=...
                mex_svrg_smooth_delta(Ytrain,Xtrain,label,param.Lips,param.mu,z0,param.m,param.eta,param.kappa,yk,param.delta); 
        end
    end
end

g_out = param.kappa*(yk-z_out);

end