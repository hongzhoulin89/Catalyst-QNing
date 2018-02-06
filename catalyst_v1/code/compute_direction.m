function [d, s_list, y_list, rho_list]= compute_direction(s_list,y_list,rho_list,s,y,g_test,param)

% Description: Apply L-BFGS with inexact gradient 



l = param.limit_mem;
kappa = param.kappa;
sTy = s'*y;
scale_parameter = sTy/(y'*y);

if param.lbfgs_type == 0  %%%%% Check sTy >0
    if (sTy>0) 
        if size(s_list,2)< l
            s_list = [s_list s];
            y_list = [y_list y];
            rho_list = [rho_list 1/sTy];
        else
            s_list = [s_list(:,2:end) s];
            y_list = [y_list(:,2:end) y];
            rho_list = [rho_list(:,2:end) 1/sTy];
        end
        d = tworec(s_list, y_list, rho_list, g_test, scale_parameter*g_test); 
        fprintf('Bk updated \n');
    else
        d = tworec(s_list, y_list, rho_list, g_test, scale_parameter*g_test); 
        fprintf('Bk not update \n');
    end
elseif param.lbfgs_type == 1 %%%%%  Check two conditions: sTy > c1*|s|^2 and sTy > c2*|y|^2
    
        muF = param.mu*kappa/(param.mu+kappa);
        bool1 = (sTy> param.c1*muF*s'*s);
        bool2 = (sTy > param.c2*y'*y/kappa);

        if (bool1 && bool2)  
            if size(s_list,2)< l
                s_list = [s_list s];
                y_list = [y_list y];
                rho_list = [rho_list 1/sTy];
            else
                s_list = [s_list(:,2:end) s];
                y_list = [y_list(:,2:end) y];
                rho_list = [rho_list(:,2:end) 1/sTy];
            end
            d = tworec(s_list, y_list, rho_list, g_test, scale_parameter*g_test); 
            fprintf('Bk updated \n');
        else
            d = tworec(s_list, y_list, rho_list, g_test, scale_parameter*g_test); 
            fprintf('Bk not update \n');
        end
end
    



end