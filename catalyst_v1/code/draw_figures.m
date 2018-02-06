function [] = draw_figures(dataset, model, mu, lambda, loadfilename_catalyst, loadfilename_svrg,loadfilename_quickening)

%  Draw Figures of comparison between SVRG/ Catalyst SVRG/ QuickeNing SVRG 

load(loadfilename_catalyst);
load(loadfilename_svrg);
load(loadfilename_quickening);


% Estimate f* in order to plot in log scale
%   limit0 : estimate of f*
%            obtained using daulity gap


if strcmp(dataset,'covtype')
    if strcmp(model,'logi') 
        if mu == 0.01
            limit0 = 6.59602842590e-01 ;
        elseif mu == 0.1
            limit0 = 6.65959759569e-01; 
        else
            ind_catalyst = find(dualgaplist_catalyst > 0);
            limit0 = max(train_loss_list_catalyst(ind_catalyst)-dualgaplist_catalyst(ind_catalyst));
            
            ind = find(dualgaplist > 0);
            limit0 = max(limit0, max(train_loss_list(ind)-dualgaplist(ind)));
            
            ind_qning = find(dualgaplist_qning > 0);
            limit0 = max(limit0, max(train_loss_list_qning(ind_qning)-dualgaplist_qning(ind_qning)));
            fprintf('limit0 = %0.11e \n',limit0);
       end
    elseif strcmp(model,'elasticnet')
        if mu == 0.01 && lambda == 10 
            limit0 = 4.74882807995e-01 ;
        elseif mu == 0.1 && lambda == 10 
            limit0 = 4.74951622582e-01;
        else
            ind_catalyst = find(dualgaplist_catalyst > 0);
            limit0 = max(train_loss_list_catalyst(ind_catalyst)-dualgaplist_catalyst(ind_catalyst));
            
            ind = find(dualgaplist > 0);
            limit0 = max(limit0, max(train_loss_list(ind)-dualgaplist(ind)));
            
            ind_qning = find(dualgaplist_qning > 0);
            limit0 = max(limit0, max(train_loss_list_qning(ind_qning)-dualgaplist_qning(ind_qning)));
            fprintf('limit0 = %0.11e \n',limit0);
        end
    elseif strcmp(model,'lasso')
        if lambda ==10
            limit0 = 4.74610034875e-01  ;
        else
            ind_catalyst = find(dualgaplist_catalyst > 0);
            limit0 = max(train_loss_list_catalyst(ind_catalyst)-dualgaplist_catalyst(ind_catalyst));
            
            ind = find(dualgaplist > 0);
            limit0 = max(limit0, max(train_loss_list(ind)-dualgaplist(ind)));
            
            ind_qning = find(dualgaplist_qning > 0);
            limit0 = max(limit0, max(train_loss_list_qning(ind_qning)-dualgaplist_qning(ind_qning)));
            fprintf('limit0 = %0.11e \n',limit0);
        end
    end
end
   

%%%%%%%%%%%%%%%%% Set up logarithemic scaling %%%%%%%%%%%%%%%%%%%% 
list_svrg_log = log10((train_loss_list-limit0)/limit0);
last_ind = find(list_svrg_log < -10,1);
if isempty(last_ind)
    last_ind = size(list_svrg_log,2);
end

list_catalyst_log = log10((train_loss_list_catalyst-limit0)/limit0);
last_ind_catalyst = find(list_catalyst_log < -10,1);
if isempty(last_ind_catalyst)
    last_ind_catalyst = size(list_catalyst_log,2);
end

list_qning_log = log10((train_loss_list_qning-limit0)/limit0);
last_ind_qning = find(list_qning_log < -10,1);
if isempty(last_ind_qning)
    last_ind_qning = size(list_qning_log,2);
end


if strcmp(model,'logi')
    titlename =  sprintf('%s, logistic, $\\mu$= %g/n',dataset,mu);
elseif strcmp(model,'elasticnet')
    titlename =  sprintf('%s, %s, $\\mu$= %g/n, $\\lambda$= %g/n',dataset, model, mu, lambda);
elseif strcmp(model,'lasso')
    titlename =  sprintf('%s, %s, $\\lambda$= %g /n',dataset, model,lambda);
end


%%%%% Plot train loss
x_label = 'Number of gradient evaluations';
y_label = 'Training loss (log scale)';
multiple_plot({list_svrg_log(1:last_ind),list_catalyst_log(1:last_ind_catalyst),list_qning_log(1:last_ind_qning)}, ...
               {it(1:last_ind),it_catalyst(1:last_ind_catalyst), it_qning(1:last_ind_qning)}, ...
               {'SVRG','Catalyst SVRG', 'QuickeNing SVRG'},...
               {'r','b','g'},x_label,titlename,y_label);
saveFile = sprintf('../figures/svrg/train_loss_%s_%s_mu=%g_lambda=%g.eps', dataset, model, mu, lambda);
r = 60; % pixels per inch
set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [0 0 1080 540]/r);
saveas(gcf,saveFile,'epsc')


%%%%% Plot test loss
x_label = 'Number of gradient evaluations';
y_label = 'Test loss';
multiple_plot({test_loss_list(1:end),test_loss_list_catalyst(1:end),test_loss_list_qning(1:end)}, ...
               {it(1:end),it_catalyst(1:end), it_qning(1:end)}, ...
               {'SVRG','Catalyst SVRG', 'QuickeNing SVRG'},...
               {'r','b','g'},x_label,titlename,y_label);
saveFile = sprintf('../figures/svrg/test_loss_%s_%s_mu=%g_lambda=%g.eps',dataset, model, mu, lambda);
r = 60; % pixels per inch
set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [0 0 1080 540]/r);
saveas(gcf,saveFile,'epsc')


%%%%% Plot test error
x_label = 'Number of gradient evaluations';
y_label = 'Test accuracy';
multiple_plot({test_acc_list(2:end),test_acc_list_catalyst(2:end),test_acc_list_qning(2:end)}, ...
               {it(2:end),it_catalyst(2:end), it_qning(2:end)}, ...
               {'SVRG','Catalyst SVRG', 'QuickeNing SVRG'}, ...
               {'r','b','g'},x_label,titlename,y_label);
saveFile = sprintf('../figures/svrg/test_error_%s_%s_mu=%g_lambda=%g.eps',dataset, model, mu, lambda);
r = 60; % pixels per inch
set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [0 0 1080 540]/r);
saveas(gcf,saveFile,'epsc')

end