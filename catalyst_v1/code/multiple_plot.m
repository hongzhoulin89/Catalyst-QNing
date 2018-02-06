function [] = multiple_plot(list_plot,list_it,list_label,list_color,x_label,titlename,y_label,legend_para,x_axis,yaxis,position)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
if nargin < 11
    position = 'NorthEast';
end

if nargin<10
    y_axis.max =inf;
    y_axis.min =-inf;
else
    y_axis.max = yaxis(2);
    y_axis.min = yaxis(1);
end

if nargin<9
    x_axis =100;
end

if nargin<8
    legend_para =1;
end

if nargin < 7
    y_label = 'Loss function';
end;

if nargin < 6
    titlename = 'Comparaison algorithm';
end;

if nargin < 5
    x_label = 'nb iteration';
end;

  
    figure; 
    
    k = size(list_plot,2);
    count = 0;
    

    
    for i=1:k
        count= count+1;
        plot(list_it{i},list_plot{i},list_color{i},'LineWidth',10);
        if count < k
            hold on;
        else
            hold off
            if legend_para
                [lgd,icons,plot_h] = legend(list_label,'Location',position,'FontSize', 20,'fontWeight','bold','Interpreter','latex'); 
            else
                legend('off')
            end
        end;
    end;
    %set(lgd,'Position',[0.585, 0.3895,0.366,0.338*2]);
    %set(lgd,'PlotBoxAspectRatio','auto')
    xlim([0,min(max(xlim),x_axis)]); 
    ylim([max([min(ylim),-10,y_axis.min]),min(max(ylim),y_axis.max)]);
    
    format long;
    xlabel(x_label,'FontSize', 40,'fontWeight','bold'); 
    ylabel(y_label,'FontSize', 40,'fontWeight','bold'); 
    
%     HeightScaleFactor =1.5;
%     NewHeight = lgd.Position(4)*HeightScaleFactor;
%     lgd.Position(2)=lgd.Position(2)-(NewHeight-lgd.Position(2));
%     lgd.Position(4)=NewHeight;
    
    tit=title(titlename,'FontSize', 50,'fontWeight','bold','Interpreter','latex');
    
    set(gca, 'FontSize', 25,'fontWeight','bold')  

end

