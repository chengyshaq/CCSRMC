% LINE COLORS 
N=3; 
x = -10:2:10;
y1 = Y1;
y2 = Y2;
y3 = Y3;
C = linspecer(N); 
% axis([0.3,0.9,0,1])
axes('NextPlot','replacechildren', 'ColorOrder',C); 
plot(x,y1,'-*',x,y2,'-*',x,y3,'-*','linewidth',1)
xlim([-10,10])
ylim([0,1]);
legend('\alpha','\beta','\gamma');   %
xlabel('log(¦Ë)')  
ylabel('Coverage') 
%ylabel('Average precision')
% ylabel('Area Under Curve')%AUC
 %ylabel('Ranking loss')
%ylabel('Hamming loss')
  %ylabel('One error')
title('Entertainment')
% saveas(gcf,'C:\Users\jack\Desktop\LE-TLLR-Broken line diagram of defect test results\rl\Yeast.emf')
