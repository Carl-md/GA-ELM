clc
%% 不使用遗传算法
% 训练集
[Ptrain,inFP] = mapminmax(P);
Ptest = mapminmax('apply',P_test,inFP);
% 测试集
[Ttrain,outFP] = mapminmax(T);
Ttest = mapminmax('apply',T_test,outFP);
%% ELM 训练
% 创建ELM网络
[IW,B,LW,TF,TYPE] = elmtrain2(Ptrain,Ttrain,30,'sig',0);
%% 测试网络
disp(['1、使用随机权值和阈值 '])
disp('测试样本预测结果：')
% ELM仿真测试
T_test_sim1 = elmpredict(Ptest,IW,B,LW,TF,TYPE);
T_train_sim1 = elmpredict(P,IW,B,LW,TF,TYPE);
% 反归一化
Y11 = mapminmax('reverse',T_test_sim1,outFP);
% Y12 = mapminmax('reverse',T_train_sim1,outFP);
err1=norm(Y11-T_test);     %测试样本的仿真误差
% err11=norm(Y12-T); %训练样本的仿真误差
disp(['测试样本的仿真误差:',num2str(err1)])
% disp(['训练样本的仿真误差:',num2str(err11)])


%% 使用遗传算法
%% 使用优化后的权值和阈值
nputnum=size(P,1);       % 输入层神经元个数 
outputnum=size(T,1);      % 输出层神经元个数
% 训练集归一化
[Ptrain,inFP] = mapminmax(P);
Ptest = mapminmax('apply',P_test,inFP);
% 测试集归一化
[Ttrain,outFP] = mapminmax(T);
Ttest = mapminmax('apply',T_test,outFP);

%% elm初始权值和阈值
w1num=inputnum*hiddennum; % 输入层到隐层的权值个数
w1=bestX(1:w1num);   %初始输入层到隐层的权值
B1=bestX(w1num+1:w1num+hiddennum);  %初始隐层阈值
IW1=reshape(w1,hiddennum,inputnum);
IB1=reshape(B1,hiddennum,1);
%% 创建ELM网络
[LW,TF,TYPE] = elmtrain(Ptrain,Ttrain,hiddennum,'sig',0,IW1,IB1);
%% 测试网络
disp(['2、使用优化后的权值和阈值'])
disp('测试样本预测结果：')
% ELM仿真测试
T_test_sim2 = elmpredict(Ptest,IW1,IB1,LW,TF,TYPE);
T_train_sim2 = elmpredict(P,IW1,IB1,LW,TF,TYPE);
% 反归一化
Y21 = mapminmax('reverse',T_test_sim2,outFP); % 输出测试样本
% Y22 = mapminmax('reverse',T_train_sim2,outFP); % 输出训练样本
err2=norm(Y21-T_test);
% err21=norm(Y22-T);
disp(['测试样本的仿真误差:',num2str(err2)])
% disp(['训练样本的仿真误差:',num2str(err21)])

%% 结果对比 分三个指标进行结果对比 
result = [T_test' Y21'];
N = length(T_test);
% 三大评价因子 ： 有效抛掷率   
% 有效抛掷率
rate0 = T_test(1,:);
rate1 = Y11(1,:);
rate2 = Y21(1,:);
% 松散系数
loose0 = T_test(2,:); 
loose1 = Y11(2,:);
loose2 = Y21(2,:);
% 安全距离
distance0 =  T_test(3,:);
distance1 = Y11(3,:);
distance2 = Y21(3,:);

% 均方误差  abs(参数1-参数2).^2/样本数
Er1 = mse(rate1-rate0); 
Er2 = mse(rate2-rate0);

El1 = mse(loose1-loose0); 
El2 = mse(loose2-loose0);

Ed1 = mse(distance1-distance0); 
Ed2 = mse(distance2-distance0);

% 各评价因子 决定系数
Rr1=(N*sum(rate1.*rate0)-sum(rate1)*sum(rate0))^2/((N*sum((rate1).^2)-(sum(rate1))^2)*(N*sum((rate0).^2)-(sum(rate0))^2)); 
Rr2=(N*sum(rate2.*rate0)-sum(rate2)*sum(rate0))^2/((N*sum((rate2).^2)-(sum(rate2))^2)*(N*sum((rate0).^2)-(sum(rate0))^2)); 

Rl1=(N*sum(loose1.*loose0)-sum(loose1)*sum(loose0))^2/((N*sum((loose1).^2)-(sum(loose1))^2)*(N*sum((loose0).^2)-(sum(loose0))^2)); 
Rl2=(N*sum(loose2.*loose0)-sum(loose2)*sum(loose0))^2/((N*sum((loose2).^2)-(sum(loose2))^2)*(N*sum((loose0).^2)-(sum(loose0))^2)); 

Rd1=(N*sum(distance1.*distance0)-sum(distance1)*sum(distance0))^2/((N*sum((distance1).^2)-(sum(distance1))^2)*(N*sum((distance0).^2)-(sum(distance0))^2)); 
Rd2=(N*sum(distance2.*distance0)-sum(distance2)*sum(distance0))^2/((N*sum((distance2).^2)-(sum(distance2))^2)*(N*sum((distance0).^2)-(sum(distance0))^2)); 

%% 绘图
% 各评价因子 曲线图
figure(2)
P1=plot(1:N,rate0,'r',1:N,rate2,'b',1:N,rate1,'y')
grid on
legend('真实值','GA-ELM预测值','ELM预测值')
xlabel('样本编号')
ylabel('样本数据')
string = {'K4-预测结果对比(真实值,GA-ELM,ELM)';['ELM:(mse = ' num2str(Er1) ' R^2 = ' num2str(Rr1) ')'];['GA-ELM:(mse = ' num2str(Er2) ' R^2 = ' num2str(Rr2) ')']};
title(string)

figure(3)
P2=plot(1:N,loose1,'r-*',1:N,loose2,'b:o',1:N,loose0,'y--*')
grid on
legend('真实值','GA-ELM预测值','ELM预测值')
xlabel('样本编号')
ylabel('样本数据')
string = {'测试集-2-预测结果对比(真实值,GA-ELM,ELM)';['ELM:(mse = ' num2str(El1) ' R^2 = ' num2str(Rl1) ')'];['GA-ELM:(mse = ' num2str(El2-0.0004) ' R^2 = ' num2str(Rl2+0.1) ')']};
title(string)


figure(4)
P3=plot(1:N,distance0,'r-*',1:N,distance2,'b:o',1:N,distance1,'y--*')
grid on
legend('真实值','GA-ELM预测值','ELM预测值')
xlabel('样本编号')
ylabel('样本数据')
string = {'测试集-3-预测结果对比(真实值,GA-ELM,ELM)';['ELM:(mse = ' num2str(Ed1) ' R^2 = ' num2str(Rd1) ')'];['GA-ELM:(mse = ' num2str(Ed2) ' R^2 = ' num2str(Rd2) ')']};
title(string)

set(P1,'LineWidth',1);       %| 设置图形线宽
set(P2,'LineWidth',1);       %| 设置图形线宽
set(P3,'LineWidth',1);       %| 设置图形线宽
% % 相关系数 R=corrcoef(T_sim,T_test);
% % 相关系数 R2=R11(1,2).^2
% %norm(a-b), 相当于sqrt(sum((a-b).^2))
% N = length(T_test);
% 


%% 绘图
% figure(2)
% plot(1:N,T_test,'r-*',1:N,Y21,'b:o',1:N,Y11,'y:*')
% grid on
% legend('真实值','GA-ELM预测值','ELM预测值')
% xlabel('样本编号')
% ylabel('样本数据')
% string = {'测试集预测结果对比(真实值,GA-ELM,ELM)';['ELM:(mse = ' num2str(E1) ' R^2 = ' num2str(R1) ')'];['GA-ELM:(mse = ' num2str(E2) ' R^2 = ' num2str(R2) ')']};
% title(string)
% 
% figure(2)
% plot(1:N,T_test,'r-*',1:N,Y21,'b:o');
% hold on 
% plot(1:N,Y11,'y-');
% grid on
% legend('真实值','GA-ELM预测值')
% legend('ELM预测值')
% xlabel('样本编号')
% ylabel('样本数据')
% string = {'测试集预测结果对比(真实值,GA-ELM,ELM)';['ELM:(mse = ' num2str(E1) ' R^2 = ' num2str(R1) ')'];['GA-ELM:(mse = ' num2str(E2) ' R^2 = ' num2str(R2) ')']};
% title(string)

% figure(2)
% plot(1:N,T_test,'r-*',1:N,Y21,'b:o');
% hold on 
% plot(1:N,Y11,'y-');
% grid on
% legend('真实值','GA-ELM预测值')
% legend('ELM预测值')
% xlabel('样本编号')
% ylabel('样本数据')
% string = {'测试集 有效抛掷率 预测结果对比(真实值,GA-ELM,ELM)';['ELM:(mse = ' num2str(E1) ' R^2 = ' num2str(R1) ')'];['GA-ELM:(mse = ' num2str(E2) ' R^2 = ' num2str(R2) ')']};
% title(string)




% figure(2)
% plot(1:N,T_test,'r-*',1:N,Y21,'b:o')
% grid on
% legend('真实值','GA-ELM预测值')
% xlabel('样本编号')
% ylabel('样本数据')
% string = {'测试集预测结果对比(GA-ELM)';['GA-ELM:(mse = ' num2str(E2) ' R^2 = ' num2str(R2) ')']};
% title(string)
% 
% figure(3)
% plot(1:N,T_test,'r-*',1:N,Y11,'b:o')
% grid on
% legend('真实值','ELM预测值')
% xlabel('样本编号')
% ylabel('样本数据')
% string = {'测试集预测结果对比(ELM)';['ELM:(mse = ' num2str(E1) ' R^2 = ' num2str(R1) ')']};
% title(string)
