clc;
clear all;
%% 加载神经网络的训练样本 测试样本每列一个样本 输入P 输出T
%样本数据就是前面问题描述中列出的数据
load data;
 % 打乱原始数据的样本位置
temp1 = randperm(size(data,1));
P= data(1:10,temp1(1:11));
T= data(11:13,temp1(1:11));
 % 打乱原始数据的样本位置
temp2 = randperm(size(data,1));
% 测试集10个样本
P_test = data(1:10,temp2(1:10));
T_test = data(11:13,temp2(1:10));

% 初始隐层神经元个数
hiddennum=40;
% 输入向量的最大值和最小值
threshold= minmax(P) ;
inputnum=size(P,1);       % 输入层神经元个数
outputnum=size(T,1);      % 输出层神经元个数
w1num=inputnum*hiddennum; % 输入层到隐层的权值个数
w2num=outputnum*hiddennum;% 隐层到输出层的权值个数
N=w1num+hiddennum+w2num+outputnum; %待优化的变量的个数

%% 定义遗传算法参数
NIND=20;        %个体数目
MAXGEN=100;      %最大遗传代数
PRECI=10;       %变量的二进制位数
GGAP=0.95;      %代沟
px=0.7;         %交叉概率
pm=0.01;        %变异概率
trace=zeros(N+1,MAXGEN);                        %寻优结果的初始值

FieldD=[repmat(PRECI,1,N);repmat([-0.5;0.5],1,N);repmat([1;0;1;1],1,N)];                      %区域描述器
Chrom=crtbp(NIND,PRECI*N);                      %初始种群
%% 优化
gen=0;                                 %代计数器
X=bs2rv(Chrom,FieldD);                 %计算初始种群的十进制转换
ObjV=Objfun(X,P,T,hiddennum,P_test,T_test);        %计算目标函数值
while gen<MAXGEN
   fprintf('%d\n',gen)
   FitnV=ranking(ObjV);                              %分配适应度值
   SelCh=select('sus',Chrom,FitnV,GGAP);              %选择
   SelCh=recombin('xovsp',SelCh,px);                  %重组
   SelCh=mut(SelCh,pm);                               %变异
   X=bs2rv(SelCh,FieldD);               %子代个体的十进制转换
   ObjVSel=Objfun(X,P,T,hiddennum,P_test,T_test);             %计算子代的目标函数值
   [Chrom,ObjV]=reins(Chrom,SelCh,1,1,ObjV,ObjVSel); %重插入子代到父代，得到新种群
   X=bs2rv(Chrom,FieldD);
   gen=gen+1;                                             %代计数器增加
   %获取每代的最优解及其序号，Y为最优解,I为个体的序号
   [Y,I]=min(ObjV);
   trace(1:N,gen)=X(I,:);                       %记下每代的最优值
   trace(end,gen)=Y;                               %记下每代的最优值
end
%% 画进化图
figure(1);
P0 = plot(1:MAXGEN,trace(end,:));
grid on
xlabel('遗传代数')
ylabel('误差的变化')
title('进化过程')
bestX=trace(1:end-1,end);
bestErr=trace(end,end);
fprintf(['最优初始权值和阈值:\nX=',num2str(bestX'),'\n最小误差err=',num2str(bestErr),'\n'])
set(P0,'LineWidth',1.5);       %| 设置图形线宽


%% 比较优化前后的训练&测试
callELMfun