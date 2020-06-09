clc;
clear all;
%% �����������ѵ������ ��������ÿ��һ������ ����P ���T
%�������ݾ���ǰ�������������г�������
load data;
 % ����ԭʼ���ݵ�����λ��
temp1 = randperm(size(data,1));
P= data(1:10,temp1(1:11));
T= data(11:13,temp1(1:11));
 % ����ԭʼ���ݵ�����λ��
temp2 = randperm(size(data,1));
% ���Լ�10������
P_test = data(1:10,temp2(1:10));
T_test = data(11:13,temp2(1:10));

% ��ʼ������Ԫ����
hiddennum=40;
% �������������ֵ����Сֵ
threshold= minmax(P) ;
inputnum=size(P,1);       % �������Ԫ����
outputnum=size(T,1);      % �������Ԫ����
w1num=inputnum*hiddennum; % ����㵽�����Ȩֵ����
w2num=outputnum*hiddennum;% ���㵽������Ȩֵ����
N=w1num+hiddennum+w2num+outputnum; %���Ż��ı����ĸ���

%% �����Ŵ��㷨����
NIND=20;        %������Ŀ
MAXGEN=100;      %����Ŵ�����
PRECI=10;       %�����Ķ�����λ��
GGAP=0.95;      %����
px=0.7;         %�������
pm=0.01;        %�������
trace=zeros(N+1,MAXGEN);                        %Ѱ�Ž���ĳ�ʼֵ

FieldD=[repmat(PRECI,1,N);repmat([-0.5;0.5],1,N);repmat([1;0;1;1],1,N)];                      %����������
Chrom=crtbp(NIND,PRECI*N);                      %��ʼ��Ⱥ
%% �Ż�
gen=0;                                 %��������
X=bs2rv(Chrom,FieldD);                 %�����ʼ��Ⱥ��ʮ����ת��
ObjV=Objfun(X,P,T,hiddennum,P_test,T_test);        %����Ŀ�꺯��ֵ
while gen<MAXGEN
   fprintf('%d\n',gen)
   FitnV=ranking(ObjV);                              %������Ӧ��ֵ
   SelCh=select('sus',Chrom,FitnV,GGAP);              %ѡ��
   SelCh=recombin('xovsp',SelCh,px);                  %����
   SelCh=mut(SelCh,pm);                               %����
   X=bs2rv(SelCh,FieldD);               %�Ӵ������ʮ����ת��
   ObjVSel=Objfun(X,P,T,hiddennum,P_test,T_test);             %�����Ӵ���Ŀ�꺯��ֵ
   [Chrom,ObjV]=reins(Chrom,SelCh,1,1,ObjV,ObjVSel); %�ز����Ӵ����������õ�����Ⱥ
   X=bs2rv(Chrom,FieldD);
   gen=gen+1;                                             %������������
   %��ȡÿ�������Ž⼰����ţ�YΪ���Ž�,IΪ��������
   [Y,I]=min(ObjV);
   trace(1:N,gen)=X(I,:);                       %����ÿ��������ֵ
   trace(end,gen)=Y;                               %����ÿ��������ֵ
end
%% ������ͼ
figure(1);
P0 = plot(1:MAXGEN,trace(end,:));
grid on
xlabel('�Ŵ�����')
ylabel('���ı仯')
title('��������')
bestX=trace(1:end-1,end);
bestErr=trace(end,end);
fprintf(['���ų�ʼȨֵ����ֵ:\nX=',num2str(bestX'),'\n��С���err=',num2str(bestErr),'\n'])
set(P0,'LineWidth',1.5);       %| ����ͼ���߿�


%% �Ƚ��Ż�ǰ���ѵ��&����
callELMfun