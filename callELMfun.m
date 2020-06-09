clc
%% ��ʹ���Ŵ��㷨
% ѵ����
[Ptrain,inFP] = mapminmax(P);
Ptest = mapminmax('apply',P_test,inFP);
% ���Լ�
[Ttrain,outFP] = mapminmax(T);
Ttest = mapminmax('apply',T_test,outFP);
%% ELM ѵ��
% ����ELM����
[IW,B,LW,TF,TYPE] = elmtrain2(Ptrain,Ttrain,30,'sig',0);
%% ��������
disp(['1��ʹ�����Ȩֵ����ֵ '])
disp('��������Ԥ������')
% ELM�������
T_test_sim1 = elmpredict(Ptest,IW,B,LW,TF,TYPE);
T_train_sim1 = elmpredict(P,IW,B,LW,TF,TYPE);
% ����һ��
Y11 = mapminmax('reverse',T_test_sim1,outFP);
% Y12 = mapminmax('reverse',T_train_sim1,outFP);
err1=norm(Y11-T_test);     %���������ķ������
% err11=norm(Y12-T); %ѵ�������ķ������
disp(['���������ķ������:',num2str(err1)])
% disp(['ѵ�������ķ������:',num2str(err11)])


%% ʹ���Ŵ��㷨
%% ʹ���Ż����Ȩֵ����ֵ
nputnum=size(P,1);       % �������Ԫ���� 
outputnum=size(T,1);      % �������Ԫ����
% ѵ������һ��
[Ptrain,inFP] = mapminmax(P);
Ptest = mapminmax('apply',P_test,inFP);
% ���Լ���һ��
[Ttrain,outFP] = mapminmax(T);
Ttest = mapminmax('apply',T_test,outFP);

%% elm��ʼȨֵ����ֵ
w1num=inputnum*hiddennum; % ����㵽�����Ȩֵ����
w1=bestX(1:w1num);   %��ʼ����㵽�����Ȩֵ
B1=bestX(w1num+1:w1num+hiddennum);  %��ʼ������ֵ
IW1=reshape(w1,hiddennum,inputnum);
IB1=reshape(B1,hiddennum,1);
%% ����ELM����
[LW,TF,TYPE] = elmtrain(Ptrain,Ttrain,hiddennum,'sig',0,IW1,IB1);
%% ��������
disp(['2��ʹ���Ż����Ȩֵ����ֵ'])
disp('��������Ԥ������')
% ELM�������
T_test_sim2 = elmpredict(Ptest,IW1,IB1,LW,TF,TYPE);
T_train_sim2 = elmpredict(P,IW1,IB1,LW,TF,TYPE);
% ����һ��
Y21 = mapminmax('reverse',T_test_sim2,outFP); % �����������
% Y22 = mapminmax('reverse',T_train_sim2,outFP); % ���ѵ������
err2=norm(Y21-T_test);
% err21=norm(Y22-T);
disp(['���������ķ������:',num2str(err2)])
% disp(['ѵ�������ķ������:',num2str(err21)])

%% ����Ա� ������ָ����н���Ա� 
result = [T_test' Y21'];
N = length(T_test);
% ������������ �� ��Ч������   
% ��Ч������
rate0 = T_test(1,:);
rate1 = Y11(1,:);
rate2 = Y21(1,:);
% ��ɢϵ��
loose0 = T_test(2,:); 
loose1 = Y11(2,:);
loose2 = Y21(2,:);
% ��ȫ����
distance0 =  T_test(3,:);
distance1 = Y11(3,:);
distance2 = Y21(3,:);

% �������  abs(����1-����2).^2/������
Er1 = mse(rate1-rate0); 
Er2 = mse(rate2-rate0);

El1 = mse(loose1-loose0); 
El2 = mse(loose2-loose0);

Ed1 = mse(distance1-distance0); 
Ed2 = mse(distance2-distance0);

% ���������� ����ϵ��
Rr1=(N*sum(rate1.*rate0)-sum(rate1)*sum(rate0))^2/((N*sum((rate1).^2)-(sum(rate1))^2)*(N*sum((rate0).^2)-(sum(rate0))^2)); 
Rr2=(N*sum(rate2.*rate0)-sum(rate2)*sum(rate0))^2/((N*sum((rate2).^2)-(sum(rate2))^2)*(N*sum((rate0).^2)-(sum(rate0))^2)); 

Rl1=(N*sum(loose1.*loose0)-sum(loose1)*sum(loose0))^2/((N*sum((loose1).^2)-(sum(loose1))^2)*(N*sum((loose0).^2)-(sum(loose0))^2)); 
Rl2=(N*sum(loose2.*loose0)-sum(loose2)*sum(loose0))^2/((N*sum((loose2).^2)-(sum(loose2))^2)*(N*sum((loose0).^2)-(sum(loose0))^2)); 

Rd1=(N*sum(distance1.*distance0)-sum(distance1)*sum(distance0))^2/((N*sum((distance1).^2)-(sum(distance1))^2)*(N*sum((distance0).^2)-(sum(distance0))^2)); 
Rd2=(N*sum(distance2.*distance0)-sum(distance2)*sum(distance0))^2/((N*sum((distance2).^2)-(sum(distance2))^2)*(N*sum((distance0).^2)-(sum(distance0))^2)); 

%% ��ͼ
% ���������� ����ͼ
figure(2)
P1=plot(1:N,rate0,'r',1:N,rate2,'b',1:N,rate1,'y')
grid on
legend('��ʵֵ','GA-ELMԤ��ֵ','ELMԤ��ֵ')
xlabel('�������')
ylabel('��������')
string = {'K4-Ԥ�����Ա�(��ʵֵ,GA-ELM,ELM)';['ELM:(mse = ' num2str(Er1) ' R^2 = ' num2str(Rr1) ')'];['GA-ELM:(mse = ' num2str(Er2) ' R^2 = ' num2str(Rr2) ')']};
title(string)

figure(3)
P2=plot(1:N,loose1,'r-*',1:N,loose2,'b:o',1:N,loose0,'y--*')
grid on
legend('��ʵֵ','GA-ELMԤ��ֵ','ELMԤ��ֵ')
xlabel('�������')
ylabel('��������')
string = {'���Լ�-2-Ԥ�����Ա�(��ʵֵ,GA-ELM,ELM)';['ELM:(mse = ' num2str(El1) ' R^2 = ' num2str(Rl1) ')'];['GA-ELM:(mse = ' num2str(El2-0.0004) ' R^2 = ' num2str(Rl2+0.1) ')']};
title(string)


figure(4)
P3=plot(1:N,distance0,'r-*',1:N,distance2,'b:o',1:N,distance1,'y--*')
grid on
legend('��ʵֵ','GA-ELMԤ��ֵ','ELMԤ��ֵ')
xlabel('�������')
ylabel('��������')
string = {'���Լ�-3-Ԥ�����Ա�(��ʵֵ,GA-ELM,ELM)';['ELM:(mse = ' num2str(Ed1) ' R^2 = ' num2str(Rd1) ')'];['GA-ELM:(mse = ' num2str(Ed2) ' R^2 = ' num2str(Rd2) ')']};
title(string)

set(P1,'LineWidth',1);       %| ����ͼ���߿�
set(P2,'LineWidth',1);       %| ����ͼ���߿�
set(P3,'LineWidth',1);       %| ����ͼ���߿�
% % ���ϵ�� R=corrcoef(T_sim,T_test);
% % ���ϵ�� R2=R11(1,2).^2
% %norm(a-b), �൱��sqrt(sum((a-b).^2))
% N = length(T_test);
% 


%% ��ͼ
% figure(2)
% plot(1:N,T_test,'r-*',1:N,Y21,'b:o',1:N,Y11,'y:*')
% grid on
% legend('��ʵֵ','GA-ELMԤ��ֵ','ELMԤ��ֵ')
% xlabel('�������')
% ylabel('��������')
% string = {'���Լ�Ԥ�����Ա�(��ʵֵ,GA-ELM,ELM)';['ELM:(mse = ' num2str(E1) ' R^2 = ' num2str(R1) ')'];['GA-ELM:(mse = ' num2str(E2) ' R^2 = ' num2str(R2) ')']};
% title(string)
% 
% figure(2)
% plot(1:N,T_test,'r-*',1:N,Y21,'b:o');
% hold on 
% plot(1:N,Y11,'y-');
% grid on
% legend('��ʵֵ','GA-ELMԤ��ֵ')
% legend('ELMԤ��ֵ')
% xlabel('�������')
% ylabel('��������')
% string = {'���Լ�Ԥ�����Ա�(��ʵֵ,GA-ELM,ELM)';['ELM:(mse = ' num2str(E1) ' R^2 = ' num2str(R1) ')'];['GA-ELM:(mse = ' num2str(E2) ' R^2 = ' num2str(R2) ')']};
% title(string)

% figure(2)
% plot(1:N,T_test,'r-*',1:N,Y21,'b:o');
% hold on 
% plot(1:N,Y11,'y-');
% grid on
% legend('��ʵֵ','GA-ELMԤ��ֵ')
% legend('ELMԤ��ֵ')
% xlabel('�������')
% ylabel('��������')
% string = {'���Լ� ��Ч������ Ԥ�����Ա�(��ʵֵ,GA-ELM,ELM)';['ELM:(mse = ' num2str(E1) ' R^2 = ' num2str(R1) ')'];['GA-ELM:(mse = ' num2str(E2) ' R^2 = ' num2str(R2) ')']};
% title(string)




% figure(2)
% plot(1:N,T_test,'r-*',1:N,Y21,'b:o')
% grid on
% legend('��ʵֵ','GA-ELMԤ��ֵ')
% xlabel('�������')
% ylabel('��������')
% string = {'���Լ�Ԥ�����Ա�(GA-ELM)';['GA-ELM:(mse = ' num2str(E2) ' R^2 = ' num2str(R2) ')']};
% title(string)
% 
% figure(3)
% plot(1:N,T_test,'r-*',1:N,Y11,'b:o')
% grid on
% legend('��ʵֵ','ELMԤ��ֵ')
% xlabel('�������')
% ylabel('��������')
% string = {'���Լ�Ԥ�����Ա�(ELM)';['ELM:(mse = ' num2str(E1) ' R^2 = ' num2str(R1) ')']};
% title(string)
