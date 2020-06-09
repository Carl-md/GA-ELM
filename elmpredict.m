function Y = elmpredict(P,IW,B,LW,TF,TYPE);
% 如果子函数输入参数少于6个，报错
if nargin < 6
    error('ELM:Arguments','Not enough input arguments.');
end

% Calculate the Layer Output Matrix H
%  计算输出层举证
%  获取测试输入样本的行数
Q = size(P,2);
% 根据输入样本行数，扩充偏差矩阵（复制方式）
BiasMatrix = repmat(B,1,Q);
% 计算测试输出矩阵
tempH = IW * P + BiasMatrix;
% 判断输出传递函数
switch TF
    case 'sig'
        H = 1 ./ (1 + exp(-tempH));
    case 'sin'
        H = sin(tempH);
    case 'hardlim'
        H = hardlim(tempH);
end

% Calculate the Simulate Output
% 计算函数F(iw*p+e)的结果
Y = (H' * LW)';

if TYPE == 1
    temp_Y = zeros(size(Y));
    for i = 1:size(Y,2)
        [max_Y,index] = max(Y(:,i));
        temp_Y(index,i) = 1;
    end
    Y = vec2ind(temp_Y); 
end

% ELMPREDICT Simulate a Extreme Learning Machine
% Syntax
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% Description
% Input
% P   - Input Matrix of Training Set  (R*Q)
% IW  - Input Weight Matrix (N*R)
% B   - Bias Matrix  (N*1)
% LW  - Layer Weight Matrix (N*S)
% TF  - Transfer Function:
%       'sig' for Sigmoidal function (default)
%       'sin' for Sine function
%       'hardlim' for Hardlim function
% TYPE - Regression (0,default) or Classification (1)
% Output
% Y   - Simulate Output Matrix (S*Q)
% Example
% Regression:
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20,'sig',0)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% Classification
% [IW,B,LW,TF,TYPE] = elmtrain(P,T,20,'sig',1)
% Y = elmtrain(P,IW,B,LW,TF,TYPE)
% See also ELMTRAIN
% Yu Lei,11-7-2010
% Copyright www.matlabsky.com
% $Revision:1.0 $