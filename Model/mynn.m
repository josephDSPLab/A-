classdef mynn < handle
    properties
      input_num
      output_num
      bias_num
      weight
      delt
      act
      input
      output
      grad
      act_type
    end
    methods
        function obj = mynn(input_num, output_num,edge,bias_num,act_type)
            obj.input_num = input_num;
            obj.output_num = output_num;
            obj.bias_num = bias_num;
            obj.weight = normrnd(0,1/sqrt(8),[obj.output_num obj.input_num]);
            obj.act_type = act_type;
            if ~isempty(edge)
                for i = 1:size(edge,1)
                	obj.weight(edge(i,1),edge(i,2)) = 0;
                end
            end
            obj.act = zeros([obj.output_num 1]);
            obj.input = zeros([obj.input_num 1]);
            obj.delt = diag(ones([obj.output_num+obj.bias_num obj.input_num]));
            obj.grad = zeros([obj.output_num+obj.bias_num obj.input_num]);
        end
        function y = forward(obj,input)
            obj.output = obj.weight*input;
            if obj.act_type == 0
                obj.act = ReLU(obj.output);
            elseif obj.act_type == 1
                obj.act = obj.output;
            end
            obj.act = [obj.act; ones([obj.bias_num, 1])];
            y = obj.act;
            obj.input = input;
        end
        function del = backward(obj, det_t1)
            if obj.act_type == 0
                d_sig = [dReLU(obj.output);zeros(obj.bias_num,1)];
            elseif obj.act_type == 1
                d_sig = [ones(size(obj.output));zeros(obj.bias_num,1)];
            end
                
            del = (det_t1*diag(d_sig))*[obj.weight;zeros([obj.bias_num,obj.input_num])];
            obj.grad = (det_t1'.*d_sig ) * obj.input'; 
        end
        function optimize(obj, eta, lamb)
            obj.weight = obj.weight - eta*(obj.grad(1:size(obj.weight,1),1:size(obj.weight,2)) + lamb*obj.weight);
        end
    end

end
function y = sigmoid(x)
    y = 1./(1+exp(-x));
end
function y = delsigmoid(x)
    y = sigmoid(x).*sigmoid(-x);
end

function y = ReLU(x)
    y = max(0,x);
end

function  y = dReLU(x)
    y = zeros(size(x));
    y(x>0) = 1;
end