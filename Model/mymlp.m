classdef mymlp < handle
    properties
        num_layer
        net_stack
        num_weight
        sigma
    end
    methods
        function obj = mymlp(node,edge,sigma)
            obj.net_stack = {};
            obj.num_layer = size(node,2)-1;
            obj.num_weight = 0;
            obj.sigma = sigma;
            for i=1:obj.num_layer
                if i==1
                    obj.net_stack{end+1} = mynn(node(i),node(i+1),edge,1,0);
                elseif i==obj.num_layer
                    obj.net_stack{end+1} = mynn(node(i)+1,node(i+1),edge,0,1);
                else
                    obj.net_stack{end+1} = mynn(node(i)+1,node(i+1),edge,1,0);
                end
                obj.num_weight = obj.num_weight + (obj.net_stack{i}.input_num)*(obj.net_stack{i}.input_num + obj.net_stack{i}.bias_num);
            end
        end
        function out = forward(obj,out)
            for i=1:obj.num_layer
                out = obj.net_stack{i}.forward(out);
            end 
        end
        function grad = backward(obj,delt)
            for i=obj.num_layer:-1:1
                delt = obj.net_stack{i}.backward(delt);
            end 
            grad = obj.net_stack{1}.grad;
        end
        function optimize(obj,eta,lamb)
            for i=1:obj.num_layer
                obj.net_stack{i}.optimize( eta, lamb);
            end 
        end
    end
end