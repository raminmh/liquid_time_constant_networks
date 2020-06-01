function dydt = ctrnn_def(t,y,gt,inputs,w_l,b_l,w_l_r,b_l_r,tau,N,k,activation_)
% Implementation of CT-RNN

    dydt = zeros(N,1);

            for i= 1:size(inputs,1)
                % Interpolate the data set (gt,g) at time t
                g_l(:,i) = interp1(gt,inputs(i,:),t);
            end
            
    if isequal(activation_,'sigmoid')
    net_out = 1 ./ (1 + exp(-(reshape(w_l(1,1:size(inputs,1),:),size(inputs,1),k)'*g_l' + reshape(b_l(1,:,:),k,1))));
    net_recurr = 1 ./ (1 +exp(-(reshape(w_l_r(1,:,:),k,k)'*y(1:k) + reshape(b_l_r(1,:,:),k,1))));
    end
    
    if isequal(activation_,'relu')
    net_out = max(0,(reshape(w_l(1,1:size(inputs,1),:),size(inputs,1),k)'*g_l' + reshape(b_l(1,:,:),k,1)));
    net_recurr = max(0,(reshape(w_l_r(1,:,:),k,k)'*y(1:k) + reshape(b_l_r(1,:,:),k,1)));
    end
    
    if isequal(activation_,'tanh')
    net_out = tanh(reshape(w_l(1,1:size(inputs,1),:),size(inputs,1),k)'*g_l' + reshape(b_l(1,:,:),k,1));
    net_recurr = tanh(reshape(w_l_r(1,:,:),k,k)'*y(1:k) + reshape(b_l_r(1,:,:),k,1));
    end
    
    if isequal(activation_,'Htanh')
    net_out = Htanh(reshape(w_l(1,1:size(inputs,1),:),size(inputs,1),k)'*g_l' + reshape(b_l(1,:,:),k,1));
    net_recurr = Htanh(reshape(w_l_r(1,:,:),k,k)'*y(1:k) + reshape(b_l_r(1,:,:),k,1));
    end
            
            for i = 1:k
                 dydt(i) = -y(i)./tau(i,1) + net_out(i,:)';
                 %dydt(i) = net_out(i,:)';
                 %dydt(i) = -y(i).*(1./tau(i,1)+abs(net_out(i,:)')) + 10*net_out(i,:)';
            end
            
            if N/k >1
            for j = 2:N/k
                
                if isequal(activation_,'sigmoid')
                net_out((j-1)*k+1:j*k,:) = 1 ./ (1 + exp(-(reshape(w_l(j,:,:),k,k)'*y((j-2)*k+1:(j-1)*k,:) + reshape(b_l(j,:,:),k,1))));
                %net_recurr((j-1)*k+1:j*k,:) = 1 ./ (1 + exp(-(reshape(w_l_r(j,:,:),k,k)'*y((j-1)*k+1:(j)*k,:) + reshape(b_l_r(j,:,:),k,1))));
                end
                if isequal(activation_,'relu')
                net_out((j-1)*k+1:j*k,:) = max(0,(reshape(w_l(j,:,:),k,k)'*y((j-2)*k+1:(j-1)*k,:) + reshape(b_l(j,:,:),k,1)));
                %net_recurr((j-1)*k+1:j*k,:) = max(0,(reshape(w_l_r(j,:,:),k,k)'*y((j-1)*k+1:(j)*k,:) + reshape(b_l_r(j,:,:),k,1)));
                end

                if isequal(activation_,'tanh')
                net_out((j-1)*k+1:j*k,:) = tanh(reshape(w_l(j,:,:),k,k)'*y((j-2)*k+1:(j-1)*k,:) + reshape(b_l(j,:,:),k,1));
                %net_recurr((j-1)*k+1:j*k,:) = tanh(reshape(w_l_r(j,:,:),k,k)'*y((j-1)*k+1:(j)*k,:) + reshape(b_l_r(j,:,:),k,1));
                end
                
                if isequal(activation_,'Htanh')
                net_out((j-1)*k+1:j*k,:) = Htanh(reshape(w_l(j,:,:),k,k)'*y((j-2)*k+1:(j-1)*k,:) + reshape(b_l(j,:,:),k,1));
                %net_recurr((j-1)*k+1:j*k,:) = Htanh(reshape(w_l_r(j,:,:),k,k)'*y((j-1)*k+1:(j)*k,:) + reshape(b_l_r(j,:,:),k,1));
                end   
                
                for i = (j-1)*k+1:j*k
                    dydt(i) = -y(i)./tau(i,1) + net_out(i,:)';
                    %dydt(i) = net_out(i,:)';
                    %dydt(i) = -y(i).*(1./tau(i,1)+abs(net_out(i,:)')) + 10*net_out(i,:)';
                end  
            end
            end
end

