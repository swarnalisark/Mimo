clear
numx=0;
M0203_sort = importfile("C:\Users\Swarnali\visionbf\vision-bf\data\UAV-benchmark-MOTD_v1.0\RES_MOT\YOLOv5\SORT\M0203.txt", [1, Inf]);
K = [10];
R_K = ones(length(K));

for ik = 1:length(K)
    k = K(ik); %SORT index
    moving_car_idx = M0203_sort.car_id == k;
    desired_data = M0203_sort(moving_car_idx,:);
    M = 4;
    N = 4;
    N_t = M*N;
    
    
    %%%%%%%%%finding a %%%%%%%
    theta = desired_data.theta; % class: double
    phi = desired_data.phi;
    
    % a_v shape: M rows, length(theta) columns
    % a_h shape: N rows, length(theta) columns
    a_v = ones(M,length(theta));
    a_h = ones(M,length(theta));
    a = [];
    
    for it = 1:length(theta)
        for im = 1:M
            a_v(im,it) = exp(-1i*pi*(im-1)*cos(theta(it)));
        end
    
        for in = 1:N
            a_h(in,it) = exp(-1i*pi*(in-1)*sin(theta(it))*cos(phi(it)));
        end
    
        a(:,it) = kron(a_v(:,it),a_h(:,it));
    
    end
    
    M0203_xywh_bbox = importvelocity("C:\Users\swarn\Documents\MATLAB\mimo\M0203.txt", [1, Inf]);
    desired_data_xywh = M0203_xywh_bbox(moving_car_idx,:);
    center_x = desired_data_xywh.x + desired_data_xywh.w/2;
    center_y = desired_data_xywh.y - desired_data_xywh.h/2;
    
    % u_k : v_x = center_x;
    % v_k : v_y
    v_y = center_y;
    
    c=3*10^8;
    f_c=60*10^9;
    
    rho_k = ones(length(v_y),1);
    for i = 1:length(v_y)
        rho_k(i) = (1/c)*v_y(i)*cos(theta(i))*f_c;
    end
    
    aplpha_til=10;
    
    distance = desired_data.distance;
    
    alpha_k = ones(length(distance),1);
    for i_alpha = 1:length(distance)
        alpha_k(i_alpha) = aplpha_til*(1/distance(i_alpha));
    end
    
    kappa=sqrt(N_t);
    
    h_k = ones(16,length(rho_k));
    for it=1:length(alpha_k)
        h_k(:,it) =kappa*alpha_k(it)*exp(1i*2*pi*rho_k(it)*it).*a(:,it);
    end
    
    h_k_T = h_k';
    f_k = (1/sqrt(N_t)).*a;
    
    %%% for p_k, p/sigma^2+sigma_est^2=30dbm
    gama_k = ones(length(alpha_k),1);
    
    for i = 1: length(alpha_k)
        gama_k(i) = 30*(abs(h_k_T(i,:)*f_k(:,i)))^2;
    end
  
end
%%%%%energy efficency%%%%%%
p0=30;
p=20:.1:40;
e=0.35;
EE=[];
for pp=20:0.1:40
    EE_single=(sum(log(1+gama_k),"all"))/(((1/e)*pp)+p0);
    EE=[EE;EE_single];
end

figure
plot(p,EE);
xlabel('Transmission Power(dBm');
ylabel('avarage energy efficiency (bits/Hz)')
title('avarage energy efficiency vs Transmission Power');
% str{ik}=['moving vehicle ',num2str(K(ik))]; 
% str{ik}=strcat(str{ik});
% legend(str(1:ik));
% hold on


   