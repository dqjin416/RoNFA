%The classification algorithm
function [Yt,test_time]=MSNMclassifier(S_set,S_set_label,Q_set)
W0=MSNMtrain(S_set_label);%The Connection Matrix
[mt,~]=size(Q_set);
Yt=zeros(mt,1);
tic;
adt=100;%The number of finding the maximum sigm
for i=1:mt
    sigm1=1;
    sigm1U=sigm1*3;
    sigm1L=sigm1/3;       
    xt=Q_set(i,:);
    ud=MSNMtest(xt,S_set,S_set_label,sigm1,W0);
    [ud_pos,~]=size(ud(ud>0));
    for j=1:adt       
        if ud_pos~=1
            [sigm1,sigm1U,sigm1L]=SigAdapt(ud_pos,sigm1,sigm1U,sigm1L);           
            [ud,~]=MSNMtest(xt,S_set,S_set_label,sigm1,W0);
            [ud_pos,~]=size(ud(ud>0));
        end
    end
    [~,c]=max(ud);
    Yt(i)=c;  
    
end
test_time=toc;
end