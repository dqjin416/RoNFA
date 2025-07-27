%The Dog matrix
function dist=dogdistm(x,sigm1)
sigm2=3*sigm1;
[num,~]=size(x);
if num==1
   Distx=0; 
else
    Distx=squareform(pdist(x,'cosine'));
end
dist=1.5*exp(-1/2*(Distx/sigm1^2))-0.5*(exp(-1/2*(Distx/sigm2^2)));
