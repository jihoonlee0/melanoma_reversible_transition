surprisal

lnall=log(all); %400*17
[u r v]=svd(lnall,0);
vt=v';
lambdaall=r*vt;
Gall=u;

%energy
e=zeros(4,4);
for i=1:4
    for j=1:4
        e(i,j)=lambdaall(i,j)*sum(all(:,j).*Gall(:,i))
    end
end
all


n=16;%analyte number

%for single-cell
lambdaall=u*r;
gall=vt;

    n=20
    GkLkall=[]; %kth l*g
    k=2;
    for i=1:n
        l= lambda2cell (k+1,i);
        g=G2cell (:,k+1);
        lg=l*g;
        GkLkall=[GkLkall,lg];

    end
    G2L2all=GkLkall;


single-cell
lambdaall=u*r;
gall=vt;