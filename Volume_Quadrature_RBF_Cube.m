function Quadrature_Weights=Volume_Quadrature_RBF_Cube(Quadrature_Nodes,Tetrahedra,Polynomial_Order_Volume,Number_of_Nearest_Neighbors_Volume)
%==========================================================================
%
% This function computes quadrature weights for evaluating the volume
% integral of a scalar function f(x,y,z) over the unit cube.
%
%   Inputs: Quadrature_Nodes - A set of points located on or inside the
%   cube (Number_of_Quadrature_Nodes X 3) Array
% 
%   Tetrahedra - A tesselation of the set Quadrature_Nodes.  This
%           should be an array where row k contains the indicdes in
%           Quadrature_Nodes of the vertices of tetrahedron k
%
%   Polynomial_Order_Volume - The order of trivariate polynomials to be
%           used along with RBFs in the approximation of the integrand
%
%   Number_of_Nearest_Neighbors_Volume - The number of RBF centers to use
%           in the approximation over a single tetrahedron in the array 
%           Tetrahedra.
%
%   Output: Quadrature Weights - A set of quadrature weights-
%   (Number_of_Quadrature_Nodes X 1) vector-corresponding
%   to the set of points Quadrature_Nodes
%
% This implementation uses the method and default settings discussed in the
% preprint:
%
% J. A. Reeger. "Approximation of Definite Integrals Over the Volume of the Ball."
%
% NOTE: The main loop of this method (over each Tetrahedron) can be easily
% changed to a standar for loop if you do not have access to the parallel 
% toolbox.  In such a case, change the parfor loop to a for loop.
%
%==========================================================================

warning off all

%==========================================================================
% Compute legendre weights and nodes for numerical integration of
% polynomial terms and for the integral over lambda_1 over the tetrahedra.
%==========================================================================
n=20;
[Legendre_Nodes,Legendre_Weights]=Legendre_Gauss_Lobatto_Quadrature_Nodes(n,0,1);
Legendre_Weights_1D=flipud(Legendre_Weights);
Legendre_Nodes_1D=flipud(Legendre_Nodes.');

[Legendre_Nodes_3D_x,Legendre_Nodes_3D_y,Legendre_Nodes_3D_z]=ndgrid(Legendre_Nodes_1D);
[Legendre_Weights_3D_x,Legendre_Weights_3D_y,Legendre_Weights_3D_z]=ndgrid(Legendre_Weights_1D);
Legendre_Nodes_3D=[Legendre_Nodes_3D_x(:),Legendre_Nodes_3D_y(:),Legendre_Nodes_3D_z(:)];
Legendre_Weights_3D=Legendre_Weights_3D_x(:).*Legendre_Weights_3D_y(:).*Legendre_Weights_3D_z(:);
%==========================================================================

%==========================================================================
% Problem information
%==========================================================================
% Parameter set by the choice of quadrature nodes
Number_of_Quadrature_Nodes=size(Quadrature_Nodes,1); %N in the paper

% Determine the number of tetrahedra in the tesellation.
Number_of_Tetrahedra=size(Tetrahedra,1);

% Find the midpoints of the tetrahedra
Tetrahedra_Midpoints=(Quadrature_Nodes(Tetrahedra(:,1),:)+...
    Quadrature_Nodes(Tetrahedra(:,2),:)+Quadrature_Nodes(Tetrahedra(:,3),:)+Quadrature_Nodes(Tetrahedra(:,4),:))/4;

% Find the quadrature nodes closest to the midpoint of the tetrahedron
Nearest_Neighbor_Indices_Tetrahedra=knnsearch(Quadrature_Nodes,Tetrahedra_Midpoints,...
    'K',Number_of_Nearest_Neighbors_Volume);

% Initialize a vector to store the quadrature weights.
Quadrature_Weights=zeros(Number_of_Quadrature_Nodes,1);
%==========================================================================

%==========================================================================
%
% Generate the set of coefficients for the set of polynomial terms to
% include.
%
%==========================================================================
if Polynomial_Order_Volume==0
    Polynomial_Exponents_Volume=[0 0 0];
elseif Polynomial_Order_Volume>=1
    Polynomial_Exponents_Volume=[0 0 0;1 0 0;0 1 0;0 0 1];
    Old_Polynomial_Exponents_Volume=[1 0 0;0 1 0;0 0 1];
    for Order_Index=2:Polynomial_Order_Volume
        New_Polynomial_Exponents_Volume=[Old_Polynomial_Exponents_Volume(:,1)+1,Old_Polynomial_Exponents_Volume(:,2:3);
            Old_Polynomial_Exponents_Volume((end-Order_Index+1):end,1),Old_Polynomial_Exponents_Volume((end-Order_Index+1):end,2)+1,Old_Polynomial_Exponents_Volume((end-Order_Index+1):end,3);
            Old_Polynomial_Exponents_Volume(end,1:2),Old_Polynomial_Exponents_Volume(end,3)+1];
        Polynomial_Exponents_Volume=[Polynomial_Exponents_Volume;New_Polynomial_Exponents_Volume];
        Old_Polynomial_Exponents_Volume=New_Polynomial_Exponents_Volume;
    end
end
Number_of_Polynomial_Terms_Volume=(Polynomial_Order_Volume+1)*(Polynomial_Order_Volume+2)*(Polynomial_Order_Volume+3)/6;
%==========================================================================

% Define an array to store quadrature weights for all Tetrahedra (this is
% useful if the main loop is changed to a parfor loop)
Quadrature_Weights_Tetrahedra=zeros(Number_of_Nearest_Neighbors_Volume,Number_of_Tetrahedra);

% Loop over each tetrahedron.  Note that this loop can be easily changed to a
% for loop if the parallel toolbox is unavailable
parfor Current_Tetrahedron_Index=1:Number_of_Tetrahedra
    
    %==========================================================================
    % Define the vertices of the tetrahedron and the RBF centers nearest the
    % current Tetrahedron.
    %==========================================================================
    % Get the indices of the nearest neighbors to the current Tetrahedron
    % midpoint
    nni_Volume=Nearest_Neighbor_Indices_Tetrahedra(Current_Tetrahedron_Index,:);
    
    % We will shift the Tetrahedron by this value for numerical stability.
    M_Volume=ones(Number_of_Nearest_Neighbors_Volume,1)*Tetrahedra_Midpoints(Current_Tetrahedron_Index,:);%ones(Number_of_Nearest_Neighbors_Volume,1)*((ones(1,Number_of_Nearest_Neighbors_Volume)*Quadrature_Nodes(nni_Volume,:))/Number_of_Nearest_Neighbors_Volume);
    
    % The RBF centers make up the points x_{k,j} of the four tetrahedra.
    X_Volume=Quadrature_Nodes(nni_Volume,:)-M_Volume;
    
    A_Volume=ones(Number_of_Nearest_Neighbors_Volume,1)*Quadrature_Nodes(Tetrahedra(Current_Tetrahedron_Index,1),:)-M_Volume;
    B_Volume=ones(Number_of_Nearest_Neighbors_Volume,1)*Quadrature_Nodes(Tetrahedra(Current_Tetrahedron_Index,2),:)-M_Volume;
    C_Volume=ones(Number_of_Nearest_Neighbors_Volume,1)*Quadrature_Nodes(Tetrahedra(Current_Tetrahedron_Index,3),:)-M_Volume;
    D_Volume=ones(Number_of_Nearest_Neighbors_Volume,1)*Quadrature_Nodes(Tetrahedra(Current_Tetrahedron_Index,4),:)-M_Volume;
    %==========================================================================
    
    %==========================================================================
    % Set up the matrices containing the RBF centers and define the radial 
    % variable r.  This bit of code defines a matrix  r2 such that
    % r2(i,j)=(xi'-xj')^2+(yi'-yj')^2+(zi'-zj')^2;
    %==========================================================================
    r2_Volume = (X_Volume(:,1)-X_Volume(:,1).').^2+(X_Volume(:,2)-X_Volume(:,2).').^2+(X_Volume(:,3)-X_Volume(:,3).').^2;
    %==========================================================================
    
    %==========================================================================
    % Construct the matrix P and compute the volume integral of the polynomial 
    % terms over the Tetrahedron.
    %==========================================================================
    P_Volume=zeros(Number_of_Nearest_Neighbors_Volume,Number_of_Polynomial_Terms_Volume);
    
    for Term_Index=1:Number_of_Polynomial_Terms_Volume
        P_Volume(:,Term_Index)=(X_Volume(:,1).^Polynomial_Exponents_Volume(Term_Index,1)).*(X_Volume(:,2).^Polynomial_Exponents_Volume(Term_Index,2)).*(X_Volume(:,3).^Polynomial_Exponents_Volume(Term_Index,3));
    end
    
    xx= Legendre_Nodes_3D(:,1).*A_Volume(1,1)+(1-Legendre_Nodes_3D(:,1)).*(Legendre_Nodes_3D(:,2).*B_Volume(1,1)+(1-Legendre_Nodes_3D(:,2)).*(Legendre_Nodes_3D(:,3).*C_Volume(1,1)+(1-Legendre_Nodes_3D(:,3)).*D_Volume(1,1)));
    yy= Legendre_Nodes_3D(:,1).*A_Volume(1,2)+(1-Legendre_Nodes_3D(:,1)).*(Legendre_Nodes_3D(:,2).*B_Volume(1,2)+(1-Legendre_Nodes_3D(:,2)).*(Legendre_Nodes_3D(:,3).*C_Volume(1,2)+(1-Legendre_Nodes_3D(:,3)).*D_Volume(1,2)));
    zz= Legendre_Nodes_3D(:,1).*A_Volume(1,3)+(1-Legendre_Nodes_3D(:,1)).*(Legendre_Nodes_3D(:,2).*B_Volume(1,3)+(1-Legendre_Nodes_3D(:,2)).*(Legendre_Nodes_3D(:,3).*C_Volume(1,3)+(1-Legendre_Nodes_3D(:,3)).*D_Volume(1,3)));
    Change_of_Variable_Jacobian=abs((Legendre_Nodes_3D(:,2)-1).*(A_Volume(1,1).*B_Volume(1,2).*C_Volume(1,3)-A_Volume(1,1).*C_Volume(1,2).*B_Volume(1,3)-B_Volume(1,1).*A_Volume(1,2).*C_Volume(1,3)+B_Volume(1,1).*C_Volume(1,2).*A_Volume(1,3)+C_Volume(1,1).*A_Volume(1,2).*B_Volume(1,3)-C_Volume(1,1).*B_Volume(1,2).*A_Volume(1,3)-A_Volume(1,1).*B_Volume(1,2).*D_Volume(1,3)+A_Volume(1,1).*D_Volume(1,2).*B_Volume(1,3)+B_Volume(1,1).*A_Volume(1,2).*D_Volume(1,3)-B_Volume(1,1).*D_Volume(1,2).*A_Volume(1,3)-D_Volume(1,1).*A_Volume(1,2).*B_Volume(1,3)+D_Volume(1,1).*B_Volume(1,2).*A_Volume(1,3)+A_Volume(1,1).*C_Volume(1,2).*D_Volume(1,3)-A_Volume(1,1).*D_Volume(1,2).*C_Volume(1,3)-C_Volume(1,1).*A_Volume(1,2).*D_Volume(1,3)+C_Volume(1,1).*D_Volume(1,2).*A_Volume(1,3)+D_Volume(1,1).*A_Volume(1,2).*C_Volume(1,3)-D_Volume(1,1).*C_Volume(1,2).*A_Volume(1,3)-B_Volume(1,1).*C_Volume(1,2).*D_Volume(1,3)+B_Volume(1,1).*D_Volume(1,2).*C_Volume(1,3)+C_Volume(1,1).*B_Volume(1,2).*D_Volume(1,3)-C_Volume(1,1).*D_Volume(1,2).*B_Volume(1,3)-D_Volume(1,1).*B_Volume(1,2).*C_Volume(1,3)+D_Volume(1,1).*C_Volume(1,2).*B_Volume(1,3))).*(Legendre_Nodes_3D(:,1)-1).^2;
    Legendre_Weights_Tetrahedron=(Legendre_Weights_3D.*Change_of_Variable_Jacobian).';
    I_Polynomials=(Legendre_Weights_Tetrahedron*(bsxfun(@power,xx,Polynomial_Exponents_Volume(:,1).').*bsxfun(@power,yy,Polynomial_Exponents_Volume(:,2).').*bsxfun(@power,zz,Polynomial_Exponents_Volume(:,3).'))).';
    %==========================================================================
    
    %==========================================================================
    % Compute the volume integral of the RBF terms over the Tetrahedron.
    %==========================================================================
    I_Radial_Basis_Functions=Integral_of_RBF_Over_Tetrahedron(X_Volume,A_Volume,B_Volume,C_Volume,D_Volume,Legendre_Nodes_1D,Legendre_Weights_1D);
    %==========================================================================
    
    %==========================================================================
    % Compute the quadrature weights over the current tetrahedron.  This
    % solves the linear system A_k*w_k=I_k from the paper.
    %==========================================================================
    w=[(r2_Volume.^(3./2)),P_Volume;P_Volume.',zeros(((Polynomial_Order_Volume+1)*(Polynomial_Order_Volume+2)*(Polynomial_Order_Volume+3))/6)]\[I_Radial_Basis_Functions;I_Polynomials];
    %==========================================================================
    
    %==========================================================================
    % Pick out the quadrature weights from the solution vector w.
    %==========================================================================
    % Compute the distortion of the quadrature weights
    w=w(1:Number_of_Nearest_Neighbors_Volume,1);
    
    % Store the weights for this Tetrahedron
    Quadrature_Weights_Tetrahedra(:,Current_Tetrahedron_Index)=w;
    
    %==========================================================================
end

% Sum all quadrature weights corresponding to a given Quadrature node on
% the ball.  This implementation is useful when considering the use of
% parfor.
for Current_Tetrahedron_Index=1:Number_of_Tetrahedra
    nni=Nearest_Neighbor_Indices_Tetrahedra(Current_Tetrahedron_Index,:);
    Quadrature_Weights(nni,1)=Quadrature_Weights(nni,1)+Quadrature_Weights_Tetrahedra(:,Current_Tetrahedron_Index);
end

end


function I_RBF=Integral_of_RBF_Over_Tetrahedron(X,A,B,C,D,Legendre_Nodes,Legendre_Weights_1D)
%==========================================================================
% This subfunction implements the algorithm used to integrate the RBF
% phi(r)=r^3 over 4 tetrahedra, each with X as one of their vertices and 
% three of  A, B, C or D as the others.  This is described in
% the section "Integrals of RBFs Over Tetrahedra" in the paper.
%==========================================================================
nABC=vcross(B-A,C-A);
nABC=nABC./(vnorm(nABC)*ones(1,3));
nADB=vcross(D-A,B-A);
nADB=nADB./(vnorm(nADB)*ones(1,3));
nACD=vcross(C-A,D-A);
nACD=nACD./(vnorm(nACD)*ones(1,3));
nBDC=vcross(D-B,C-B);
nBDC=nBDC./(vnorm(nBDC)*ones(1,3));

%==========================================================================
% Find the orthogonal projections of O onto the sides of the
% tetrahedron.
%==========================================================================
E=X+((vdot((A-X),nABC)./vdot(nABC,nABC))*ones(1,3)).*nABC;
F=X+((vdot((A-X),nADB)./vdot(nADB,nADB))*ones(1,3)).*nADB;
G=X+((vdot((A-X),nACD)./vdot(nACD,nACD))*ones(1,3)).*nACD;
H=X+((vdot((B-X),nBDC)./vdot(nBDC,nBDC))*ones(1,3)).*nBDC;
%==========================================================================


I_OABC=Integral_of_RBF_Over_One_Tetrahedron(X,A,B,C,nABC,Legendre_Nodes,Legendre_Weights_1D);
I_OADB=Integral_of_RBF_Over_One_Tetrahedron(X,A,D,B,nADB,Legendre_Nodes,Legendre_Weights_1D);
I_OACD=Integral_of_RBF_Over_One_Tetrahedron(X,A,C,D,nACD,Legendre_Nodes,Legendre_Weights_1D);
I_OBDC=Integral_of_RBF_Over_One_Tetrahedron(X,B,D,C,nBDC,Legendre_Nodes,Legendre_Weights_1D);

I_RBF=(sign(vdot((X-E),nABC)).*(I_OABC)+...
    sign(vdot((X-F),nADB)).*(I_OADB)+...
    sign(vdot((X-G),nACD)).*(I_OACD)+...
    sign(vdot((X-H),nBDC)).*(I_OBDC));

end

function I=Integral_of_RBF_Over_One_Tetrahedron(X,A,B,C,nABC,Legendre_Nodes,Legendre_Weights)
%==========================================================================
% This subfunction implements the algorithm used to integrate the RBF
% phi(r)=r^3 over a tetrahedron with vertices X, A, B and C and normal
% vector nABC to the side with vertices A, B and C.  This is described in
% the section "Integrals of RBFs Over Tetrahedra" in the paper.
%==========================================================================
c1=vnorm(A-C).^2;
c2=2*vdot(A-C,B-C);
c3=vnorm(B-C).^2;
c4=2*vdot(A-C,C-X);
c5=2*vdot(B-C,C-X);
c6=vnorm(C-X).^2;
J=abs(vdot(A-X,vcross(B-X,C-X)));
Sigma_Plane=vdot(1/3*(A+B+C)-X,nABC)./vdot(C-X,nABC);
Sigma_Plane(isnan(Sigma_Plane) | isinf(Sigma_Plane))=0;

Integrand=@(lambda1) Sigma_Plane.^6./6.*J.*((1./(128.*c3.^(5./2))).*(2.*(c5+c2.*lambda1).*sqrt(c3.*(c6+lambda1.*(c4+c1.*lambda1))).*(3.*c5.^2+6.*c2.*c5.*lambda1+3.*c2.^2.*lambda1.^2-20.*c3.*(c6+lambda1.*(c4+c1.*lambda1)))-2.*(c5-2.*c3.*(-1+lambda1)+c2.*lambda1).*sqrt(c3.*(c3+c5+c6+(c2-2.*c3+c4-c5).*lambda1+(c1-c2+c3).*lambda1.^2)).*(-8.*c3.^2.*(-1+lambda1).^2+3.*(c5+c2.*lambda1).^2-4.*c3.*(5.*c6-2.*c5.*(-1+lambda1)+lambda1.*(2.*c2+5.*c4+5.*c1.*lambda1-2.*c2.*lambda1)))+3.*(c5.^2+2.*c2.*c5.*lambda1+c2.^2.*lambda1.^2-4.*c3.*(c6+lambda1.*(c4+c1.*lambda1))).^2.*log(c5-2.*c3.*(-1+lambda1)+c2.*lambda1+2.*sqrt(c3.*(c3+c5+c6+(c2-2.*c3+c4-c5).*lambda1+(c1-c2+c3).*lambda1.^2)))-3.*(c5.^2+2.*c2.*c5.*lambda1+c2.^2.*lambda1.^2-4.*c3.*(c6+lambda1.*(c4+c1.*lambda1))).^2.*log(c5+c2.*lambda1+2.*sqrt(c3.*(c6+lambda1.*(c4+c1.*lambda1))))));

I=real(Integrand(Legendre_Nodes.'))*Legendre_Weights;

I(J<10*eps)=0;

end

function v=vcross(v1,v2)
% This is a vectorized implementation of the cross product
v=[v1(:,2).*v2(:,3) - v1(:,3).*v2(:,2), v1(:,3).*v2(:,1) - v1(:,1).*v2(:,3), v1(:,1).*v2(:,2) - v1(:,2).*v2(:,1)];
end

function v=vdot(v1,v2)
% This is a vectorized implementation of the dot product.
v=v1(:,1).*v2(:,1)+v1(:,2).*v2(:,2)+v1(:,3).*v2(:,3);
end

function v=vnorm(v)
% This is a vectorized implementation of the 2-norm on R^3
v=sqrt(v(:,1).^2+v(:,2).^2+v(:,3).^2);
end

function [x,w]=Legendre_Gauss_Lobatto_Quadrature_Nodes(N,a,b)
% This function computes Legendre Gauss Lobatto Quadrature Nodes and
% Weights
N1=N+1;

x=cos(pi*(0:N)/N)';
P=zeros(N1,N1);
xold=2;
while max(abs(x-xold))>eps
    xold=x;
    P(:,1)=1;P(:,2)=x;
    for k=2:N
        P(:,k+1)=((2*k-1)*x.*P(:,k)-(k-1)*P(:,k-1))/k;
    end
    x=xold-(x.*P(:,N1)-P(:,N))./(N1*P(:,N1));
end
w=2./(N*N1*P(:,N1).^2);
x=(x.'+1)*(b-a)/2+a;
w=(b-a)./2*w;
end
