#!/usr/bin/env python
# coding: utf-8

from sympy import *
import sympy

m=var("m1 m2 m3")
l=var("l1 l2 l3")
th=var("th1 th2 th3")
thd=var("th1d th2d th3d")
a=var("a1 a2 a3")
alpha=var("alpha1 alpha2 alpha3")
d=var("d1 d2 d3")
R=var("R01 R12 R23")
P=var("P01 P12 P23 ")
g0=zeros(3,1);g0[-1]=-9.8

m1=0.5
m2=0.5
m3=0.5
l1=0.4
l2=0.4
l3=0.4
a1=pi/2.0
a2=0
a3=0
alpha1=0
alpha2=0.4
alpha3=0.4
d1=0.4
d2=0
d3=0


I1=Matrix([[(1/12)*m1*l1**2,0,0],[0,(1/12)*m1*l1**2,0],[0,0,0]])

I2=Matrix([[0,0,0],[0,(1/12)*m2*l2**2,0],[(1/12)*m2*l2**2,0,0],])

I3=Matrix([[0,0,0],[0,(1/12)*m3*l3**2,0],[(1/12)*m3*l3**2,0,0],])

def T(th,a,alpha,d):
    A=zeros(4,4)
    A[0,0]=cos(th)
    A[0,1]=-sin(th)*cos(a)
    A[0,2]=sin(th)*sin(a)
    A[0,3]=alpha*cos(th)
    A[1,0]=sin(th)
    A[1,1]=cos(th)*cos(a)
    A[1,2]=-cos(th)*sin(a)
    A[1,3]=alpha*sin(th)
    A[2,1]=sin(a)
    A[2,2]=cos(a)
    A[2,3]=d
    A[3,3]=1.0
    P=A[0:3,-1]
    R=A[0:3,0:3]
    return A, R, P

z0=zeros(3,1);z0[-1]=1
p0=zeros(4,1);p0[-1]=1


z1=T(th1,a1,alpha1,d1/2)[1]*z0
z2=T(th1,a1,alpha1,d1)[1]*T(th2,a2,alpha2/2,d2)[1]*z0
z3=T(th1,a1,alpha1,d1)[1]*T(th2,a2,alpha2,d2)[1]*T(th3,a3,alpha3/2,d3)[1]*z0

p1=T(th1,a1,alpha1,d1/2)[0]*p0
p21=T(th1,a1,alpha1,d1)[0]*p0
p22=T(th1,a1,alpha1,d1)[0]*T(th2,a2,alpha2/2,d2)[0]*p0

p31=T(th1,a1,alpha1,d1)[0]*p0
p32=T(th1,a1,alpha1,d1)[0]*T(th2,a2,alpha2,d2)[0]*p0
p33=T(th1,a1,alpha1,d1)[0]*T(th2,a2,alpha2,d2)[0]*T(th3,a3,alpha3/2,d3)[0]*p0
pe=T(th1,a1,alpha1,d1)[0]*T(th2,a2,alpha2,d2)[0]*T(th3,a3,alpha3,d3)[0]*p0

R1=T(th1,a1,alpha1,d1)[1]
R2=T(th1,a1,alpha1,d1)[1]*T(th2,a2,alpha2,d2)[1]
R3=T(th1,a1,alpha1,d1)[1]*T(th2,a2,alpha2,d2)[1]*T(th3,a3,alpha3,d3)[1]

JP1=z0.cross((p1)[0:3,:]).row_join(zeros(3,2))
JP2=z0.cross((p22)[0:3,:])
JP2=JP2.row_join(z1.cross((p22-p21)[0:3,:]).row_join(zeros(3,1)))

JP3=z0.cross((p33)[0:3,:])
JP3=JP3.row_join(z1.cross((p33-p31)[0:3,:]))
JP3=JP3.row_join(z2.cross((p33-p32)[0:3,:]))

JO1=z0.row_join(zeros(3,2))
JO2=z0
JO2=JO2.row_join(z1).row_join(zeros(3,1))
JO3=z0.row_join(z1).row_join(z2)

D=simplify(m1*JP1.T*JP1+m2*JP2.T*JP2+m3*JP3.T*JP3+JO1.T*R1*I1*R1.T*JO1+JO2.T*R2*I2*R2.T*JO2+JO3.T*R3*I3*R3.T*JO3)

c111=(1/2)*(diff(D[0,0],th1)+diff(D[0,0],th1)-diff(D[0,0],th1))
#cijk = bij + bik -bjk
c112=c121=(1/2)*(diff(D[0,0],th2)+diff(D[0,1],th1)-diff(D[0,1],th1))

c113=c131=(1/2)*(diff(D[0,0],th3)+diff(D[0,2],th1)-diff(D[0,2],th1))

c122=(1/2)*(diff(D[0,1],th2)+diff(D[0,1],th2)-diff(D[1,1],th1))

c123=c132=(1/2)*(diff(D[0,1],th3)+diff(D[0,2],th2)-diff(D[1,2],th1))

c133=(1/2)*(diff(D[0,2],th3)+diff(D[0,2],th3)-diff(D[2,2],th1))

c211=(1/2)*(diff(D[1,0],th1)+diff(D[1,0],th1)-diff(D[0,0],th2))
#cijk = bij + bik -bjk
c212=c221=(1/2)*(diff(D[1,0],th2)+diff(D[1,1],th1)-diff(D[0,1],th2))

c213=c231=(1/2)*(diff(D[1,0],th3)+diff(D[1,2],th1)-diff(D[0,2],th2))

c222=(1/2)*(diff(D[1,1],th2)+diff(D[1,1],th2)-diff(D[1,1],th2))

c223=c232=(1/2)*(diff(D[1,1],th3)+diff(D[1,2],th2)-diff(D[1,2],th2))

c233=(1/2)*(diff(D[1,2],th3)+diff(D[1,2],th3)-diff(D[2,2],th2))

c311=(1/2)*(diff(D[2,1],th1)+diff(D[2,0],th1)-diff(D[0,0],th3))

c312=c321=(1/2)*(diff(D[2,0],th2)+diff(D[2,1],th1)-diff(D[0,1],th3))

c313=c331=(1/2)*(diff(D[2,0],th3)+diff(D[2,2],th1)-diff(D[0,2],th3))

c322=(1/2)*(diff(D[2,1],th2)+diff(D[2,1],th2)-diff(D[1,1],th3))

c323=c332=(1/2)*(diff(D[2,1],th3)+diff(D[2,2],th2)-diff(D[1,2],th3))

c333=(1/2)*(diff(D[2,2],th3)+diff(D[2,2],th3)-diff(D[2,2],th3))

c11=c111*th1d+c112*th2d+c113*th3d
c12=c121*th1d+c122*th2d+c123*th3d
c13=c131*th1d+c132*th2d+c133*th3d
c21=c211*th1d+c212*th2d+c213*th3d
c22=c221*th1d+c222*th2d+c223*th3d
c23=c231*th1d+c232*th2d+c233*th3d
c31=c311*th1d+c312*th2d+c313*th3d
c32=c321*th1d+c322*th2d+c323*th3d
c33=c331*th1d+c332*th2d+c333*th3d


C=simplify(Matrix([[c11,c12,c13],[c21,c22,c23],[c31,c32,c33]]))

#G=-simplify(m1*JP1.T*g0+m2*JP2.T*g0+m3*JP3.T*g0)
#G=-simplify(m1*g0.T*JP1+m2*g0.T*JP2+m3*g0.T*JP3)
P=m1*g0.T*p1[0:3,0]+m2*g0.T*p22[0:3,0]+m3*g0.T*p33[0:3,0]
g1=-diff(P,th1)
g2=-diff(P,th2)
g3=-diff(P,th3)

G=simplify(Matrix([g1,g2,g3]))

D=lambdify(th,D,"numpy")
C=lambdify((th,thd),C,"numpy")
G=lambdify(th,G,"numpy")

