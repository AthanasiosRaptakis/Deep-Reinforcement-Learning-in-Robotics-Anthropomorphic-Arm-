#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as manimation
from scipy.integrate import solve_ivp
from Dynamics import D,C,G


class Robot():
    def __init__(self,DH_params,q0):
        self.DH_params=DH_params
        self.n=DH_params.shape[0]
        self.q=q0
        self.O=np.zeros([self.n+1,3])
        self.goal=np.array([[0.3],[0.3],[0.3]])
        self.x0=np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.])
        self.acc=np.array([0.,0.,0.])
        self.step_size=0.01
        self.ts=0.
        self.tf=self.step_size
        

        
    def T(self,th,a,alpha,d):
        A=np.zeros([4,4])
        A[0,0]=np.cos(th)
        A[0,1]=-np.sin(th)*np.cos(a)
        A[0,2]=np.sin(th)*np.sin(a)
        A[0,3]=alpha*np.cos(th)
        A[1,0]=np.sin(th)
        A[1,1]=np.cos(th)*np.cos(a)
        A[1,2]=-np.cos(th)*np.sin(a)
        A[1,3]=alpha*np.sin(th)
        A[2,1]=np.sin(a)
        A[2,2]=np.cos(a)
        A[2,3]=d
        A[3,3]=1.0
        return A
    
    def kinematics(self,):
        temp=np.eye(4)
        for i in range(self.n):
            th=self.q[i]
            a=self.DH_params[i,0]
            alpha=self.DH_params[i,1]
            d=self.DH_params[i,2]
            o=np.dot(temp,self.T(th,a,alpha,d))
            temp=o
            self.O[i+1]=o[0:3,3]  
    
    def end_effector(self,):
        return self.O[-1]  
    
    
    def plot(self,):

        get_ipython().run_line_magic('matplotlib', 'inline')
        fig=plt.figure(figsize=(10,5))
        ax1 = fig.add_subplot(121, projection='3d')
       
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')
        
        ax1.set_xlim([-1.4,1.4])
        ax1.set_ylim([-1.4,1.4]) 
        ax1.set_zlim([-1,1]) 
        
        plt.plot(self.O[:,0],self.O[:,1],self.O[:,2])
        plt.plot(self.O[:,0],self.O[:,1],self.O[:,2],'ro')
        plt.plot([self.goal[0]],[self.goal[1]],[self.goal[2]],'*g')
        ax2 = fig.add_subplot(122, projection='3d')
       
        ax2.view_init(azim=45)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('z')
        
        ax2.set_xlim([-1.4,1.4])
        ax2.set_ylim([-1.4,1.4]) 
        ax2.set_zlim([-1,1]) 
        
        plt.plot(self.O[:,0],self.O[:,1],self.O[:,2])
        plt.plot(self.O[:,0],self.O[:,1],self.O[:,2],'ro')
        plt.plot([self.goal[0]],[self.goal[1]],[self.goal[2]],'*g')

        plt.show()

    def show(self,sol):
        
        fig=plt.figure(figsize=(5,5))
              
        self.q[0]=sol.y[3,0]
        self.q[1]=sol.y[4,0]
        self.q[2]=sol.y[5,0]
        self.kinematics()

        Q=np.array([self.end_effector()])
        for i in range(sol.t.shape[0]):

            self.q[0]=sol.y[3,i]
            self.q[1]=sol.y[4,i]
            self.q[2]=sol.y[5,i]
            self.kinematics()
            Q=np.append(Q,[self.end_effector()],0)

            ax=plt.axes(projection="3d")

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

            ax.set_xlim([-1.4,1.4])
            ax.set_ylim([-1.4,1.4]) 
            ax.set_zlim([-1,1]) 


            plt.plot(self.O[:,0],self.O[:,1],self.O[:,2])
            plt.plot(self.O[0:3,0],self.O[0:3,1],self.O[0:3,2],'o')
            plt.plot([self.O[3,0]],[self.O[3,1]],[self.O[3,2]],'ro')
            plt.plot(Q[:,0],Q[:,1],Q[:,2],'--',linewidth=1)
            plt.plot([self.goal[0]],[self.goal[1]],[self.goal[2]],'go')
        plt.show()
                
        
    def Distance(self,):
        return np.sqrt( np.sum((self.end_effector()-self.goal)**2))
    
    def Get_Reward(self,):
        r=-100
        d=0.0
        dist = np.sqrt( np.sum((self.end_effector()-self.goal)**2))
        
        velocity=np.max(np.abs(self.x0[0:3]))
        #accl=np.linalg.norm(self.acc,'inf')
        
        c1=1000
        c2=100
        u=np.array([self.x0[6],self.x0[7],self.x0[8]])
        
        reward=c1*np.min([1.0,(1.0- self.Distance()/0.8)])-c2*np.linalg.norm(u)
        if (velocity<0.05):
            r=-5000
            d=1
        #if (dist<0.1):
        #    r=5000
            
        if (dist<0.05):
            r=100000000
            d=1
            
        #return 1000*np.min([1.0,(1.0- self.Distance()/0.8)])+r,d
        #return 2*np.min([1.0,(1.0- dist**(0.4))])+r,d
        #return r,d
        return reward+r,d
    
    def get_input(self,u):
        self.x0[6]=u[0]
        self.x0[7]=u[1]
        self.x0[8]=u[2]
        
        
    def rhs(self,t,x):
        T=np.array([x[6],x[7],x[8]])
#        T=T-0.1*np.array([x[0],x[1],x[2]])
        DD=D(x[3],x[4],x[5])
        Dinv=np.linalg.inv(DD)
        CC=C((x[3],x[4],x[5]),(x[0],x[1],x[2]))
        GG=G(x[3],x[4],x[5])
        qd=np.array([x[0],x[1],x[2]])

        a1=-np.dot(Dinv,np.dot(CC,qd))

        a2=-np.dot(Dinv,GG).reshape(3,)

        a3=np.dot(Dinv,T)

        qdd=a1+a2+a3
        qdd= np.clip(qdd, -6.0*np.pi, 6.0*np.pi)        
        self.acc=qdd
        
        return np.append(np.append(qdd, qd),np.array([0,0,0]))
    
    def step(self,):
        
        sol=self.simulate()
        self.ts=self.tf
        self.tf=self.tf+self.step_size
        r,d=self.Get_Reward()
        
        return sol,r,d
        

    def simulate(self,):
        N=3
        teval=np.linspace(self.ts, self.tf, N, endpoint=True)
        sol = solve_ivp(self.rhs, (self.ts,self.tf),self.x0,'RK45',teval)
       
        i11= sol.y[3] <=-2.0*np.pi
        i12= sol.y[3] >= 2.0*np.pi
        i21= sol.y[4] <=-np.pi/3.0
        i22= sol.y[4] >= 4.0*(np.pi/3.0)
        i31= sol.y[5] <=-5.0*(np.pi/6.0)
        i32= sol.y[5] >= 5.0*(np.pi/6.0)
        
        sol.y[0,i11]=0.0
        sol.y[0,i12]=0.0
        sol.y[1,i21]=0.0
        sol.y[1,i22]=0.0
        sol.y[2,i31]=0.0
        sol.y[2,i32]=0.0
        
        sol.y[3,:]= np.clip(sol.y[3,:], -2.0*np.pi, 2.0*np.pi)
        sol.y[4,:]= np.clip(sol.y[4,:], -np.pi/3.0, 4.0*(np.pi/3.0))
        sol.y[5,:]= np.clip(sol.y[5,:], -5.0*(np.pi/6.0), 5.0*(np.pi/6.0))
        
        sol.y[0,:]= np.clip(sol.y[0,:], -2.0*np.pi, 2.0*np.pi)
        sol.y[1,:]= np.clip(sol.y[1,:], -2.0*np.pi, 2.0*np.pi)
        sol.y[2,:]= np.clip(sol.y[2,:], -2.0*np.pi, 2.0*np.pi)
        
        
        self.x0=sol.y[:,-1]
        
        self.q[0]=self.x0[3]
        self.q[1]=self.x0[4]
        self.q[2]=self.x0[5]
        self.kinematics()
        return sol
    
    def plot_sol(self,sol):
        get_ipython().run_line_magic('matplotlib', 'inline')

        plt.figure(figsize=(15,7))
        plt.subplot(2, 2, 1)
        plt.title(r"$q(t)$")
        plt.plot(sol.t,sol.y[3,:].T,'r')
        plt.plot(sol.t,sol.y[4,:].T,'g')
        plt.plot(sol.t,sol.y[5,:].T,'b')
        #plt.xlabel(r"t")
        plt.legend([r"$q_0(t)$",r"$q_1(t)$",r"$q_2(t)$"])
        
        plt.subplot(2, 2, 2)
        plt.title(r"$\dot q(t)$")
        plt.plot(sol.t,sol.y[0,:].T,'r')
        plt.plot(sol.t,sol.y[1,:].T,'g')
        plt.plot(sol.t,sol.y[2,:].T,'b')
        #plt.xlabel(r"t")
        plt.legend([r"$\.q_0(t)$",r"$\.q_1(t)$",r"$\.q_2(t)$"])

        plt.subplot(2, 2, 3)
        plt.title(r"$u(t)$")
        plt.plot(sol.t,sol.y[6,:].T,'r')
        plt.plot(sol.t,sol.y[7,:].T,'g')
        plt.plot(sol.t,sol.y[8,:].T,'b')
        plt.xlabel(r"t")
        plt.legend([r"$u_0(t)$",r"$u_1(t)$",r"$u_2(t)$"])

        plt.subplot(2, 2, 4)
        plt.title(r"Reward - $r_i(t)$")
        plt.plot(sol.r,"-*")
        plt.xlabel(r"$t$")
        plt.show()
        plt.close()

    def show(self,sol):
        
        fig=plt.figure(figsize=(5,5))
              
        self.q[0]=sol.y[3,0]
        self.q[1]=sol.y[4,0]
        self.q[2]=sol.y[5,0]
        self.kinematics()

        Q=np.array([self.end_effector()])
        for i in range(sol.t.shape[0]):

            self.q[0]=sol.y[3,i]
            self.q[1]=sol.y[4,i]
            self.q[2]=sol.y[5,i]
            self.kinematics()
            Q=np.append(Q,[self.end_effector()],0)

            ax=plt.axes(projection="3d")

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

            ax.set_xlim([-1.4,1.4])
            ax.set_ylim([-1.4,1.4]) 
            ax.set_zlim([-1,1]) 


            plt.plot(self.O[:,0],self.O[:,1],self.O[:,2])
            plt.plot(self.O[:,0],self.O[:,1],self.O[:,2],'ro')
            plt.plot(Q[:,0],Q[:,1],Q[:,2],'--',linewidth=1)
            plt.plot([self.goal[0]],[self.goal[1]],[self.goal[2]],'go')
        plt.show()
    
    def animate(self,sol,filename):
        #FFMpegWriter = manimation.writers['ffmpeg']
        #writer = manimation.FFMpegWriter(fps=15, codec='hevc')

        metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        writer = manimation.FFMpegWriter(fps=30,metadata=metadata, codec='libx264')

        #writer = FFMpegWriter(fps=15, metadata=metadata)

#        get_ipython().run_line_magic('matplotlib', 'inline')

        fig=plt.figure(figsize=(5,5))

        self.q[0]=sol.y[3,0]
        self.q[1]=sol.y[4,0]
        self.q[2]=sol.y[5,0]
        self.kinematics()

        Q=np.array([self.end_effector()])
        with writer.saving(fig, filename, sol.t.shape[0]):
            for i in range(sol.t.shape[0]):

                self.q[0]=sol.y[3,i]
                self.q[1]=sol.y[4,i]
                self.q[2]=sol.y[5,i]
                self.kinematics()
                Q=np.append(Q,[self.end_effector()],0)

                ax=plt.axes(projection="3d")

                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')

                ax.set_xlim([-1.4,1.4])
                ax.set_ylim([-1.4,1.4]) 
                ax.set_zlim([-1,1]) 

                plt.plot(self.O[:,0],self.O[:,1],self.O[:,2])
                plt.plot(self.O[:,0],self.O[:,1],self.O[:,2],'ro')
                plt.plot(Q[:,0],Q[:,1],Q[:,2],'--',linewidth=1)
                plt.plot([self.goal[0]],[self.goal[1]],[self.goal[2]],'go')

                writer.grab_frame()
        plt.close()
