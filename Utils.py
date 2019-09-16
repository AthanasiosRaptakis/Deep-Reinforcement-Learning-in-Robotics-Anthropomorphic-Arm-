import numpy as np
import random
import torch

def Save_States(P_net,Pt_net,Q_net,Qt_net,P_optimizer,Q_optimizer,episode,batch_size,return_t,filename):


    state={
            'episode'         : episode,
            'P_state_dict'    : P_net.state_dict(),
            'Q_state_dict'    : Q_net.state_dict(),
            'Pt_state_dict'   : Pt_net.state_dict(),
            'Qt_state_dict'   : Qt_net.state_dict(),
            'P_optimizer'     : P_optimizer.state_dict(),
            'Q_optimizer'     : Q_optimizer.state_dict(),
            'batch_size'      : batch_size,
            'return'          :return_t
        }
    torch.save(state, filename)

class buffer():
    def __init__(self,):
        self.t=np.empty((0, ))
        self.y=np.empty((9,0 ))
        self.r=[]
    def update(self,sol,r):
        self.y = np.append(self.y,sol.y,axis=1)
        self.t = np.append(self.t,sol.t,axis=0)
        self.r.append(r)
        
class Replay_Buffer():
    def __init__(self,batch_size=62,max_size=1e6):
        self.state=[]
        self.next_state=[]
        self.r=[]
        self.d=[]
        self.u=[]
        
        self.Batch_Size=batch_size
        self.Max_size=max_size
        self.size=0
        
    def store(self,data):
        self.size=self.size+1
        self.state.append(data[0])
        self.u.append(data[1])
        self.r.append(data[2])
        self.d.append(data[3])
        self.next_state.append(data[4])
        if (self.size>self.Max_size):
            self.size=self.size-1
            self.state.pop(0)
            self.u.pop(0)
            self.r.pop(0)
            self.d.pop(0)
            self.next_state.pop(0)
        
    def get_batch(self):
        index=np.random.choice(self.size,self.Batch_Size)
        state=[ self.state[i] for i in index]
        u=[ self.u[i] for i in index]
        r=[ self.r[i] for i in index]
        d=[ self.d[i] for i in index]
        next_state=[ self.next_state[i] for i in index]
        
        return torch.FloatTensor(state),torch.FloatTensor(u),torch.FloatTensor(r).unsqueeze_(1),torch.FloatTensor(d).unsqueeze_(1),torch.FloatTensor(next_state)    
    
    


    