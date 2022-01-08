import dgl
import dgl.function as fn
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from ipt_cuda import generate_files,generate_data,render_with_materials
from tqdm import tqdm
import shutil,os,zipfile

def newdir(name,remove=True):

    if os.path.isdir(name) and remove:
        shutil.rmtree(name)
    os.mkdir(name)

def zipdir(name):
    
    zipf = zipfile.ZipFile(f'{name}.zip','w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(name):
        for file in files:
            zipf.write(os.path.join(root,file))
    zipf.close()

device = torch.device('cuda')
P_MIN = 1e-3

class GCN(nn.Module):

    def __init__(self, in_feats, out_feats):
        super(GCN, self).__init__()
        
        hidden = 100
        self.lift = nn.Linear(in_feats,hidden)
        self.network = nn.Sequential(
            MPL(hidden,hidden,F.relu),
            MPL(hidden,hidden,F.relu),
            MPL(hidden,hidden,F.relu)
        )
        self.out = nn.Linear(hidden,out_feats)
        
    def forward(self, g):
        g.ndata['node_feats'] = self.lift(g.ndata['node_feats']).tanh()
        g = self.network(g)
        g.ndata['node_feats'] = self.out(g.ndata['node_feats']).sigmoid()
        return g

    def loss(self, output, labels):
        preds = output.ndata['node_feats']
        return (preds - labels).abs().mean()

class MPL(nn.Module):

    def __init__(self, in_feats, out_feats, activation=None):
        super(MPL, self).__init__()
        self.linear = nn.Linear(in_feats*2,out_feats)
        self.activation = activation

    def forward(self, g):
        
        g.update_all(
            fn.src_mul_edge('node_feats','edge_feats','msg'),
            fn.sum('msg','reduced')
        )
        g.ndata['node_feats'] = self.linear(torch.cat((g.ndata['node_feats'],g.ndata['reduced']),dim=-1))
        if self.activation is not None: g.ndata['node_feats'] = self.activation(g.ndata['node_feats'])
        return g
        
def build_graph(w,pixel,light):
    w[w < P_MIN] = 0.
    w_sum = w.sum(axis=-1,keepdims=True)
    w = w / np.where(w_sum != 0, w_sum, np.ones_like(w_sum))
    w,w_eye = w[:-1],w[-1]
    
    pixel,pixel_eye = pixel[:-1],pixel[-1]

    dst,src = w.nonzero()
    dst,src = torch.tensor(dst),torch.tensor(src)

    g = dgl.graph((src,dst),num_nodes=len(w_eye))
    g.ndata['node_feats'] = torch.tensor(pixel_eye).float()
    g.edata['edge_feats'] = torch.tensor(w[w != 0]).float()
    g = dgl.add_self_loop(g)
    return g

def main():
    
    n = 100
    #generate_files(n)
    """
    data = []
    for i in tqdm(range(n)):
        scenefile = f'scenes/{i}.txt'
        imgfile = f'imgs/{i}.png'
    
        d = generate_data(scenefile,imgfile)
        data.append(d)
    torch.save(data,'data.pt')
    """
    data = torch.load('data.pt')[:1]

    graphs = []
    labels = []
    for w,pixel,light,lbls in data:
        #pixel[:,1:] = 0; lbls[:,1:] = 0
        graphs.append(build_graph(w,pixel,light))
        labels.append(torch.tensor(lbls))
    
    gcn = GCN(3,3).to(device)
    opt = torch.optim.Adam(gcn.parameters(),lr=1e-4)
    epochs = 100000
    step = 1000

    split = 1#int(len(graphs) * .9)
    for i in range(epochs):
        x = dgl.batch(graphs[:split]).to(device)
        y = torch.cat(labels[:split],dim=0).to(device)

        out = gcn(x)
        loss = gcn.loss(out,y)
        if (i + 1) % step == 0: print((i+1)//step,loss.detach().cpu())

        opt.zero_grad()
        loss.backward()
        opt.step()

    split = 0#int(len(graphs) * .8)
    newdir('preds')
    for i in range(len(graphs) - split):
        x = graphs[i].to(device)
        y = labels[i].to(device)
        out = gcn(x).ndata['node_feats'].detach().cpu()

        scenefile = f'scenes/{i}.txt'
        imgfile = f'preds/{i}_true.png'
        imgfile_pred = f'preds/{i}_pred.png'

        shutil.copy(f'imgs/{i+split}.png',imgfile)
        render_with_materials(scenefile,imgfile_pred,out)
    zipdir('preds')

if __name__ == "__main__":
    main()
    #generate_files(100)
