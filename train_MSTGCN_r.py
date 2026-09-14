"""Optional static MSTGCN baseline on the same Divvy 6->6 samples."""
import argparse, configparser, os, shutil
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from lib.utils import load_graphdata_channel1, get_adjacency_matrix
from model.MSTGCN_r import make_model

p=argparse.ArgumentParser(); p.add_argument('--config',default='configurations/DIVVY_astgcn.conf'); a=p.parse_args()
c=configparser.ConfigParser(); c.read(a.config); d=c['Data']; t=c['Training']
os.environ['CUDA_VISIBLE_DEVICES']=t.get('ctx','0'); device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
load=load_graphdata_channel1(d['graph_signal_matrix_filename'],d['dynamic_graph_filename'],int(t['num_of_hours']),int(t['num_of_days']),int(t['num_of_weeks']),device,int(t['batch_size']),int(t['in_channels']),True)
train_loader,_,val_loader,_,test_loader,_,_,_,_=load
A,_=get_adjacency_matrix(d['adj_filename'],int(d['num_of_vertices']),d.get('id_filename',fallback=None))
net=make_model(device,int(t['nb_block']),int(t['in_channels']),int(t['K']),int(t['nb_chev_filter']),int(t['nb_time_filter']),int(t['num_of_hours']),A,int(d['num_for_predict']),int(d['len_input']))
opt=optim.Adam(net.parameters(),lr=float(t['learning_rate'])); crit=nn.MSELoss(); path=Path('experiments')/'DIVVY'/'mstgcn_static'; path.mkdir(parents=True,exist_ok=True)
best=np.inf
for e in range(int(t['epochs'])):
    net.eval(); vl=[]
    with torch.no_grad():
        for x,y,_ in val_loader: vl.append(crit(net(x.to(device)),y.to(device)).item())
    v=float(np.mean(vl));
    if v<best: best=v; torch.save(net.state_dict(),path/'best.params')
    net.train(); tl=[]
    for x,y,_ in train_loader:
        x=x.to(device); y=y.to(device); opt.zero_grad(); o=net(x); loss=crit(o,y); loss.backward(); opt.step(); tl.append(loss.item())
    print(e,'train',float(np.mean(tl)),'val',v,'best',best)
