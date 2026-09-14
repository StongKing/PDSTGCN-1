"""Small inspection utility for the generated Divvy signal and prepared samples."""
import argparse, configparser, os, numpy as np

p=argparse.ArgumentParser(); p.add_argument('--config', default='configurations/DIVVY_astgcn.conf'); a=p.parse_args()
c=configparser.ConfigParser(); c.read(a.config)
f=c['Data']['graph_signal_matrix_filename']; z=np.load(f, allow_pickle=False)
print('raw data:', z['data'].shape)
base=os.path.splitext(os.path.basename(f))[0]
out=os.path.join(os.path.dirname(f), base+'_r1_d0_w0_astcgn.npz')
if os.path.exists(out):
    q=np.load(out, allow_pickle=False)
    for k in ['train_x','train_target','val_x','val_target','test_x','test_target']:
        print(k, q[k].shape)
else:
    print('prepared file not found; run prepareData.py first:', out)
