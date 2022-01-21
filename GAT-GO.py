import os
import torch
import argparse
from src import GnnPF, data_loader
from torch.utils import data as D
from torch_geometric.data import DataLoader
current_file_path = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = current_file_path + '/data/'

def predict(model, loader, device):
    torch.cuda.set_device(device)
    results = dict()
    for data in loader:
        with torch.cuda.amp.autocast():
            esm_rep, seq, contact, pssm, seq_embed = data.x.T.unsqueeze(0).cuda(), data.seq.T.unsqueeze(0).cuda(), data.edge_index.cuda(), data.pssm.T.unsqueeze(0).cuda(), data.seq_embed.cuda()
            label = data.label
            batch_idx = data.batch.cuda()
            model_pred = torch.sigmoid(model(esm_rep=esm_rep, seq = seq, pssm = pssm, seq_embed=seq_embed, A = contact, batch = batch_idx)).cpu().detach().numpy()
        for i, chain_id in enumerate(data.chain_id):
            results[chain_id] = model_pred[i,:]
        break
    return results

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Predicting Protein function with GAT-GO')
    parser.add_argument('--ModelPath', help='Model to be used for inference', type=str)
    parser.add_argument('--Device', help='CUDA device for inference', type = int, default=0)
    parser.add_argument('--BatchSize', help='Batch size for inference', type = str, default=4)
    parser.add_argument('--SeqIDs', help='Input seq file for inference', type=str)
    parser.add_argument('--OutDir', help='Output Directory to store result', type=str)
    args = parser.parse_args()

    if not os.path.isfile(args.SeqIDs):
        print('Error:Input file does not exist')
        exit(1)
    if not os.path.isfile(args.ModelPath):
        print('Error: Model file does not exist')
    
    elif os.path.isdir(os.path.join(args.OutDir, '')):
        pass
    else:
        cmd = 'mkdir -p ' + os.path.join(args.OurDir, '')
        os.system(cmd)

    Dset = data_loader.Protein_Gnn_data(root = DATA_PATH + '/seq_features/', chain_list = args.SeqIDs)
    loader = DataLoader(Dset,  batch_size=args.BatchSize)
    device = torch.device('cpu')
    torch.cuda.set_device(args.Device)
    check_point = torch.load(args.ModelPath, map_location = device)
    model = GnnPF.GnnPF().cuda()
    model.load_state_dict(check_point['state_dict'])
    model.eval().cuda()

    results = predict(model, loader, args.Device)
    torch.save(results, args.OutDir + 'GAT-GO_Results.pt')
