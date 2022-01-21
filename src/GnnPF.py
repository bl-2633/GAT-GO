from torch_geometric.nn import GATConv,global_mean_pool,SAGPooling,global_max_pool, global_add_pool,EdgePooling
from torch_geometric.nn import global_mean_pool
from torch import nn
import torch




class GnnPF(torch.nn.Module):
    def __init__(self, seq = True, pssm = True, esm=True, embed =True):
        super(GnnPF, self).__init__()
        
        embed_channels = 512
        hidden_channels = [512, 512, 1024, 1024]
        self.block_config = [2,2,2,2]
        self.planes = [128, 256, 512, 1024]
        self.current_plane = 64
        self.conv_kernel = 5
        self.H1 = 64
        self.use_pssm = pssm
        self.use_esm = esm
        self.use_embed = embed
        self.use_seq = seq

        self.esm_in = nn.Sequential(
            nn.Conv1d(1280, embed_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        self.seq_in = nn.Sequential(
            nn.Conv1d(25, embed_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.pssm_in = nn.Sequential(
            nn.Conv1d(20, embed_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        self.full1d_in = nn.Sequential(
            nn.Conv1d(51, self.H1, kernel_size = 1),
            nn.ReLU(inplace=True),
            nn.InstanceNorm1d(self.H1)
        )

        if self.use_embed:
            self.rep_dim = hidden_channels[-1] + 1280
        else:
            self.rep_dim = hidden_channels[-1]

        self.classifier = nn.Sequential(
            nn.Linear(self.rep_dim, 2752),
            nn.ReLU(inplace=True),
            nn.Linear(2752, 2752)
        )

        self.gc1 = GATConv(embed_channels, hidden_channels[0], heads = 12, dropout = 0.5, bias=False, concat = False)
        self.gc2 = GATConv(hidden_channels[0], hidden_channels[1], heads = 12, dropout = 0.5, bias = False, concat = False)
        self.gc3 = GATConv(hidden_channels[1], hidden_channels[2], heads = 12, dropout = 0.5, bias = False, concat = False)
        self.gc4 = GATConv(hidden_channels[2], hidden_channels[3], heads = 12, dropout = 0.5, bias = False, concat = False)
        self.gp1 = SAGPooling(in_channels=hidden_channels[0])
        self.gp2 = SAGPooling(in_channels=hidden_channels[1])
        self.gp3 = SAGPooling(in_channels=hidden_channels[2])
        self.gp4 = SAGPooling(in_channels=hidden_channels[3])
        
    def forward(self, esm_rep, seq, pssm, A, seq_embed, batch):
        #Graph embedding section
        graph_embeddings = []
        input_feats = []
        esm = self.esm_in(esm_rep)
        seq = self.seq_in(seq)
        pssm = self.pssm_in(pssm)
        if self.use_seq:
            input_feats.append(seq)
        if self.use_pssm:
            input_feats.append(pssm)    
        if self.use_esm:
            input_feats.append(esm)
        if input_feats == []:
            print('Input Feature not specified')
            exit()
        
        embed = input_feats[0]
        if len(embed) > 1:
            for input_feat in input_feats[1:]:
                embed += input_feat

        embed = embed.T.squeeze(2)
        embed.relu()

        out = self.gc1(embed, A)
        out.relu()
        out, A, _, batch, perm, score = self.gp1(
            out, A, None, batch)


        out = self.gc2(out, A)
        out.relu()        
        out, A, _, batch, perm, score = self.gp2(
            out, A, None, batch)

        out = self.gc3(out, A)
        out.relu()        
        out, A, _, batch, perm, score = self.gp3(
            out, A, None, batch)
      
        out = self.gc4(out, A)
        out.relu()        
        out, A, _, batch, perm, score = self.gp4(
            out, A, None, batch)

        out = global_mean_pool(out, batch)

        if self.use_embed:
            feat = torch.cat([out, seq_embed], dim = 1)
        else:
            feat = out
        #function predictor
        out = self.classifier(feat)
                
        return out