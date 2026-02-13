import torch
import torch.nn as nn
import torch.nn.functional as F

class NazariStylePolicy(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=128):  
        super().__init__()
        self.static_emb = nn.Linear(2, embed_dim)
        self.dyn_dem_emb = nn.Linear(1, embed_dim)
        self.dyn_load_emb = nn.Linear(1, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)

        self.Wq = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v  = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, depot_xy, locs_xy, remaining_dem, remaining_load, current_xy, mask):  # olasılık üretir
        B, N, _ = locs_xy.shape

        static_c = self.static_emb(locs_xy)
        dyn_c = self.dyn_dem_emb(remaining_dem[:, :, None])
        node_c = static_c + dyn_c

        depot_e = self.static_emb(depot_xy)[:, None, :]
        nodes = torch.cat([node_c, depot_e], dim=1)

        dec_in = self.static_emb(current_xy)
        dec_in += self.dyn_load_emb(remaining_load[:, None])

        out, _ = self.rnn(dec_in[:, None, :])
        h = out[:, 0, :]

        q = self.Wq(h)[:, None, :]
        k = self.Wk(nodes)
        u = torch.tanh(q + k)
        logits = self.v(u).squeeze(-1)
        logits = logits.masked_fill(~mask, float("-inf"))

        return F.softmax(logits, dim=-1)
