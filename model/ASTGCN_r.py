from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import scaled_Laplacian, cheb_polynomial, first_order_graph


class Spatial_Attention_layer(nn.Module):
    def __init__(self, device, in_channels, num_of_vertices, num_of_timesteps):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(num_of_timesteps, device=device))
        self.W2 = nn.Parameter(torch.empty(in_channels, num_of_timesteps, device=device))
        self.W3 = nn.Parameter(torch.empty(in_channels, device=device))
        self.bs = nn.Parameter(torch.empty(1, num_of_vertices, num_of_vertices, device=device))
        self.Vs = nn.Parameter(torch.empty(num_of_vertices, num_of_vertices, device=device))

    def forward(self, x):
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)                 # B,N,T
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)                     # B,T,N
        product = torch.matmul(lhs, rhs)                                      # B,N,N
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))
        return F.softmax(S, dim=1)


class Temporal_Attention_layer(nn.Module):
    def __init__(self, device, in_channels, num_of_vertices, num_of_timesteps):
        super().__init__()
        self.U1 = nn.Parameter(torch.empty(num_of_vertices, device=device))
        self.U2 = nn.Parameter(torch.empty(in_channels, num_of_vertices, device=device))
        self.U3 = nn.Parameter(torch.empty(in_channels, device=device))
        self.be = nn.Parameter(torch.empty(1, num_of_timesteps, num_of_timesteps, device=device))
        self.Ve = nn.Parameter(torch.empty(num_of_timesteps, num_of_timesteps, device=device))

    def forward(self, x):
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)  # B,T,N
        rhs = torch.matmul(self.U3, x)                                             # B,N,T
        E = torch.matmul(self.Ve, torch.sigmoid(torch.matmul(lhs, rhs) + self.be))
        return F.softmax(E, dim=1)


class cheb_conv_withSAt(nn.Module):
    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        super().__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.register_buffer(
            "cheb_stack",
            torch.stack([p.float() for p in cheb_polynomials], dim=0),
        )
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.empty(in_channels, out_channels)) for _ in range(K)]
        )

    def forward(self, x, spatial_attention):
        B, N, _, T = x.shape
        outputs = []
        for t in range(T):
            graph_signal = x[:, :, :, t]
            out = torch.zeros(B, N, self.out_channels, device=x.device, dtype=x.dtype)
            for k in range(self.K):
                Tk = self.cheb_stack[k].to(x.device)
                Tk_att = Tk.unsqueeze(0) * spatial_attention
                rhs = Tk_att.transpose(1, 2).matmul(graph_signal)
                out = out + rhs.matmul(self.Theta[k])
            outputs.append(out.unsqueeze(-1))
        return F.relu(torch.cat(outputs, dim=-1))


class dynamic_graph_conv(nn.Module):
    """Forward/backward diffusion on the predefined sample-level dynamic graph.

    The original repository explicitly formed P^k.  For Divvy (~568 nodes),
    that performs unnecessary N^3 matrix-matrix products.  Here the exactly
    equivalent P^k X recurrence is used, reducing the dominant operation to
    N^2 F and preserving the K-order diffusion semantics.
    """

    def __init__(self, K, in_channels, out_channels):
        super().__init__()
        self.K = K
        self.W_k1 = nn.ParameterList(
            [nn.Parameter(torch.empty(in_channels, out_channels)) for _ in range(K)]
        )
        self.W_k2 = nn.ParameterList(
            [nn.Parameter(torch.empty(in_channels, out_channels)) for _ in range(K)]
        )
        self.time_transform = nn.Linear(2, 1)

    def forward(self, x, P_f, P_b, time_of_day, day_of_week):
        outputs = []
        B, N, _, T = x.shape
        for t in range(T):
            X0 = x[:, :, :, t]
            time_feat = torch.stack([time_of_day[:, :, t], day_of_week[:, :, t]], dim=-1)
            scale = self.time_transform(time_feat)                             # B,N,1

            Hf = X0
            Hb = X0
            out = torch.zeros(B, N, self.W_k1[0].shape[1], device=x.device, dtype=x.dtype)
            for k in range(self.K):
                out = out + (Hf * scale).matmul(self.W_k1[k])
                out = out + (Hb * scale).matmul(self.W_k2[k])
                if k + 1 < self.K:
                    Hf = P_f.matmul(Hf)
                    Hb = P_b.matmul(Hb)
            outputs.append(out.unsqueeze(-1))
        return F.relu(torch.cat(outputs, dim=-1))


class DynamicGraphConvWithFlow(nn.Module):
    """Data-generated dynamic aggregation branch from the current PDST-GCN idea."""

    def __init__(self, in_channels, out_channels, embedding_dim=10):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.theta = nn.Parameter(torch.empty(in_channels, embedding_dim))
        self.W = nn.Parameter(torch.empty(in_channels, out_channels))
        self.alpha = nn.Parameter(torch.tensor([1.0]))
        # Current repository used two learned source/target embedding types and
        # broadcast them over nodes.  Parameterizing them directly is equivalent
        # but avoids an artificial Embedding(in_channels, ... ) size dependency.
        self.source_embedding = nn.Parameter(torch.empty(embedding_dim))
        self.target_embedding = nn.Parameter(torch.empty(embedding_dim))
        self.time_transform = nn.Linear(2, 1)

    def forward(self, x, F_t, L_f, time_of_day, day_of_week):
        B, N, _, T = x.shape
        outputs = []
        L = L_f.to(x.device)
        eye = torch.eye(N, device=x.device, dtype=x.dtype).unsqueeze(0)

        for t in range(T):
            Ft = F_t[:, :, :, t]
            DF = torch.matmul(L.unsqueeze(0), Ft).matmul(self.theta)           # B,N,d
            E1 = self.source_embedding.view(1, 1, -1)
            E2 = self.target_embedding.view(1, 1, -1)
            DE1 = torch.tanh(self.alpha * (DF * E1))
            DE2 = torch.tanh(self.alpha * (DF * E2))
            A = F.softmax(F.relu(DE1.matmul(DE2.transpose(1, 2))), dim=-1)
            A = A + eye
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)

            time_feat = torch.stack([time_of_day[:, :, t], day_of_week[:, :, t]], dim=-1)
            scale = self.time_transform(time_feat)                             # B,N,1
            Z = (A.matmul(x[:, :, :, t]) * scale).matmul(self.W)
            outputs.append(Z.unsqueeze(-1))
        return F.relu(torch.cat(outputs, dim=-1))


class ASTGCN_block(nn.Module):
    def __init__(
        self,
        device,
        in_channels,
        K,
        nb_chev_filter,
        nb_time_filter,
        time_strides,
        cheb_polynomials,
        num_of_vertices,
        num_of_timesteps,
    ):
        super().__init__()
        self.TAt = Temporal_Attention_layer(device, in_channels, num_of_vertices, num_of_timesteps)
        self.SAt = Spatial_Attention_layer(device, in_channels, num_of_vertices, num_of_timesteps)
        self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, in_channels, nb_chev_filter)
        self.dynamic_gcn = dynamic_graph_conv(K, in_channels, nb_chev_filter)
        self.dynamic_gcn2 = DynamicGraphConvWithFlow(in_channels, nb_chev_filter)

        self.gate1 = nn.Sequential(nn.Conv2d(nb_chev_filter, nb_chev_filter, 1), nn.Sigmoid())
        self.gate2 = nn.Sequential(nn.Conv2d(nb_chev_filter, nb_chev_filter, 1), nn.Sigmoid())
        self.gate3 = nn.Sequential(nn.Conv2d(nb_chev_filter, nb_chev_filter, 1), nn.Sigmoid())

        self.time_conv = nn.Conv2d(
            nb_chev_filter * 3,
            nb_time_filter,
            kernel_size=(1, 3),
            stride=(1, time_strides),
            padding=(0, 1),
        )
        self.residual_conv = nn.Conv2d(
            in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides)
        )
        self.ln = nn.LayerNorm(nb_time_filter)

    def forward(self, x, P_f, P_b, F_t, L_f, time_of_day, day_of_week):
        B, N, Fin, T = x.shape

        temporal_At = self.TAt(x)
        x_TAt = torch.matmul(x.reshape(B, -1, T), temporal_At).reshape(B, N, Fin, T)
        spatial_At = self.SAt(x_TAt)

        g1 = self.cheb_conv_SAt(x, spatial_At)
        g2 = self.dynamic_gcn(x, P_f, P_b, time_of_day, day_of_week)
        g3 = self.dynamic_gcn2(x, x, L_f, time_of_day, day_of_week)

        def gate(g, layer):
            w = layer(g.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            return g * w

        spatial = torch.cat([gate(g1, self.gate1), gate(g2, self.gate2), gate(g3, self.gate3)], dim=2)
        time_conv_output = self.time_conv(spatial.permute(0, 2, 1, 3))
        residual = self.residual_conv(x.permute(0, 2, 1, 3))
        x = F.relu(residual + time_conv_output)
        x = self.ln(x.permute(0, 2, 3, 1)).permute(0, 1, 3, 2)
        return x


class IntelligentAdjustment(nn.Module):
    """Current repository's learnable fleet-total correction, made configurable."""

    def __init__(self, num_of_vertices, num_timesteps, target_sum, embedding_dim=16):
        super().__init__()
        self.target_sum = float(target_sum)
        self.node_embeddings = nn.Embedding(num_of_vertices, embedding_dim)
        self.dynamic_weight_net = nn.Sequential(
            nn.Linear(num_timesteps + embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_timesteps),
            nn.Sigmoid(),
        )
        self.residual_scaling = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, apply_rounding=None):
        if apply_rounding is None:
            apply_rounding = not self.training
        B, N, T = x.shape
        residual = self.target_sum - x.sum(dim=1)                              # B,T
        node_features = self.node_embeddings.weight.unsqueeze(0).expand(B, -1, -1)
        r = residual.unsqueeze(1).expand(-1, N, -1)
        weights = self.dynamic_weight_net(torch.cat([r, node_features], dim=-1))
        x = x + self.residual_scaling * r * weights
        # Inventory cannot be negative.
        x = F.relu(x)
        # Keep the current project's straight-through integerization semantics.
        xr = torch.round(x)
        x = x + (xr - x).detach()
        return x


class ASTGCN_submodule(nn.Module):
    def __init__(
        self,
        device,
        nb_block,
        in_channels,
        K,
        nb_chev_filter,
        nb_time_filter,
        time_strides,
        cheb_polynomials,
        num_for_predict,
        len_input,
        num_of_vertices,
        L_f,
        fleet_size,
    ):
        super().__init__()
        self.BlockList = nn.ModuleList()
        self.BlockList.append(
            ASTGCN_block(
                device,
                in_channels,
                K,
                nb_chev_filter,
                nb_time_filter,
                time_strides,
                cheb_polynomials,
                num_of_vertices,
                len_input,
            )
        )
        for _ in range(nb_block - 1):
            self.BlockList.append(
                ASTGCN_block(
                    device,
                    nb_time_filter,
                    K,
                    nb_chev_filter,
                    nb_time_filter,
                    1,
                    cheb_polynomials,
                    num_of_vertices,
                    len_input // time_strides,
                )
            )
        self.final_conv = nn.Conv2d(
            int(len_input / time_strides), num_for_predict, kernel_size=(1, nb_time_filter)
        )
        self.intelligentadjustment = IntelligentAdjustment(
            num_of_vertices=num_of_vertices,
            num_timesteps=num_for_predict,
            target_sum=fleet_size,
        )
        self.register_buffer("L_f", L_f.float())
        self.DEVICE = device
        self.to(device)

    def forward(self, x, A_t, apply_rounding=None):
        # x: B,N,3,T.  Time channels are retained exactly as in the current project.
        time_of_day = x[:, :, 1, :].to(self.DEVICE)
        day_of_week = x[:, :, 2, :].to(self.DEVICE)

        I = torch.eye(A_t.size(-1), device=self.DEVICE, dtype=A_t.dtype).unsqueeze(0)
        A = A_t.to(self.DEVICE) + I
        At = A.transpose(-1, -2)
        P_f = A / (A.sum(dim=-1, keepdim=True) + 1e-8)
        P_b = At / (At.sum(dim=-1, keepdim=True) + 1e-8)

        F_t = x.to(self.DEVICE)
        h = x
        for block in self.BlockList:
            h = block(h, P_f, P_b, F_t, self.L_f, time_of_day, day_of_week)

        output = self.final_conv(h.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        return self.intelligentadjustment(output, apply_rounding=apply_rounding)


def make_model(
    DEVICE,
    nb_block,
    in_channels,
    K,
    nb_chev_filter,
    nb_time_filter,
    time_strides,
    adj_mx,
    num_for_predict,
    len_input,
    num_of_vertices,
    fleet_size,
):
    L_tilde = scaled_Laplacian(adj_mx)
    cheb = [torch.from_numpy(x).float().to(DEVICE) for x in cheb_polynomial(L_tilde, K)]
    L_f = torch.from_numpy(first_order_graph(adj_mx)).float().to(DEVICE)

    net = ASTGCN_submodule(
        DEVICE,
        nb_block,
        in_channels,
        K,
        nb_chev_filter,
        nb_time_filter,
        time_strides,
        cheb,
        num_for_predict,
        len_input,
        num_of_vertices,
        L_f,
        fleet_size,
    )

    for p in net.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p, -0.1, 0.1)
    return net
