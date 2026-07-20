"""
Small GRU classifier: sequence of time-varying biomarkers -> final hidden
state at each patient's LAST REAL (non-padded) timestep -> concatenated
with static (time-invariant) covariates -> single logit.

The GRU runs over the full zero-padded MAX_SEQ_LEN sequence (fine at this
short a length -- no real benefit to pack_padded_sequence here), but we
only ever READ the hidden state at index (length-1) per patient via
`gather`, so padded steps' outputs are simply never used downstream.
"""

import torch
import torch.nn as nn


class MortalityGRU(nn.Module):
    def __init__(self, n_time_features, n_static_features,
                 hidden_size=24, num_layers=1, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_time_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size + n_static_features, 1)

    def forward(self, x_time, lengths, x_static):
        # x_time: (batch, MAX_SEQ_LEN, n_time_features)
        # lengths: (batch,) int64, number of REAL timesteps per patient
        # x_static: (batch, n_static_features)
        out, _ = self.gru(x_time)  # out: (batch, MAX_SEQ_LEN, hidden_size)

        idx = (lengths - 1).clamp(min=0)  # guard against length==0 edge case
        idx = idx.view(-1, 1, 1).expand(-1, 1, out.size(2))
        last_hidden = out.gather(1, idx).squeeze(1)  # (batch, hidden_size)

        last_hidden = self.dropout(last_hidden)
        combined = torch.cat([last_hidden, x_static], dim=1)
        logit = self.fc(combined).squeeze(1)
        return logit
