import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable, Function

"""
Early fusion
"""
class ConcatEarly(nn.Module):
    """Concatenation of input data on dimension -1."""

    def __init__(self):
        super(ConcatEarly, self).__init__()

    def forward(self, x, y):
        return torch.cat([x, y], dim=-1)


"""
Cross-attention
"""
class CrossAttention(nn.Module):
    """Cross Attention module with additional feed-forward network and residual connections."""

    def __init__(self, embed_dim=768, num_heads=8, ff_dim=1024, dropout=0.1):
        super(CrossAttention, self).__init__()
        
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim*2, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim*2),
        )
        
        self.norm1_x = nn.LayerNorm(embed_dim)
        self.norm1_y = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim*2)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, y):
        x1, _ = self.attn(x, y, y)
        y1, _ = self.attn(y, x, x)
        
        x1 = self.norm1_x(x + x1)
        y1 = self.norm1_y(y + y1)
        
        fused = torch.cat([x1, y1], dim=-1)
        fused = self.ffn(fused)
        
        fused = self.dropout(self.norm2(fused + torch.cat([x1, y1], dim=-1)))
        
        return fused


"""
Tensor fusion
Adapted from https://github.com/Justin1904/TensorFusionNetworks/blob/master/model.py
"""
class TensorFusion(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, x, y):
        
        nonfeature_size = x.shape[:-1]

        m = torch.cat((torch.ones(*nonfeature_size, 1, device=x.device, dtype=x.dtype), x), dim=-1)
        y = torch.cat((torch.ones(*nonfeature_size, 1, device=y.device, dtype=y.dtype), y), dim=-1)

        fused = torch.einsum('...i,...j->...ij', m, y)
        fused = fused.reshape([*nonfeature_size, -1])
        
        return fused


"""
NL-gate
See section F4 of https://arxiv.org/pdf/1905.12681.pdf for details
"""
class NLgate(torch.nn.Module):

    def __init__(self, thw_dim=1, c_dim=768, tf_dim=1, q_linear=None, k_linear=None, v_linear=None):
        super(NLgate, self).__init__()
        self.qli = None
        self.kli = None
        self.vli = None
        self.thw_dim = thw_dim
        self.c_dim = c_dim
        self.tf_dim = tf_dim
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, y):
        """
        Apply Low-Rank TensorFusion to input.
        """
        q = x
        k = y
        v = y
        if self.qli is None:
            qin = q.view(-1, self.thw_dim, self.c_dim)
        else:
            qin = self.qli(q).view(-1, self.thw_dim, self.c_dim)
        if self.kli is None:
            kin = k.view(-1, self.c_dim, self.tf_dim)
        else:
            kin = self.kli(k).view(-1, self.c_dim, self.tf_dim)
        if self.vli is None:
            vin = v.view(-1, self.tf_dim, self.c_dim)
        else:
            vin = self.vli(v).view(-1, self.tf_dim, self.c_dim)
        matmulled = torch.matmul(qin, kin)
        finalout = torch.matmul(self.softmax(matmulled), vin)
        return torch.flatten(qin + finalout, 1)


"""
MISA
Adapted from: https://github.com/declare-lab/MISA
Paper: MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis
"""

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse

class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, x, y):

        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        y = y.view(batch_size, -1)

        # Zero mean
        x_mean = torch.mean(x, dim=0, keepdims=True)
        y_mean = torch.mean(y, dim=0, keepdims=True)
        x = x - x_mean
        y = y - y_mean

        x_l2_norm = torch.norm(x, p=2, dim=1, keepdim=True).detach()
        x_l2 = x.div(x_l2_norm.expand_as(x) + 1e-6)
        

        y_l2_norm = torch.norm(y, p=2, dim=1, keepdim=True).detach()
        y_l2 = y.div(y_l2_norm.expand_as(y) + 1e-6)

        diff_loss = torch.mean((x_l2.t().mm(y_l2)).pow(2))

        return diff_loss


class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """
    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x, y, n_moments):
        mx = torch.mean(x, 0)
        my = torch.mean(y, 0)
        sx = x-mx
        sy = y-my
        dm = self.matchnorm(mx, my)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx, sy, i + 2)
        return scms

    def matchnorm(self, x, y):
        power = torch.pow(x-y,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt

    def scm(self, sx, sy, k):
        ss1 = torch.mean(torch.pow(sx, k), 0)
        ss2 = torch.mean(torch.pow(sy, k), 0)
        return self.matchnorm(ss1, ss2)


class MISA(nn.Module):
    def __init__(self):
        super(MISA, self).__init__()

        audio_dim   = 768
        text_dim    = 768
        hidden_dim  = 768
        
        output_dim = hidden_dim
        
        # shared encoder
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        self.shared.add_module('shared_activation_1', nn.ReLU())

        # reconstruct
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=hidden_dim*2, out_features=hidden_dim))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=hidden_dim*2, out_features=hidden_dim))

        # fusion + cls
        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=hidden_dim*4, out_features=output_dim*2))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim,  nhead=24)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        

    def shared_private(self, utterance_t,  utterance_a):

        self.utterance_t = utterance_t
        self.utterance_a = utterance_a

        self.utt_shared_t = self.shared(self.utterance_t)
        self.utt_shared_a = self.shared(self.utterance_a)

        
    def reconstruct(self):
        self.utt_t = torch.cat((self.utterance_t, self.utt_shared_t), dim=1)
        self.utt_a = torch.cat((self.utterance_a, self.utt_shared_a), dim=1)    

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_a_recon = self.recon_a(self.utt_a)

    # inter loss calculation
    def get_recon_loss(self):
        loss =  MSE()(self.utt_t_recon, self.utterance_t)
        loss += MSE()(self.utt_a_recon, self.utterance_a)
        loss = loss / 3.0
        return loss

    def get_diff_loss(self):
        shared_t = self.utt_shared_t
        shared_a = self.utt_shared_a
        private_t = self.utterance_t
        private_a = self.utterance_a

        # Between private and shared
        loss =  DiffLoss()(private_t, shared_t)
        loss += DiffLoss()(private_a, shared_a)

        # Across privates
        loss += DiffLoss()(private_a, private_t)
        return loss

    def get_cmd_loss(self):
        # losses between shared states
        loss += CMD()(self.utt_shared_t, self.utt_shared_a, 5)
        loss = loss/3.0
        return loss
    
    def forward(self, text, audio):
        utterance_audio = audio.squeeze(1) # [batch, hidden]
        utterance_text  = text.squeeze(1)   # [batch, hidden]

        # shared-private encoders
        self.shared_private(utterance_text, utterance_audio)

        # reconstruction
        self.reconstruct()
        
        # 1-LAYER TRANSFORMER FUSION
        h = torch.stack((self.utterance_t, self.utterance_a, self.utt_shared_t, self.utt_shared_a), dim=0)
        h = self.transformer_encoder(h)
        h = torch.cat((h[0], h[1], h[2], h[3]), dim=1)
        features = self.fusion(h)

        return features


"""
Modality-gated fusion
Adapted from "Cross-Attention is Not Enough": https://arxiv.org/abs/2305.13583
"""

class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.attn = nn.MultiheadAttention(2304, 16, batch_first=True)

    def forward(self, h):
        self_attn, _ = self.attn(h, h, h)
        return self_attn
    
class ModalityGatedFusion(nn.Module):
    def __init__(self):
        super(ModalityGatedFusion, self).__init__()
        self.W = nn.Parameter(torch.ones(2), requires_grad=True)
        self.W2 = nn.Parameter(torch.ones(3), requires_grad=True)
        self.cross_attention = CrossAttention()
        self.self_attention = SelfAttention()

    def forward(self, x, y):
        W_prime = F.softmax(self.W, dim=0)
        W2_prime = F.softmax(self.W2, dim=0)
        W_x = W_prime[0]
        W_y = W_prime[1]
        W2_1 = W2_prime[0]
        W2_2 = W2_prime[1]
        W2_3 = W2_prime[2]
        
        if torch.argmax(W_prime) == 0:
            x_prime = W_x * x
            y_prime = W_y * self.cross_attention(x, y)
        else:
            x_prime = W_x * self.cross_attention(y, x)
            y_prime = W_y * y
                    
        H = torch.cat([W2_1 * x, W2_2 * y, W2_3 * self.self_attention(torch.cat((x_prime, y_prime), dim=-1))], dim = -1)

        return H


"""
Late fusion

There is no model for late fusion as it is performed at the decision level.
You should build two baseline models for text and audio respectively.
And then use the following code to select the prediction.
Our strategy is that if the predictions from text and audio are the same, then the prediction is selected.
Otherwise, the one with higher probability is selected.

Note that the following code should be removed from this .py file and attached in your training model.
"""

if np.argmax(predictions_test_t[i]) == np.argmax(predictions_test_a[i]):
    predictions_test.append([np.argmax(predictions_test_t[i])])
else:
    if max(predictions_test_t[i]) > max(predictions_test_a[i]):
        predictions_test.append([np.argmax(predictions_test_t[i])])
    else:
        predictions_test.append([np.argmax(predictions_test_a[i])])
