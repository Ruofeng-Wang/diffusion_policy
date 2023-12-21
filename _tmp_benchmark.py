import time
import math
from typing import Union, Optional, Tuple

import torch
from torch.nn import *


class SinusoidalPosEmb(Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb



class TransformerForDiffusion(Module):
    def __init__(self):
        super().__init__()
        
        self.input_emb = Linear(in_features=31, out_features=512, bias=True)
        
        self.pos_emb = Parameter(torch.zeros(1, 8, 512))
        self.drop = Dropout(p=0.0, inplace=False)
        self.time_emb = SinusoidalPosEmb(512)
        self.cond_obs_emb = Linear(in_features=317, out_features=512, bias=True)
        self.encoder = Sequential(
            Linear(in_features=512, out_features=2048, bias=True),
            GELU(),
            Linear(in_features=2048, out_features=512, bias=True),
        )
        
        decoder_layer = TransformerDecoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=4*512,
            dropout=0.01,
            activation="gelu",
            batch_first=True,
            norm_first=True # important for stability
        )
        
        self.decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=16
        )
        
        self.cond_pos_emb = Parameter(torch.zeros(1, 5, 512))
        
        self.ln_f = LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        self.head = Linear(in_features=512, out_features=31, bias=True)

        sz = 8
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        self.register_buffer("mask", mask)
        
        S = 5
        t, s = torch.meshgrid(
            torch.arange(8),
            torch.arange(S),
            indexing='ij'
        )
        mask = t >= (s-1) # add one dimension since time is the first token in cond
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        self.register_buffer('memory_mask', mask)


    
    def forward(self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        cond: Optional[torch.Tensor]=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,T',cond_dim)
        output: (B,T,input_dim)
        """
        
        # import time
        # _time_cnt = time.perf_counter()
        
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B,1,n_emb)
        
        # print(time.perf_counter() - _time_cnt)
        # _time_cnt = time.perf_counter()

        # process input
        input_emb = self.input_emb(sample)

        # encoder
        cond_embeddings = time_emb
        cond_obs_emb = self.cond_obs_emb(cond)
        # (B,To,n_emb)
        cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
        tc = cond_embeddings.shape[1]
        position_embeddings = self.cond_pos_emb[
            :, :tc, :
        ]  # each position maps to a (learnable) vector
        x = self.drop(cond_embeddings + position_embeddings)
        x = self.encoder(x)
        memory = x
        # (B,T_cond,n_emb)
        
        # decoder
        token_embeddings = input_emb
        t = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[
            :, :t, :
        ]  # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        # (B,T,n_emb)
        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=self.mask,
            memory_mask=self.memory_mask
        )
        # (B,T,n_emb)
        
        # print(time.perf_counter() - _time_cnt)
        # _time_cnt = time.perf_counter()
        
        # head
        x = self.ln_f(x)
        x = self.head(x)
        # (B,T,n_out)
        
        # print(time.perf_counter() - _time_cnt)
        # _time_cnt = time.perf_counter()
        
        return x






# DEVICE = "cuda:0"
DEVICE = "cpu"

print(f"using device {DEVICE}")

#model = torch.load("modell.pt")
model = TransformerForDiffusion()

model.eval()
model.to(DEVICE)


# print("quantizing model...")
# model_int8 = torch.ao.quantization.quantize_dynamic(
#     model,  # the original model
#     {
#         torch.nn.Linear,
#         # torch.nn.MultiheadAttention
#     },  # a set of layers to dynamically quantize
#     dtype=torch.qint8)


# print("compiling model...")
# #torch._dynamo.list_backends()
# #['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'openxla_eval', 'tvm']

# model = torch.compile(model, mode="reduce-overhead")
# #model = torch.compile(model, backend="tvm")
# # model = torch.compile(model, backend="onnxrt")


print("done")


with torch.no_grad():
    for i in range(100):
        trajectrory = torch.randn(size=(1, 8, 31), dtype=torch.float32, device=DEVICE)
        t = 90
        cond = torch.randn(size=(1, 4, 317), dtype=torch.float32, device=DEVICE)

        result = model.forward(trajectrory, t, cond)

time_start = time.perf_counter()

with torch.no_grad():
    for i in range(100):
        trajectrory = torch.randn(size=(1, 8, 31), dtype=torch.float32, device=DEVICE)
        t = 90
        cond = torch.randn(size=(1, 4, 317), dtype=torch.float32, device=DEVICE)

        result = model.forward(trajectrory, t, cond)

time_elapsed = time.perf_counter() - time_start

print(f"Time elapsed: {time_elapsed:.6f} s \t frequency: {1. / time_elapsed:.6f} Hz")

