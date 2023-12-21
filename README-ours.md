# Towards Seamless Humanoid Control: Generation of Physically Feasible Motions from User Text Input


```bash
python eval.py --checkpoint latest_tf.ckpt -o eval_output_dir/


python eval.py --checkpoint full_5e7.ckpt --output_dir data/pusht_eval_output --device cuda:0
```


## FAQ

### libpython3.8 error

```python
ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory
```

#### Solution

```bash
export LD_LIBRARY_PATH=/home/tk/Documents/mambaforge/envs/calm/lib:$LD_LIBRARY_PATH
```


# Model Optimization

raw model

```python
TransformerForDiffusion(
  (input_emb): Linear(in_features=31, out_features=512, bias=True)
  (drop): Dropout(p=0.0, inplace=False)
  (time_emb): SinusoidalPosEmb()
  (cond_obs_emb): Linear(in_features=317, out_features=512, bias=True)
  (encoder): Sequential(
    (0): Linear(in_features=512, out_features=2048, bias=True)
    (1): Mish()
    (2): Linear(in_features=2048, out_features=512, bias=True)
  )
  (decoder): TransformerDecoder(
    (layers): ModuleList(
      (0-15): 16 x TransformerDecoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
        )
        (multihead_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
        )
        (linear1): Linear(in_features=512, out_features=2048, bias=True)
        (dropout): Dropout(p=0.01, inplace=False)
        (linear2): Linear(in_features=2048, out_features=512, bias=True)
        (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.01, inplace=False)
        (dropout2): Dropout(p=0.01, inplace=False)
        (dropout3): Dropout(p=0.01, inplace=False)
      )
    )
  )
  (ln_f): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  (head): Linear(in_features=512, out_features=31, bias=True)
)
```

evaluation model
```python
TransformerForDiffusion(
  (input_emb): Linear(in_features=31, out_features=512, bias=True)
  (drop): Dropout(p=0.0, inplace=False)
  (time_emb): SinusoidalPosEmb()
  (cond_obs_emb): Linear(in_features=317, out_features=512, bias=True)
  (encoder): Sequential(
    (0): Linear(in_features=512, out_features=2048, bias=True)
    (1): Mish()
    (2): Linear(in_features=2048, out_features=512, bias=True)
  )
  (decoder): TransformerDecoder(
    (layers): ModuleList(
      (0-15): 16 x TransformerDecoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
        )
        (multihead_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
        )
        (linear1): Linear(in_features=512, out_features=2048, bias=True)
        (dropout): Dropout(p=0.01, inplace=False)
        (linear2): Linear(in_features=2048, out_features=512, bias=True)
        (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (norm3): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.01, inplace=False)
        (dropout2): Dropout(p=0.01, inplace=False)
        (dropout3): Dropout(p=0.01, inplace=False)
      )
    )
  )
  (ln_f): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
  (head): Linear(in_features=512, out_features=31, bias=True)
)
```

after compile

```python
OptimizedModule(
  (_orig_mod): TransformerForDiffusion(
    (input_emb): Linear(in_features=31, out_features=512, bias=True)
    (drop): Dropout(p=0.0, inplace=False)
    (time_emb): SinusoidalPosEmb()
    (cond_obs_emb): Linear(in_features=317, out_features=512, bias=True)
    (encoder): Sequential(
      (0): Linear(in_features=512, out_features=2048, bias=True)
      (1): Mish()
      (2): Linear(in_features=2048, out_features=512, bias=True)
    )
    (decoder): TransformerDecoder(
      (layers): ModuleList(
        (0-15): 16 x TransformerDecoderLayer(
          (self_attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
          )
          (multihead_attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
          )
          (linear1): Linear(in_features=512, out_features=2048, bias=True)
          (dropout): Dropout(p=0.01, inplace=False)
          (linear2): Linear(in_features=2048, out_features=512, bias=True)
          (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (norm3): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.01, inplace=False)
          (dropout2): Dropout(p=0.01, inplace=False)
          (dropout3): Dropout(p=0.01, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    (head): Linear(in_features=512, out_features=31, bias=True)
  )
)
```