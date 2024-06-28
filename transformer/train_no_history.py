from transformer.transformer import Encoder

# TODO: Create Dataloader and training loop

dataset = HistoryDataset()

encoder = Encoder(d_model=d_model,
                  n_head=n_head,
                  max_len=max_len,
                  ffn_hidden=ffn_hidden,
                  enc_voc_size=enc_voc_size,
                  drop_prob=drop_prob,
                  n_layers=n_layers,
                  device=device)