import torch
import torch.nn as nn
import torch.nn.functional as F


class HistoryModelBlind(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        img_dim,
        num_heads=4,
        num_layers=4,
        max_len=5000,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.img_dim = img_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_len = max_len

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=0)
        self.positional_encoding = nn.Parameter(
            self.get_positional_encoding(self.max_len, self.embedding_dim),
            requires_grad=False,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        self.linear = nn.Linear(self.img_dim, self.hidden_dim)
        self.linear_separate = nn.Linear(self.img_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.0)  # no dropout

    def get_positional_encoding(self, max_len, embedding_dim):
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(
        self,
        segment,
        prev_histories,
        lengths,
        separate_images,
        visual_context,
        normalize,
        device,
    ):
        batch_size = segment.shape[0]
        seq_len = segment.shape[1]

        embeds_words = self.embedding(segment)  # b, l, d

        # Apply positional encoding up to the length of the sequence
        positional_encoding = self.positional_encoding[:, :seq_len, :].to(
            embeds_words.device
        )
        embeds_words = embeds_words + positional_encoding
        # embeds_words = self.dropout(embeds_words)

        embeds_words = embeds_words.permute(1, 0, 2)

        # transformer encoder forward
        transformer_out = self.transformer_encoder(embeds_words)

        transformer_out = transformer_out.permute(1, 0, 2)

        # only take the last hidden state for each sequence
        batch_out_hidden = transformer_out[:, -1, :]

        separate_images = self.linear_separate(separate_images)
        separate_images = self.dropout(separate_images)

        # get linguistic history per image specific to that game up to that segment
        for b in range(batch_size):
            prev_hist = prev_histories[b]
            for p in range(len(prev_hist)):
                cur_img_hist = torch.tensor(prev_hist[p]).long().to(device)
                if len(prev_hist[p]) > 0:
                    # encode linguistic background
                    hist_embeds = self.embedding(cur_img_hist).view(
                        1, -1, self.embedding_dim
                    )
                    hist_len = hist_embeds.size(1)
                    hist_positional_encoding = self.positional_encoding[
                        :, :hist_len, :
                    ].to(hist_embeds.device)
                    hist_embeds = hist_embeds + hist_positional_encoding
                    # hist_embeds = self.dropout(hist_embeds)

                    # transformer encoder for history
                    hist_embeds = hist_embeds.permute(1, 0, 2)
                    hist_out = self.transformer_encoder(hist_embeds)
                    hist_out = hist_out.permute(1, 0, 2)

                    separate_images[b][p] += hist_out[:, -1, :].squeeze()

                else:
                    # no linguistic background, just use image features
                    pass

        if normalize:
            separate_images = self.relu(separate_images)
            separate_images = F.normalize(separate_images, p=2, dim=2)

        dot = torch.bmm(
            separate_images, batch_out_hidden.view(batch_size, self.hidden_dim, 1)
        )

        return dot
