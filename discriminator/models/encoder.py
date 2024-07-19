import torch.nn as nn
from transformers import BertModel
from transformers import ViTModel


class ImageEncoder(nn.Module):
    """A simple image encoder using Vision Transformer (ViT). It returns the embeddings of the [CLS] token.

    Example:
    >>> import torch
    >>> from encoder import ImageEncoder
    >>> batch_size, num_channels, img_size = 32, 3, 224
    >>> x = torch.randn(batch_size, num_channels, img_size, img_size)
    >>> image_encoder = ImageEncoder()
    >>> embeddings = image_encoder(x)
    >>> print(embeddings.shape)
    torch.Size([32, 768])

    Note:
    - The input to the model is expected to be a tensor of shape (batch, num_channels, img_size, img_size).
    - The output of the model is a tensor of shape (batch, 768).
    - All the parameters of the model are frozen by default, run `unfreeze()` to unfreeze the parameters.
    """

    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.freeze()

    def forward(self, x):
        """x: (batch, num_channels, img_size, img_size)"""
        # pass the data through vit and return embeddings
        outputs = self.vit(x)
        return outputs.last_hidden_state[
            :, 0, :
        ]  # only return the embeddings of the [CLS] token

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class TextEncoder(nn.Module):
    """A simple text encoder using BERT. It returns the embeddings of the [CLS] token.

    Example:
    >>> from transformers import BertTokenizer
    >>> from encoder import TextEncoder
    >>> seq_len, batch_size = 128, 32
    >>> sentences = ["This is a test sentence"] * batch_size
    >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    >>> encoding = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    >>> input_ids = encoding['input_ids']
    >>> attention_mask = encoding['attention_mask']
    >>> text_encoder = TextEncoder()
    >>> embeddings = text_encoder(input_ids, attention_mask)
    >>> print(embeddings.shape)
    torch.Size([32, 768])

    Note:
    - The input to the model is expected to be a tensor of shape (batch, seq_len).
    - The output of the model is a tensor of shape (batch, 768).
    - All the parameters of the model are frozen by default, run `unfreeze()` to unfreeze the parameters.
    """

    def __init__(self):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.freeze()

    def forward(self, input_ids, attention_mask):
        """x: (batch, seq_len)"""
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[
            :, 0, :
        ]  # only return the embeddings of the [CLS] token

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    import doctest

    doctest.testmod()
