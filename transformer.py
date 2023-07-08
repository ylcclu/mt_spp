import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
import copy, math, time
from config import NUM_LAYERS, DIM_MODEL, DIM_FF, NUM_HEADS, DROPOUT, TRANSFORMER_BATCH_SIZE, index_padding, BASE_LR, WARMUP, TRANSFORMER_NUM_EPOCHS, ACCUM_ITER
from transformer_dataset import TransformerDataset
import textloader as tl
from dictionary import Dictionary
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformer_dataset import Seq2SeqBatch, TransformerDataset, generate_translation

class EncoderDecoder(nn.Module):
    "    A standard Encoder-Decoder architecture."

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and tgt sequences"
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    "Define standard linear + softmax generation step"
    
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        # d_model: output dim of previous sublayer
        #          input dim of the generator layer
        # output dim of generator layer is vocab_size
        # since final output is prob distr of tgt vocab
    
    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)
    
# for stacking of encoder or decoder layers
def clones(module, N):
    "Produce N identical layers"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    """
        Construct a layernorm module for rnn
        cp. batchnorm for ffnn
        to stabilize hidden states and reduce training time
    """
    
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn"
        for layer in self.layers:
            x = layer(x, mask)
            # masking during encoding so as not to attend to padding tokens
        return self.norm(x)

# out of each sub-layer is LayerNorm(x + Sublayer(x))
# where Sublayer(x) is the function implemented by the sub-layer itself
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        # during training, randomly drop units (along with their connections)
        # to prevent overfitting
        # the argument dropout is the probability where some elements of
        # the input tensor will randomly be zeroed

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size"
        # The norm is first as opposed to last for code simplicity
        return x + self.dropout(sublayer(self.norm(x)))
                                
# Each layer consists of two sub-layers:
# 1. a multi-head self-attention mechanism
# 2. a simple, position-wise fully connected ffn

class EncoderLayer(nn.Module):
    "Encoder made up of self-attn and ffn (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # 2 because EncoderLayer consists of self-attn and ffn
        self.size = size

    def forward(self, x, mask):
        "See Transformer archtecture in paper"
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
    
class Decoder(nn.Module):
    "Generic N layer decoder with masking"
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and ffn (defined below) => 3"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "See Transformer archtecture in paper"
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # masking to prevent positions from attending to subsequent positions (future words)
        # output embeddings are also offset by one position
        # => predicitons for position i can depend only on the known outputs at position < i
        return self.sublayer[2](x, self.feed_forward)

def attention(query, key, value, mask=None, dropout=None):
    """
    Scaled Dot-Product Attention
    calculate the attention function on a set of queries simultaneously 
    by way of matrix operations
    """

    d_k = query.size(-1) # dim of key
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # transpose(-2, -1): the last to dims are swapped
    # scare by \frac{1}{\sqrt(d_k)}
    # to counteract the effect of dot products growing large for large d_k
    # large dot product leads to small gradients after softmax
    # which is suspected to be why additive attn outperforms dot product attn
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
    # TODO: why also return attention score?

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """
        Args:
            h: number of heads, h=8 in this model
            d_model: model size
                d_k = d_v = d_model/h = 64 in this model
            dropout: prob rate for dropout. Defaults to 0.1.
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # assum d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        # 4 linear layers: see Figure 2
        self.attn = None # attention score is saved in self.attn
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, query, key, value, mask=None):
        "See Figure 2 in paper"
        if mask is not None:
            # same mask applied to all h heads
            mask = mask.unsqueeze(1)
            # unsqueeze returns a new tensor with a dim of size one inserted at the specified position
        nbatches = query.size(0)

        # 1) Do all the linear projection in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            # TODO: what do view and transpose(1,2) do exactly?
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = attention(
            query, key, value, mask=mask, dropout = self.dropout
        )

        # 3) "Concat" using a view and apply a final linear
        x = (
            x.transpose(1,2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        # TODO: what do transpose(1,2), contiguous(), and view() do exactly?
        del query
        del key
        del value
        return self.linears[-1](x)

# position-wise fully connected ffn in each of the encoder and decoder layer
# applied to each position seperately and identically (??????)
# ? while the linear transformation are the same across different positions, they use different parameters from layer to layer
# ? == two convolutions with kernel size 1
# consists of two linear transformation with a ReLU activation in between
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation"

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
        # TODO: does the relu function call work????

# Embeddings and Softmax
# use learned embeddings to convert input tokens and output tokens to vectors of dim d_model
# use learned linear transformation and softmax to convert decoder output to predicted next-token probabilities
# In this model: the same weight matrix is shared between the two embedding layers and pre-softmax linear transformation
# additional in the embedding layers: weights are multiplied by \sqrt(d_model)
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        # TODO: lut????
        self.d_model = d_model
    
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    
# Positional Encoding
# has the sam dim as the embeddings = d_model
# so that the two can be summed, then applied dropout
class PositionalEncoding(nn.Module):
    "Implement the PE function"

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # even position with sinus
        # slicing: [start:stop:step]
        pe[:, 0::2] = torch.sin(position * div_term)
        # odd position with cosinus
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
    
# define a function from hyperparameter to a full model
def make_model(
        src_vocab, tgt_vocab, N=NUM_LAYERS, d_model=DIM_MODEL,
        d_ff=DIM_FF, h=NUM_HEADS, dropout=DROPOUT
):
    "Helper: Construct a model from hyperparameters"
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )

    # TODO: why????
    # Initialize parameters with Glorot / fan_avg
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

class TrainState:
    "Track number of steps, examples, and tokens processed"

    step: int = 1 # Steps in the current epoch
    accum_step: int = 0 # Number of gradient accumulation steps
    samples: int = 0 # total # of examples used
    tokens: int = 0 # total # of tokens processed

def run_epoch(
        data_iter,
        model, loss_compute,
        optimizer,
        scheduler,
        mode="train",
        accum_iter=1,
        train_state=TrainState()
):
    "Train a single epoch"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        batch.src, batch.tgt, batch.tgt_y, batch.src_mask, batch.tgt_mask = (batch.src.to(device=device), 
                                               batch.tgt.to(device=device), 
                                               batch.tgt_y.to(device=device), 
                                               batch.src_mask.to(device=device), 
                                               batch.tgt_mask.to(device=device))
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter ?????
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                f"Epoch Step: {i} | Accumulation Step : {n_accum} "
                + f"| Loss: {(loss / batch.ntokens):.2f} " # padding tokens ignored
                + f"| Perplexity: {(math.exp(loss / batch.ntokens)):.2f}"
                + f"| Tokens / Sec: {(tokens / elapsed):.1f} "
                + f"| Learning Rate: {lr}"
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state

def learning_rate(step, model_size, factor, warmup):
    """
        default the step to 1 for LambdaLR function
        to avoid zero raising to negative power
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

# Regularization
# label smoothing introduces noise for the labels
# hurts perplexity, as the model learns to be more unsure
# but improves accuracy and BLEU score
class LabelSmoothing(nn.Module):
    # TODO: ????????
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())
    
class SimpleLossCompute:
    "A simple loss compute and train function"

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss
    
class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None
    
def train_model(model, train_dataloader, dev_loader, src_vocab_size, tgt_vocab_size, gpu=0,
                 save_path=None, save=False):

    model.cuda(gpu)
    module = model

    criterion = LabelSmoothing(
        size=tgt_vocab_size, padding_idx=index_padding, smoothing=0.1
    )
    criterion.cuda(gpu)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=BASE_LR, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer, 
        lr_lambda=lambda step: learning_rate(
            step, DIM_MODEL, factor=1, warmup=WARMUP
        )
    )
    train_state = TrainState()

    for epoch in range(TRANSFORMER_NUM_EPOCHS):
        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        loss, train_state = run_epoch(
            (Seq2SeqBatch(src, tgt, index_padding) for (src, tgt) in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=ACCUM_ITER,
            train_state=train_state
        )

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Seq2SeqBatch(src, tgt, index_padding) for (src, tgt) in dev_loader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(sloss)
        torch.cuda.empty_cache()

        if save:
            # save after each epoch
            # checkpoint = {
            #     'epoch': epoch,
            #     'model_state_dict': module.state_dict(),
            #     'optim_state_dict': optimizer.state_dict(),
            #     'loss': loss
            # }
            # torch.save(checkpoint, save_path+f"/epoch_{epoch}.pt")
            torch.save(module.state_dict(), save_path+f"/epoch_{epoch}.pt")
            print(f"Model saved at the end of epoch {epoch}.")
    
    print("Training ended.")

from os.path import exists

def load_trained_model(model: nn.Module, model_path):
    assert exists(model_path)
    model.load_state_dict(torch.load(model_path))
    return model

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from config import TRANSFORMER_NUM_EPOCHS
from dictionary import remove_special_symbols
from bleu import bleu

def showPlot(points):
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()

def plot_bleu(ref, dev_dataloader, model, model_path, vocab_tgt):
    plotpoints = []
    for idx in range(TRANSFORMER_NUM_EPOCHS):
        hypo_file_path = ""
        if idx == (TRANSFORMER_NUM_EPOCHS - 1):
            hypo_file_path = f"{model_path}/epoch_{idx}.txt"
        model_path = f"{model_path}/epoch_{idx}.pt"
        model = load_trained_model(model, model_path)
        hypos = remove_special_symbols(generate_translation(model, dev_dataloader, vocab_tgt))

        # save hypos in .txt file
        if idx == (TRANSFORMER_NUM_EPOCHS - 1):
            with open(hypo_file_path, 'w') as f:
                for sentence in hypos:
                    f.write(f"{sentence}\n")
        
        # plot bleu
        bleu_score = bleu(ref, hypos)
        plotpoints.append(bleu_score)
        print("Bleu done for " + model_path + f": {bleu_score}")
    showPlot(plotpoints)

if __name__ == "__main__":
    save_path = 'model/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src = tl.loadData('data/bpe_de_7000.txt')
    tgt = tl.loadData('data/bpe_en_7000.txt')
    dev_src = tl.loadData('data/bpe_7k_multi30k.dev.de')
    dev_tgt = tl.loadData('data/bpe_7k_multi30k.dev.en')
    ref = tl.loadData('data/bpe_7k_multi30k.dev.en')

    src_dict = Dictionary(src)
    tgt_dict = Dictionary(tgt)

    train_dataset = TransformerDataset(src, tgt, src_dict, tgt_dict)
    dev_dataset = TransformerDataset(dev_src, dev_tgt, src_dict, tgt_dict)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=TRANSFORMER_BATCH_SIZE,
                              shuffle=True,
                              drop_last=False)
    dev_loader = DataLoader(dataset=dev_dataset,
                            batch_size=TRANSFORMER_BATCH_SIZE,
                            shuffle=True,
                            drop_last=False)
    
    model = make_model(src_dict.n_words, tgt_dict.n_words, N=NUM_HEADS)

    train_model(model, train_loader, dev_loader, src_dict.n_words, tgt_dict.n_words, save_path=save_path, save=True)
    
    plot_bleu(ref, dev_loader, model, save_path, tgt_dict)
