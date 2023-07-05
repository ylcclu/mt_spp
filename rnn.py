import torch.nn as nn
import batch, config
from dictionary import Dictionary, remove_special_symbols
import textloader as tl
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import optim
from rnn_dataset import RNNDataset
import rnn_search
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
import bleu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Encoder:processes an input seq and produces a compact representation that captures the important information from the input.
# This compact representation, called hidden state, can then be used by the decoder to generate the desired output sequence.
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # Embedding layer to convert input indices to dense vectors
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=config.index_padding)
        # GRU (Gated Recurrent Unit) layer for sequential processing
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        # Dropout layer for regularization and preventing overfitting
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        # input: batch_size x max_length
        # Embedding the input sequence
        embedded = self.dropout(self.embedding(input))
        # embedded: batch_size x max_length x hidden dimension

        # Passing the embedded sequence through the GRU layer
        output, hidden = self.gru(embedded)
        # output: batch_size x max_length x hidden dimension
        # hidden: 1 x batch_size x hidden
        return output, hidden

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        # Compute attention scores using Bahdanau's additive attention mechanism

        # formula for attention
        # Apply linear transformation to the query tensor
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        # scores: batch_size x max_length x 1

        # Squeeze the last dimension of scores tensor and unsqueeze a new dimension at index 1
        scores = scores.squeeze(2).unsqueeze(1)
        # scores: batch_size x 1 x max_length

        # Apply softmax activation function along the last dimension to obtain attention weights
        weights = F.softmax(scores, dim=-1)
        # weights: batch_size x 1 x max_length

        # batch matrix multiplication
        # weights contains attention probabiltities
        # keys: encoder output
        context = torch.bmm(weights, keys)

        return context, weights

# AttnDecoderRNN represents the decoder component of a sequence-to-sequence model with attention mechanism.
# generate the output sequence by attending to relevant parts of the input sequence
# The attention mechanism allows the decoder to focus on different parts of the input sequence at each time step, 
# enabling it to capture the context and align the generated output accordingly.

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=config.index_padding)
        self.attention = BahdanauAttention(hidden_size)
        # decoder expects the concatenated input of the embedded input and the context vector -> 2* hidden_size
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        # Get the batch size from the encoder outputs
        batch_size = encoder_outputs.size(0)

        # Initialize the decoder input with a tensor filled with the start-of-sequence token
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(config.index_sentence_start)
        # decoder_input: batch_size x 1, filled with SOS

        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = [] # attention weights

        for i in range(config.max_length):
            # forward step is 'recursive'
            # decoder_output holds probability (not softmaxed!) what the model would predict
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                
                # removes one dimension
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))
        print(embedded.shape)
        print(embedded)

        # permutation to match dimensions in attention
        query = hidden.permute(1, 0, 2) 

        # context vector, attention weights
        context, attn_weights = self.attention(query, encoder_outputs)
        print(input.shape)
        input_gru = torch.cat((embedded, context), dim=2)

        # input for gru: previous hidden (encoder), embedded, context vector
        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

def accuracy(prediction, labels):
    # This function calculates the accuracy of a prediction compared to the given labels.
    
    # Convert the prediction into predicted values by selecting the index with the highest probability
    pred_values = torch.argmax(prediction, dim=-1).int()

    # Compare the predicted values with the labels and create a tensor of 1s for correct predictions and 0s for incorrect predictions
    correct = torch.eq(labels, pred_values).float()

    # Create a mask to ignore padding values in the labels
    mask = torch.gt(labels, config.index_padding).float()

    # Multiply the mask with the correct predictions to keep only the correct predictions not affected by padding
    # Calculate the total number of correct predictions by summing the values of masked_correct
    n_correct = torch.sum(torch.mul(mask, correct))
    # Calculate the total number of non-padding labels
    n_total = torch.sum(mask)

    # dividing the number of correct predictions by the total number of non-padding labels
    return n_correct / n_total

def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion):
    accuracy_total = 0
    total_loss = 0
    for step, (src, tgt) in enumerate(dataloader):
        input_tensor = src.to(device=device)
        target_tensor = tgt.to(device=device)

        # Zero the gradients of the encoder and decoder optimizers
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # Forward pass through the encoder to obtain encoder outputs and hidden state
        encoder_outputs, encoder_hidden = encoder(input_tensor)

        # Forward pass through the decoder to obtain decoder outputs, hidden state, and attention weights
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        # Calculate the accuracy by comparing decoder outputs with target tensor
        accuracy_total += accuracy(decoder_outputs, target_tensor)

        loss = criterion(
            # flatten the decoder_outputs tensor into a 2D tensor 
            # where each row represents a prediction distribution over the output vocabulary for a specific time step.
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            # reshape the target_tensor tensor into a 1D tensor by flattening it
            # this is done to align the shape of the target tensor with the reshaped decoder_outputs tensor
            target_tensor.view(-1)
        )
        # Backpropagation and optimization
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
        if step % 300 == 0:
            print(f"Step: {step + 1}/{len(dataloader)} for this epoch.")

    return total_loss / len(dataloader), accuracy_total / len(dataloader)

import time
import math

# Converts a duration in seconds to the format of minutes and seconds.
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# Calculates the elapsed time since a given timestamp and estimates the remaining time based on a completion percentage.
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def train(train_dataloader, encoder, decoder, n_epochs, learning_rate=config.learning_rate, checkpoint=None,
               print_every=100, plot_every=100, save_path=None, save=False):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    print_perplexity_total = 0
    cur_epoch = 0
    # Set up optimizer for encoder and decoder
    encoder_optimizer = optim.RMSprop(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.RMSprop(decoder.parameters(), lr=learning_rate)

    # Load checkpoint if provided
    if checkpoint:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        encoder_optimizer.load_state_dict(checkpoint['en_optim_state_dict'])
        decoder_optimizer.load_state_dict(checkpoint['de_optim_state_dict'])
        loss = checkpoint['loss']
        cur_epoch = checkpoint['epoch']
        print_loss_total += loss
        plot_loss_total += loss
        print("Checkpoint loaded successfully!")

    # NLL Loss is the same as Cross Entropy, 
    # but Cross Entropy also softmaxes the tensor (already did that in decoder!)
    criterion = nn.NLLLoss(ignore_index=config.index_padding)
    # during the loss calculation, any target values that have this padding index will be ignored and not contribute to the loss.
    # helps ensure that the loss is only computed for the relevant parts of the sequences

    for epoch in range(cur_epoch + 1, n_epochs + cur_epoch + 1):
        # Perform training for the current epoch
        loss, accuracy = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

        # Update total losses and perplexity
        print_loss_total += loss
        plot_loss_total += loss
        print_perplexity_total += math.exp(loss)

        # Print progress and metrics at print_every interval
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_perplexity_avg = print_perplexity_total / print_every
            print_loss_total = 0
            print_perplexity_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, (epoch - cur_epoch) / n_epochs),
                                        epoch, (epoch - cur_epoch) / n_epochs * 100, print_loss_avg))
            print(f"Average Perplexity is {print_perplexity_avg}.")
            print(f"Average Accuracy is {accuracy}.")
            
        # Append average plot loss at plot_every interval for visualization
        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        print(f"Epoch {epoch} done!")
        
        if save:
            # save after each epoch
            checkpoint = {
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'en_optim_state_dict': encoder_optimizer.state_dict(),
                'de_optim_state_dict': decoder_optimizer.state_dict(),
                'loss': loss
            }
            torch.save(checkpoint, save_path+f"/epoch_{epoch}.pt")
            print(f"Model saved at the end of epoch {epoch}.")

    showPlot(plot_losses)


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()

def plot_epochs():
    plotpoints = []
    for idx in range(1,26):
        path = 'model/adam_gru_teacher_force/epoch_' + str(idx)+ '.pt'
        corpus_greedy = rnn_search.generate_translation(encoder, decoder, path, dev_src, src_dict, tgt_dict)
        ref = dev_tgt
        corpus_greedy = remove_special_symbols(corpus_greedy)
        plotpoints.append(bleu.bleu(ref, corpus_greedy))
        print("Bleu done for " + path + " !")
    showPlot(plotpoints)

if __name__ == '__main__':
    save_path = 'model/adam_gru_teacher_force'

    src = tl.loadData('data/bpe_de_7000.txt')
    tgt = tl.loadData('data/bpe_en_7000.txt')
    dev_src = tl.loadData('data/bpe_7k_multi30k.dev.de')
    dev_tgt = tl.loadData('data/bpe_7k_multi30k.dev.en')

    src_dict = Dictionary(src)
    tgt_dict = Dictionary(tgt)
    textdataset = RNNDataset(src, tgt, src_dict, tgt_dict)
    batch_size = config.batch_size
    dataloader = DataLoader(dataset=textdataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True)
    encoder = EncoderRNN(src_dict.n_words, config.embedding_dimension).to(device)
    decoder = AttnDecoderRNN(config.embedding_dimension, tgt_dict.n_words).to(device)

    ##load checkpoint
    checkpoint = torch.load('rnn_model/adam_gru_teacher_force/epoch_11.pt')

    ##train from checkpoint, for that checkpoint need to be loaded first (see above)
    #train(dataloader, encoder, decoder, 5, print_every=1, plot_every=1, save_path=save_path, save=True, checkpoint=checkpoint)

    ##train from scratch
    #train(dataloader, encoder, decoder, 20, print_every=1, plot_every=1, save_path=save_path, save=True)


    ## translate

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    dev_src_array = batch.text_to_array(dev_src, src_dict)
    print(dev_src[800])

    ##greedy direct
    #print(batch.array_to_text([rnn_search.greedy_search(encoder, decoder, dev_src_array[800])], tgt_dict))

    ##beam direct
    rnn_search.beam_search(encoder, decoder, dev_src_array[800])

    ## translation greedy
    #corpus_greedy = rnn_search.generate_translation(encoder, decoder, 'model/adam_gru_teacher_force/epoch_11.pt', dev_src, src_dict, tgt_dict)
    #corpus_greedy = remove_special_symbols(corpus_greedy)
    #print(tl.unformat_text(corpus_greedy))
    #ref = remove_special_symbols(dev_tgt)
    #print(bleu.bleu(ref, corpus_greedy))
