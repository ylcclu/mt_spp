import torch
import torch.nn as nn
import textloader as tl
from dictionary import Dictionary, remove_special_symbols
import dataset, config
from torch.utils.data import DataLoader
# from torchsummary import summary
import math, os
from pathlib import Path
import search, batch, bleu
from bpe import BPE # for posrprocessing

# model
class FeedForwardNet(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, 
                 win_size, embedding_dim, hidden_size_1, hidden_size_2) -> None:
        super(FeedForwardNet, self).__init__()
        # word embedding
        # transforms discret values (word indices) into continuous ones
        self.src_embedding = nn.Embedding(src_vocab_size, embedding_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embedding_dim)

        # fully connected src
        self.fc_src = nn.Linear((2*win_size+1)*embedding_dim, hidden_size_1)
        # fully connected tgt
        self.fc_tgt = nn.Linear(win_size*embedding_dim, hidden_size_1)

        # fully connected layer 1
        self.layer1 = nn.Linear(hidden_size_1*2, hidden_size_2)
        self.batchnorm = nn.BatchNorm1d(hidden_size_2)
        self.layer2 = nn.Linear(hidden_size_2, hidden_size_2)
        # activation function
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # fully connected layer 2 / projection
        self.projection = nn.Linear(hidden_size_2, tgt_vocab_size) # output dim = size of target vocab



    def forward(self, src_win, tgt_win):
        # src_win: torch.Size([200, 7]), tgt_win: torch.Size([200, 3])
        # batch_size = 200, 7 = 2*win_size +1, 3 = win_size
        src_embedded = self.src_embedding(src_win).view(src_win.shape[0], 1, -1) # view to make shape fit for fc_src
        tgt_embedded = self.tgt_embedding(tgt_win).view(tgt_win.shape[0], 1, -1) # view to make shape fit for fc_tgt
        # src_emb: torch.Size([200, 1, 448]), tgt_emb: torch.Size([200, 1, 192])
        # 7*embedding_dim = 448, 3*embedding_dim = 192
        src = self.fc_src(src_embedded).view(src_embedded.shape[0],-1) # view to enable concatenation
        tgt = self.fc_tgt(tgt_embedded).view(tgt_embedded.shape[0],-1)
        # src = torch.Size([200, 500]), tgt = torch.Size([200, 500])
        concatenated = torch.cat((src,tgt), dim=1)
        # concatenated = torch.Size([200, 1000])
        out = self.layer1(concatenated)
        out = self.batchnorm(out) # stabilisiert Training, Mittelwert und Varianz auf trainierbare Parameter (?)
        out = self.tanh(out)
        # out = self.layer2(out)
        # out = self.relu(out)
        # out = self.batchnorm(out)
        out = self.projection(out)

        return out
    

def accuracy(predictions, labels):
    classes = torch.argmax(predictions, dim=1)
    # print(f"classes {classes}")
    # print(f"classes == labels {classes == labels}")
    # print(f"(classes == labels).float(){(classes == labels).float()}")
    # print(f"torch.mean((classes == labels).float()) {torch.mean((classes == labels).float())}")
    return torch.mean((classes == labels).float())

def count_parameters(model) -> int:
    # torch.numel(input): Returns the total number of elements in the input tensor.
    # requires_grad only true for trainable parameters
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# test on dev set
def test(loss_function):
    with torch.no_grad():
        running_accuracy = 0
        running_perplexity = 0
        loss = 0

        for _, (src_win, tgt_win, tgt_lb) in enumerate(dev_loader):
            
            # predictions
            output = model(src_win, tgt_win)

            loss = loss_function(output, tgt_lb)

            running_accuracy += accuracy(output, tgt_lb)

            running_perplexity += torch.exp(loss).item()
            
        print(f"testing: accuracy on the dev set = {(running_accuracy / len(dev_loader)):.4f}, "
              f"perplexity on the dev set = {(running_perplexity /len(dev_loader)):.4f}")

    
def train(model, optimizer, loss, n_iters, checkpoint=None, 
          print_every_iter=1000, test_every_epoch=1, adjust_lr=True,
          save=True, model_path='model/model.pt') -> float:
    # we train a new model if there is no checkpoint given

    # n_iters: the total number of epochs to be iterated

    # save_epoch: save in which epoch
    # save_batch: save in which batch of the given epoch
    # save_path: the relative path of where the model will be saved, given as a string

    cur_epoch = 0
    # load previous model if there is checkpoint
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        cur_epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    # loss
    criterion = nn.CrossEntropyLoss()

    # training loop
    n_total_steps = len(training_loader)
    # keep track of the lowest perplexity
    lowest_perplexity = float('inf')

    for epoch in range(cur_epoch, n_iters): # the model has been trained over cur_epoch number of epochs
        # track fluctuations of perplexity and accuracy between two epoches
        # so as to adjust learning rate
        prev_running_perplexity = float('inf')
        running_perplexity = 0.00
        prev_running_accuracy = float('-inf')
        running_accuracy = 0.00

        for batch_idx, (src_win, tgt_win, tgt_lb) in enumerate(training_loader):

            # forward: computed predicted probability distribution
            output = model(src_win, tgt_win)

            # loss 
            loss = criterion(output, tgt_lb)
            # print(f"epoch {epoch+1}, batch {batch_idx}")
            # print(f"loss = {loss.item():.4f}")
            # print(f"perplexity = {torch.exp(loss):.4f}")

            # backward
            optimizer.zero_grad() # zero gradients
            loss.backward()
            # zero_grad before backward more readable???
            optimizer.step() # update weights
            
            # accuracy & perplexity
            running_accuracy += accuracy(output, tgt_lb)
            running_perplexity += torch.exp(loss).item()
            # print(f"batch {batch_idx}, running acc {running_accuracy}, running perp {running_perplexity}")

            if (batch_idx+1)%print_every_iter == 0:
                print(f"epoch {epoch+1} / {n_iters}, step {batch_idx+1} / {n_total_steps}, "
                    f"perplexity = {(running_perplexity/(batch_idx+1)):.4f}, "
                    f"accuracy = {(running_accuracy/(batch_idx+1)):.4f}")

        running_accuracy /= len(training_loader)
        running_perplexity /= len(training_loader)
        # update the lowest perplexity
        lowest_perplexity = min(lowest_perplexity, running_perplexity)
        print(f"End of epoch {epoch+1}, accuracy on the training set = {running_accuracy:.4f}, "
              f"perplexity on the training set {running_perplexity:.4f}")

        if adjust_lr:
            if ((running_accuracy-prev_running_accuracy < config.accuracy_threshold) 
                or (prev_running_perplexity-running_perplexity < config.perplexity_threshold)):
                # half learning rate
                for g in optimizer.param_groups:
                    g['lr'] /= 2

                print(f"epoch {epoch+1}")
                print(f"Change in accuracy: {running_accuracy-prev_running_accuracy}, "
                      f"change in perplexity: {prev_running_perplexity-running_perplexity}\n"
                      "Learning rate halved.")
        
        # update prev_perp and prev_accu
        prev_running_accuracy = running_accuracy
        prev_running_perplexity = running_perplexity
        running_accuracy = 0
        running_perplexity = 0
        
        if epoch%test_every_epoch == 0:
            test(criterion)
        
        if save:
            # save after each epoch
            checkpoint = {
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }
            torch.save(checkpoint, model_path+f"epoch_{epoch+1}.pt")
            print(f"Model saved at the end of epoch {epoch+1}.")

    print(f"Training ended, lowest perplexity during training = {lowest_perplexity}")

def print_layerinfo(model: torch.nn.Module) -> None:
    print("Print layer info:")
    for module in model.modules():
        if not isinstance(module, FeedForwardNet):
            print(module)
    print('\n')

def scorer(model, model_path, src_data, tgt_data, src_dict, tgt_dict) -> float:
    """
    Args:
        model_path: relative path of model
        src_data, tgt_data: list of list of strings
        src_dict, tgt_dict: Dictionary

    Returns:
        a dict with sentence pair as key and their log probability calculated by the given model as value
    """
    # load model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # prepare dataset to be used by the model
    score_dataset = dataset.TextDataset(src_data, tgt_data, src_dict, tgt_dict)

    score_loader = DataLoader(dataset=score_dataset, batch_size=1)

    sentence_idx = 0            # current sentence index
    sentence_log_prob = 0       # cumulative log-probabilitiy of a sentence
    scores = {}                 # an empty dict to store the computed probabilities 
    sum = 0
    # loss_function = nn.CrossEntropyLoss()
    # test(loss_function)

    # sets the model to evaluation mode to ensure consistent predictions
    # deactivates certain layers that are only needed for training and behave differently in evaluation mode
    model.eval()

    with torch.no_grad():   # make sure that no gradients are calculated

        for _, (src_win, tgt_win, tgt_lb) in enumerate(score_loader):
            # print(f"scr {src_win}")
            # print(f"tgt {tgt_win}")
            output = model(src_win, tgt_win)        # output of the model is the input for softmax to calculate the probabilities
            prob = torch.softmax(output, dim=-1)    # softmax computed for the last dimension
            # print(f"tgt_lb {tgt_lb}, argmax {torch.max(prob,-1)}")

            # calculate the prob of the tgt_lbl and add it to the cumulative sentence prob
            sentence_log_prob += math.log(prob[0][tgt_lb].item())
            # print(prob, prob.size())
            # print(prob, prob.size())
            # print(f"word prob {prob[0][tgt_lb].item()}")
            
            # at the and of a sentence create the strings and add them to the dict with their normalized prob
            if tgt_lb == config.index_sentence_end:     

                ## transform list of strings back to one string (= one sentence)
                src_string = ' '.join(src_data[sentence_idx])
                tgt_string = ' '.join(tgt_data[sentence_idx])
                # print(f"sentence {sentence_idx} prob: {math.exp(sentence_log_prob/len(tgt_data[sentence_idx]))}")

                ## save dict entry
                ## key is src and tgt sentence, value is normalized prob
                scores[(src_string, tgt_string)] = math.exp(sentence_log_prob/len(tgt_data[sentence_idx])) # length normalization

                if sentence_idx == len(tgt_data):
                    break

                ## in case we want to calculate the average score
                #sum += math.exp(sentence_log_prob/len(tgt_data[sentence_idx]))

                sentence_idx += 1
                sentence_log_prob = 0

    return scores

def checkpoints_bleu(model, model_path: str, src, tgt, align, src_dict, tgt_dict) -> dict:
    """

    Args:
        model: untrained
        model_path: path of checkpoint
        src: list of list of strings
        tgt: list of list of strings
        ...

    Returns: dict (bleus) with checkpoint name as key and BLEU as value

    """

    bleus = {}

    # get list of checkpoints
    # path leads file location
    checkpoints = os.listdir(model_path)

    for c in checkpoints:
        if c.endswith('.pt'):
            hypos = search.generate_translation(model, model_path+c, src, align, src_dict, tgt_dict)
            bleus[c] = bleu.bleu(tgt, hypos)

    return bleus



def checkpoints_ppl(model, model_path, dataloader, loss_function) -> dict:
    """

    Args:
        model: untrained
        model_path: path of checkpoint
        dataloader: loaded dataset containing dev src and dev tgt
        loss_function: to calculate the perplexity

    Returns: dict (ppls) with checkpoint name as key and average PPL as value

    """

    ppls = {}

    # gets list of checkpoints
    checkpoints = os.listdir(model_path)

    for c in checkpoints:
        if c.endswith('.pt'):

            # load model
            checkpoint = torch.load(model_path+c)
            model.load_state_dict(checkpoint['model_state_dict'])

            model.eval()
            with torch.no_grad():

                # initialize
                running_perplexity = 0
                loss = 0

                for _, (src_win, tgt_win, tgt_lb) in enumerate(dataloader):
                    
                    # predictions
                    output = model(src_win, tgt_win)

                    # calculate ppl
                    loss = loss_function(output, tgt_lb)

                    running_perplexity += torch.exp(loss).item()

                # calculates average ppl (running_perplexity / number batches (=tensors))
                running_perplexity /= len(dev_loader)
            
            ppls[c] = running_perplexity
    
    return ppls



if __name__ == '__main__':

    # device config: cuda if possible, else cpu as default
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # hyper-parameters from config
    win_size = config.window_size
    num_epochs = config.number_epochs
    embedding_dim = config.embedding_dimension
    hidden_size_1 = config.hidden_size_1
    hidden_size_2 = config.hidden_size_2
    learning_rate = config.learning_rate
    batch_size = config.batch_size
            
    ## load training and test data as list of lists of words
    training_src = tl.loadData('data/bpe_de_7000.txt')
    training_tgt = tl.loadData('data/bpe_en_7000.txt')
    dev_src = tl.loadData('data/bpe_7k_multi30k.dev.de')
    dev_tgt = tl.loadData('data/bpe_7k_multi30k.dev.en')

    # build dicts for source and target language from text data
    src_dict = Dictionary(training_src)
    tgt_dict = Dictionary(training_tgt)

    # preparing datasets
    training_dataset = dataset.TextDataset(training_src, training_tgt, src_dict, tgt_dict)
    training_loader = DataLoader(dataset=training_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 drop_last=True)

    dev_dataset = dataset.TextDataset(dev_src, dev_tgt, src_dict, tgt_dict)
    dev_loader = DataLoader(dataset=dev_dataset, 
                             batch_size=batch_size)

    ## trained model
    model_trained = FeedForwardNet(src_dict.n_words, tgt_dict.n_words, win_size, embedding_dim, hidden_size_1, hidden_size_2)
    # load model
    model_path = 'model/Adam_tanh2/epoch_4.pt'

    model_direct = 'model/Adam_tanh/random_seed_109/'
    checkpoint = torch.load(model_path)
    model_trained.load_state_dict(checkpoint['model_state_dict'])


    ## untrainiertes model
    model = FeedForwardNet(src_dict.n_words, tgt_dict.n_words, win_size, embedding_dim, hidden_size_1, hidden_size_2)



    ####################################################################################################################
    #
    # training the model
    #
    ####################################################################################################################

    # initialize loss
    loss = 0

    # sets the seed for generating random numbers
    torch.manual_seed(109)

    # choose optimizer as extensions of SGD
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)




    ####################################################################################################################
    # path to the location of the model
    ####################################################################################################################
    #model_save_path = 'model/RMSprop_batchnorm_signmoid/random_seed_73/epoch_7.pt'
    #model_save_path = 'model/Adam_tanh2/'

    # checkpoint = torch.load(model_path)
    # print(f"checkpoint epoch: {checkpoint['epoch']}")



    ####################################################################################################################
    # Option 1: load model
    ####################################################################################################################

    # checkpoint = torch.load(model_path)
    # train(model, optimizer, loss, num_epochs, checkpoints, print_every_iter=1000, test_every_epoch=1, adjust_lr=True, save=True, save_path=model_path)

    # if checkpoint given, then need to add checkpoint arguemnt (after n_iters) when calling the train() Funktion
    # checkpoint is a dict with epoch, model_state_dict, optimizer_state_dict, loss)



    ####################################################################################################################
    # Option 2: train from scratch (randomly initialized)
    ####################################################################################################################

    #train(model, optimizer, loss, num_epochs, print_every_iter=500, test_every_epoch=1, adjust_lr=True, save=True, model_path=model_save_path)



    ####################################################################################################################
    #
    # decoding
    #
    ####################################################################################################################

    dev_src_array = batch.text_to_array(dev_src, src_dict)

    align = search.alignment(training_src, training_tgt)

    ####################################################################################################################
    # decode only ONE sentence for test purpose
    ####################################################################################################################

    ## src
    #print(dev_src[800])
    #print(dev_src_array[800])

    ## ref
    #dev_tgt_array = batch.text_to_array(dev_tgt, tgt_dict)
    #print(dev_tgt[800])


    ## Beam direct
    #candidates = [list(cand) for cand in list(search.beam_search(model_trained, dev_src_array[800], align).keys())]
    #tgt_sentences = candidates[:1]
    #translation = batch.array_to_text(tgt_sentences, tgt_dict)


    #print(translation)


    # Greedy direct
    #print(batch.array_to_text([search.greedy_search(model_trained, dev_src_array[800], align)], tgt_dict))

    ####################################################################################################################
    # BLEU for corpus
    ####################################################################################################################

    #print(search.search(model, model_path, ' '.join(dev_src[0]), align, src_dict, tgt_dict, mode='beam', n_best=1, remove_bpe=True))

    #print(dev_src)
    #translation = search.generate_translation(model2, model_path, dev_src, align, src_dict, tgt_dict, mode='beam')

    #ref = dev_tgt

    #remove_special_symbols(translation)

    #print(translation)
    #print(bleu.bleu(ref, translation))

    #translation_unformatted = tl.unformat_text(translation)
    #print(translation_unformatted)
    #beam1 = beam_search.generate_translation(model, 'model/Adam_tanh/random_seed_73/epoch_4.pt', dev_src, align, src_dict, tgt_dict, mode='beam')


    #print(greedy)
    #print(beam1)
    #print(bleu.bleu(ref, greedy))
    #print(bleu.bleu(ref, beam1))
    # candidates_dict = beam_search.beam_search(model, dev_src_array[100], 15, 20, align)

    # candidates = [list(cand) for cand in list(candidates_dict.keys())]

    # print(batch.array_to_text(candidates, tgt_dict))

    #print(beam_search.search(model, model_path, ' '.join(dev_src[800]), align, src_dict, tgt_dict,
                    #    mode='beam', n_best = 5, remove_bpe=True))


    ref = BPE.bpe_reverse(dev_tgt)
    print(ref)

    #print(ref)

    #corpus_greedy = search.generate_translation(model, 'model/Adam_tanh2/epoch_7.pt', dev_src, align, src_dict,tgt_dict, mode='greedy')

    #corpus_beam = search.generate_translation(model, 'model/Adam_tanh2/epoch_4.pt', dev_src, align, src_dict, tgt_dict, mode='beam')



    #corpus_beam = remove_special_symbols(corpus_beam)
    #corpus_greedy = remove_special_symbols(corpus_greedy)

    #print(corpus_greedy)
    #print(corpus_beam)


    #print(bleu.bleu(ref, corpus_greedy))
    #print(bleu.bleu(ref, corpus_beam))

    #translation_unformatted = tl.unformat_text(corpus_beam)
    #print(translation_unformatted)


    ####################################################################################################################
    #
    # scoring
    #
    ####################################################################################################################

    #greedy = beam_search.generate_translation(model2, model_path, dev_src, align, src_dict, tgt_dict, mode='greedy')
    #print(greedy)
    #beam = beam_search.generate_translation(model2, model_path, dev_src, align, src_dict, tgt_dict, mode='beam')
    #print(dev_tgt)

    #score_ref = scorer(model, model_path, dev_src, dev_tgt, src_dict, tgt_dict)
    #print(score_ref)
    #score_hyp = scorer(model, model_path, dev_src, beam, src_dict, tgt_dict)
    #print(score_hyp)
    #print (f,e) with scores line by line
    #print(f"scores for {checkpoint_path}")
    # [print(key,':',value) for key, value in scores.items()]


    ####################################################################################################################
    # BLEU for each epoch
    ####################################################################################################################

    #bleus = checkpoints_bleu(model, model_direct, dev_src, dev_tgt, align, src_dict, tgt_dict)
    #print(f"BLEU for {model_direct}: {bleus}")

    ####################################################################################################################
    # Perplexity for each epoch
    ####################################################################################################################

    #ppls = checkpoints_ppl(model, model_direct, dev_loader, nn.CrossEntropyLoss())
    #print(f"PPL for {model_direct}: {ppls}")


    ####################################################################################################################
    #
    # TODOs
    #
    ####################################################################################################################


    # TODO loss function in config?
    # TODO: bei beam size = 3 (?) ist ein Punkt in der Mitte des ersten Satzes --> sollte nicht so sein
