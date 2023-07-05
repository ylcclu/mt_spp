import config
import textloader as dl
import re, collections
import pandas as pd
import json, pickle
import time, gzip, sys
from config import bpe_num_ops

class BPE:

    def build_corpus_vocab(trainingData: list[list[str]]) -> dict[str:int]:
        """
        Args:
            trainingData: a list of sentences (which are in turn lists of words)

        Returns:
            - a dictionary with word (with '@@ ' in between chars) as key 
              and its frequency in trainingData as value
        """
        # initialize as an empty dict with int as default value
        corpus = collections.defaultdict(int)
        for sentence in trainingData:
            for word in sentence:
                # add '@@' as in-word-symbol to each word
                # e.g. 'word' => 'w@@ o@@ r@@ d'
                word = '@@ '.join(word)
                corpus[word] += 1
                
        return corpus

    def get_pair_freqs(corpus: dict) -> dict:
        """
        Args:
            corpus: a dictionary with word (a string with '@@ ' in between learned subwords) as key 
                    and its frequency in trainingData as value

        Returns:
            a dictionary with a pair of adjacent subwords as key and its frequency in corpus as value
        """
        # initialize as an empty dict, with default value of type int
        pairs = collections.defaultdict(int)
        # iterate over all words in corpus
        for word, freq in corpus.items():
            symbols = word.split() # a list of subwords
            # iterate over all pairs of subwords in word
            for i in range(len(symbols)-1):
                # update frequency of subword pairs
                pairs[symbols[i],symbols[i+1]] += freq
        return pairs
    
    def merge_pair(pair: tuple[str, str], c_in: dict) -> dict:
        """apply the merging of a given pair of subwords on a corpus

        Args:
            pair: a pair of adjacent subwords
            c_in: a dictionary with word (a string with '@@ ' in between learned subwords) as key 
                  and its frequency in trainingData as value

        Returns:
            new dictionary with word (a string with '@@ ' in between learned subwords) as key 
            and its frequency in trainingData as value
        """
        c_out = collections.defaultdict(int)
        bigram = re.escape(' '.join(pair))
        # compile a regex pattern into a regex object for matching
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        # (?<!...) Matches if the current position in the string is not preceded by a match for ...
        # \S Matches any character which is not a whitespace character
        # (?!...) Matches if ... doesn’t match next
        # see https://docs.python.org/3/library/re.html
        for word in c_in:
            # replace substring of form 'f@@ g' with 'fg', if pair=('f@@','g')
            w_out = p.sub(''.join(pair).replace('@@', '', 1), word)
            c_out[w_out] = c_in[word]
        return c_out

    def bpe_training(num_merges: int, trainingData: list[list[str]]) -> dict[(str,str):str]:
        """learn subword splitting from given text

        Args:
            num_merges: number of BPE merge operations
            trainingData: a list of sentences (which are in turn lists of words),
                          data set with which 

        Returns:
            . a dict of merge operations with pairs as key and the merged string (learned vocab) as value
        """
        # corpus initailized as a dict of words and their frequencies in training data
        corpus= BPE.build_corpus_vocab(trainingData)
        # merges initalized as a dict with str as default value
        merges = collections.defaultdict(str)
        
        for _ in range(num_merges):
            # calculate frequencies of adjacent subwords
            pairs = BPE.get_pair_freqs(corpus)
            if pairs:
                # find the most frequent pair
                best = max(pairs, key=pairs.get)
                # apply merge on the most frequent pair on the dict of words and their frequencies
                corpus = BPE.merge_pair(best, corpus)
                # keep track of the merge operations
                best_merged = ''.join(best).replace('@@', '', 1)
                merges[best] = best_merged

        return merges
    
    def tokenize(text: list[list[str]], merges: dict[(str,str):str]) -> list[list[str]]:
        """apply the learned BPE merge on a new text (data disjoint with the training data)

        Args:
            text: a new text unseen before as a list of sentences (which are in turn lists of words)
            merges: a dict of merge operations with pairs as key and the merged string as value

        Returns:
            a list of sentences (which are in turn lists of subwords learned through BPE algorithm)
        """
        # pretokenize: split words into subwords consisting of exactly one char
        # a sentence is now a list of lists of subwords of the form 'a@@' or 'a'
        splitted_text = [['@@ '.join(word).split() for word in sentence] for sentence in text]
        # iterate over all sentences
        for sentence in splitted_text:
            # iterate over all merge operations
            for pair, merge in merges.items():
                # iterate over all words (list of subwords) in sentence
                for idx, splitted_word in enumerate(sentence):
                    # idx = index of the word in sentence
                    # splitted_word = list of subwords
                    i = 0
                    while i < len(splitted_word) - 1:
                        # look at all pairs of adjacent subwords
                        if splitted_word[i] == pair[0] and splitted_word[i + 1] == pair[1]:
                            # if matched, merge the two subwords
                            splitted_word = splitted_word[:i] + [merge] + splitted_word[i+2:]
                        else:
                            i += 1
                    sentence[idx] = splitted_word
        
        # transform sentence from list of lists of subwords into a list of subwords
        subwords_applied = [sum(sentence, []) for sentence in splitted_text]
        return subwords_applied
    
    def bpe_reverse(text: list[list[str]]) -> list[list[str]]:
        """reverse BPE subword splitting on a given text

        Args:
            text: a list of sentences (which are in turn lists of subwords learned through BPE algorithm)

        Returns:
            a list of sentences (which are in turn lists of whole words)
        """
        for idx, sentence in enumerate(text):
            i = 0
            while i < len(sentence) - 1:
                if sentence[i].endswith('@@'):
                    # if a subword ends with '@@', merge it with its successor in sentence
                    merged_subword = ''.join((sentence[i],sentence[i+1])).replace('@@', '', 1)
                    sentence = sentence[:i] + [merged_subword] + sentence[i+2:]
                else:
                    i += 1
            # update each sentence
            text[idx] = sentence
        return text
    
    def vocab_from_bpe(text: list[list[str]], merges: dict[(str,str):str]) -> list[str]:
        """compute vocabulary of a text (not yet tokenized, i.e. on which BPE operations has not yet been applied)

        Args:
            text: a new text unseen before as a list of sentences (which are in turn lists of words)
            merges: a dict of merge operations with pairs as key and the merged string as value

        Returns:
            a list of unique subwords in text
        """
        return dl.unique_and_sort(BPE.tokenize(text, merges))


if __name__ == '__main__':
    # num_ops set in config.py
    num_ops = bpe_num_ops
    # relative path of the file as first argument
    file_path = sys.argv[1]
    data = dl.loadData(file_path)
    # optional second file
    if len(sys.argv) == 3:
        file_path2 = sys.argv[2]
        data2 = dl.loadData(file_path2)
        data.extend(data2)

    merges = BPE.bpe_training(num_ops, data)
    tokenized_text_as_list = BPE.tokenize(data2, merges)
    # print(tokenized_text_as_list)
    # print(len(dl.unique_and_sort(tokenized_text_as_list)))
    tokenized_text = dl.unformat_text(tokenized_text_as_list)
    print(tokenized_text)

    # target_text = dl.loadData('./blatt2/multi30k.en.gz')
    # source_text = dl.loadData('./blatt2/multi30k.de.gz')
    # source_vocab, merges = BPE.bpe_training(100, source_text)
    # source_vocab, merges = BPE.bpe_training(15000, source_text)
    # print(merges)
    # print(BPE.tokenize(dl.loadData_of_specific_lines('./blatt2/multi30k.de.gz', 1, 3), merges))

    # text = [['zwei', 'junge', 'w@@', 'ei@@', 'ß@@', 'e', 'mä@@', 'n@@', 'n@@', 'er', 'si@@', 'n@@', 'd', 'im', 'f@@', 'r@@', 'ei@@', 'en', 'in', 'der', 'n@@', 'ä@@', 'h@@', 'e', 'v@@', 'iel@@', 'er', 'b@@', 'ü@@', 'sch@@', 'e', '.'], 
    #         ['m@@', 'e@@', 'hr@@', 'er@@', 'e', 'mä@@', 'n@@', 'n@@', 'er', 'mit', 'sch@@', 'u@@', 'tz@@', 'h@@', 'el@@', 'm@@', 'en', 'be@@', 'di@@', 'en@@', 'en', 'ein', 'an@@', 'tr@@', 'i@@', 'eb@@', 's@@', 'ra@@', 'd@@', 's@@', 'y@@', 'st@@', 'em', '.'], 
    #         ['ein', 'kl@@', 'ein@@', 'es', 'mädchen', 'kl@@', 'e@@', 't@@', 'ter@@', 't', 'in', 'ein', 'spiel@@', 'h@@', 'aus', 'aus', 'ho@@', 'l@@', 'z', '.']]
    # print(BPE.bpe_reverse(text))