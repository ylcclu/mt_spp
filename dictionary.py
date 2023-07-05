import config
import textloader as dl


class Dictionary:
    word2index = {}
    index2word = {}

    def __init__(self, trainingData: list[list[str]]) -> None:
        """
        Args:
            trainingData: a list of sentences (which are in turn lists of words)
        """
        # generate a sorted, duplicate-free list of words
        list_of_unique_words = dl.unique_and_sort(trainingData)

        # add words in training data to dict
        self.word2index = {list_of_unique_words[i]: i + 2 for i in range(len(list_of_unique_words))}
        # add start and end of text symbol to dict
        self.word2index['<s>'] = 0
        self.word2index['</s>'] = 1
        self.word2index['<UNK>'] = 2
        # print(self.word2index)

        # realize bidirection through inversed list
        self.index2word = {value: key for (key, value) in self.word2index.items()}
        self.n_words = len(self.word2index)

    def get_index(self, word: str) -> int:
        """
        Args:
            word: a given token
            dictionary: a give dictionary

        Returns:
            index of the token in the dictionary
        """
        try:
            return self.word2index[word]
        except:
            # 2 for <UNK>
            return 2

    def get_word(self, index: int) -> str:
        """
        Args:
            index: a given index of a token in the dictionary
            dictionary: a give dictionary

        Returns:
            the word correpsonding to the index in the dictionary
        """
        try:
            return self.index2word[index]
        except:
            # necessary?????????
            return "<UNK>"

    def add_vocab(self, traningData: list[list[str]]) -> None:
        """update the dictionary given new training data

        Args:
            traningData: a new list of sentences (which are in turn lists of words)
        """
        ### not yet tested

        # generate a sorted, duplicate-free list of words
        list_of_unique_words = dl.unique_and_sort(traningData)

        # generate a dict of words not seen in the old dict
        new_dict = {list_of_unique_words[i]: i for i in
                    range(len(self.word2index), len(list_of_unique_words) + len(self.word2index))}
        new_dict_reversed = {value: key for (key, value) in new_dict.items()}

        # add new words to the old dict
        self.word2index.update(new_dict)
        self.index2word.update(new_dict_reversed)
        self.n_words += len(new_dict)


def text_to_array(text: list[list[str]], dict: Dictionary) -> list[list[int]]:
    list_of_indices = []

    for sentence in text:
        sentence_as_indices = []
        for word in sentence:
            sentence_as_indices.append(dict.get_index(word))
        list_of_indices.append(sentence_as_indices)

    return list_of_indices


def array_to_text(arrays: list[list[int]], dict: Dictionary) -> list[list[str]]:
    list_of_words = []

    for array in arrays:
        sentence = []
        for idx in array:
            sentence.append(dict.get_word(idx))
        list_of_words.append(sentence)

    return list_of_words


def remove_special_symbols(ls: list[list[str]]) -> list[list[str]]:
    for sentence in ls:
        sentence[:] = [word for word in sentence if
                       word != config.unknown and word != config.sentence_start and word != config.sentence_end]
    return ls




if __name__ == '__main__':
    # load from data processed by BPE => change file name!!!!!!!!!!!!!!
    training_data = dl.loadData("./data/multi30k.en.gz")
    dictionary = Dictionary(training_data)
    # print(dictionary.word2index)
    # print(dictionary.get_index("."))
    print(dictionary.get_word(40))
