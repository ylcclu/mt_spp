import gzip, sys
from pathlib import Path

def readData(rel_path: str) -> str:
    """reads data from file

    Args:
        rel_path: relative path of the file

    Returns:
        content of the file as one string
    """
    abs_path = Path(__file__).parent.resolve() / rel_path

    if rel_path.endswith('gz'):
        file = gzip.open(abs_path, 'rt', encoding="utf-8")
    else:
        file = open(abs_path,'r', encoding='utf-8')

    data_read = file.read()
    file.close()

    return data_read

def loadData(rel_path: str) -> list[list[str]]:
    """reads training data from file and turns the whole file into a list of lists of words

    Args:
        rel_path: relative path of the file

    Returns:
        a list of sentences (sentences are lists of strings)
    """
    # total_length = 0
    data_read = readData(rel_path)

    # generate a list of strings by splitting over lines
    list_of_sentences = data_read.split('\n')

    # split each sentence (string) into a list of strings
    list_of_words = [sentence.split() for sentence in list_of_sentences if sentence]
    # for sentence in list_of_words:
    #     total_length += len(sentence)
    # print(total_length)
    # print(total_length + len(list_of_words))
    return list_of_words

def loadData_of_specific_lines(rel_path: str, line_start: int, line_end: int) -> list[list[str]]:
    """reads training data from file and turn selected consecutive lines into a list of lists of words

    Args:
        rel_path: relative path of the file
        line_start: the first sentence to be read
        line_end: the last sentence to be read

    Returns:
        a list of sentences (which in turn are lists of words)
    """
    data_read = readData(rel_path)

    # generate a list of strings by splitting over lines
    lines = data_read.split('\n')
    # select specific lines
    selected_lines = lines[line_start-1:line_end]
    # split each sentence (string) into a list of strings
    list_of_words = [sentence.split(' ') for sentence in selected_lines]
    # slicing to delete empty string at the end of each sentence

    return list_of_words

def unique_and_sort(trainingData: list[list[str]]) -> list[str]:
    """delete duplicates and sort in alphabetical order

    Args:
        trainingData: a list of sentences (which in turn are lists of words)

    Returns:
        a list of sorted words without duplication
    """
    words = []
    for sentence in trainingData:
        # sentence is a list of strings
        words.extend(sentence)

    # words is the result of all the lists (sentence) joined together
    unique_sorted = list(set(words))
    unique_sorted.sort()
    if '' == unique_sorted[0]:
        # empty string is not a word
        unique_sorted = unique_sorted[1:]

    # print(list(enumerate(unique_sorted)))    
    return unique_sorted

def unformat_text(ls: list[list[str]]) -> str:
    """
    Args:
        text_as_list: text as a list of sentences (which in turn are lists of words)

    Returns:
        a single string with words seperated by ' ' and sentences seperated by '\n'
    """
    return '\n'.join([' '.join(sentence) for sentence in ls])




def size_of_vocab(rel_path: str) -> int:

    list_of_words = loadData(rel_path)

    return len(unique_and_sort(list_of_words))

if __name__ == '__main__':
    rel_path = sys.argv[1]
    # print(size_of_vocab(rel_path))
    loadData(rel_path)
    # print(loadData_of_specific_lines('./blatt2/multi30k.en.gz', 2, 3))