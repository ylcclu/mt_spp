import math, sys
import config
import levenshtein
import textloader as tl

def generate_ngrams(words: list[str], n: int) -> list[list[str]]:
    """
    Args:
        words: a given string
        n: required length of the word sequences

    Returns:
        list of n-grams contained in sentence
    """
    
    ngrams = []

    # len(words)-n+1 possible starting positions for n-grams
    for i in range(len(words)-n+1):
        ngrams.append(words[i:i+n])
    return ngrams

def count_ngram_matches(reference: list[str], hypothesis: list[str], n: int) -> int:
    """Counts the number of matching n-grams between a hypothesis and a reference string

    Args:
        hypothesis: proposed translation by the system
        reference: translation in target language produced by human experts
        n: required length of the word sequences (n-gram) that need to match

    Returns:
        number of matching n-grams between hypothesis and reference
    """

    hypo_ngrams = generate_ngrams(hypothesis, n)
    reference_ngrams = generate_ngrams(reference, n)
    common_ngrams = []

    # Create a copy of reference n-grams list
    ref_ngrams_copy = reference_ngrams

    # iterate over each n-gram in the hypothesis
    for hypo_ngram in hypo_ngrams:
        # add common n-grams in the list
        if hypo_ngram in ref_ngrams_copy:
            common_ngrams.append(hypo_ngram)
            # Remove matched n-gram from reference n-grams list in order to avoid duplicates
            ref_ngrams_copy.remove(hypo_ngram)            

    # return the count of matching n-grams
    return len(common_ngrams)

def modified_ngram_precision(refs: list[list[str]], hypos: list[list[str]],
                             n: int) -> float:
    """
    Args:
        refs: given set of references
        hypos: given set of hypotheses
        n: required length of the word sequences (n-gram)

    Returns:
        modified n-gram precision
    """

    sum_min_ngrams_count = 0
    hypo_ngrams_count = 0

    for ref,hypo in zip(refs,hypos): # bleu() made sure len(refs) == len(hypos)
        # number of matching n-grams (correctly predicted) between r_l and h_l
        sum_min_ngrams_count += count_ngram_matches(ref, hypo, n)
        
        # numebr of n-grams in h_l
        hypo_ngrams_count += len(generate_ngrams(hypo, n))

    return sum_min_ngrams_count/hypo_ngrams_count

def brevity_penalty(c: int, r: int) -> float:
    """
    Args:
        c: length of the hypothesis
        r: length of the reference

    Returns:
        brevity penalty of the given hypothesis and reference
    """

    if c > r:
        return 1
    else:
        return math.exp(1-r/c)
    
def bleu(refs: list[list[str]], hypos: list[list[str]]) -> float:
    """
    Args:
        refs: given set of references
        hypos: given set of hypotheses
        n: required length of the word sequences (n-gram)

    Returns:
        BLEU for the given set of reference-hypothesis-pairs
    """
    
    if len(refs) != len(hypos):
        return -1 # invalid input when ref and hypo not in pairs
    else:
        c = sum([len(x) for x in hypos]) # length of hypothesis
        r = sum([len(x) for x in refs]) # length of reference

        # the exponent is the sum from 1 to bleu_big_n (see globals.py)
        percision_sum = 0

        bp = brevity_penalty(c,r)

        for i in range(config.bleu_big_n):
            p_n = modified_ngram_precision(refs, hypos,i+1)
            if p_n == 0:
                return 0
            else: 
                percision_sum += math.log(p_n)

        return bp*math.exp(percision_sum/config.bleu_big_n)

def wer(refs: list[list[str]], hypos: list[list[str]]) -> float:
    """calculates Word Error Rate (WER) for a given set of references and hypotheses

    Args:
        refs: given set of references
        hypos: given set of hypotheses

    Returns:
        WER as Levenshtein Distance over reference length
    """
    if len(refs) != len(hypos):
        return -1 # invalid input when ref and hypo not in pairs
    else:
        # calculate total reference length
        ref_length = sum([len(x) for x in refs])
        # calculate sum of all Levenshtein Distance
        sum_levenshtein_dist = 0
        for ref,hypo in zip(refs,hypos):
            sum_levenshtein_dist += levenshtein.levenshtein_without_change(ref,hypo)

    return  sum_levenshtein_dist / ref_length

def per(refs: list[list[str]], hypos: list[list[str]]) -> float:
    """
    calculates Position-independent Error Rate (PER) 
    for a given set of references and hypotheses

    Args:
        refs: given set of references
        hypos: given set of hypotheses

    Returns:
        PER
    """
    if len(refs) != len(hypos):
        return -1 # invalid input when ref and hypo not in pairs
    else:
        # calculate total reference / hypothesis length
        ref_length = sum([len(x) for x in refs])
        hypo_length = sum([len(x) for x in hypos])

        # calculate total number of matching 1-grams
        matches = sum([count_ngram_matches(ref,hypo,1) for ref,hypo in zip(refs,hypos)])

        return 1 - (matches - max(0, hypo_length-ref_length))/ref_length
    

if __name__ == '__main__':
    # first argument is path from working directory to the file containing the list of refs
    # second is path from working directory to the file containing the list of hypos
    refs_rel_path, hypos_rel_path = sys.argv[1], sys.argv[2]

    # load and turn refs and hypos into workable format
    refs_words = tl.loadData(refs_rel_path, mode="words")
    hypos_words = tl.loadData(hypos_rel_path, mode="words")
    # refs_lines = tl.loadData(refs_rel_path, mode="lines")
    # hypos_lines = tl.loadData(hypos_rel_path, mode="lines")

    print(f"BLEU: {bleu(refs_words, hypos_words)}")
    print(f"WER: {wer(refs_words,hypos_words)}")
    print(f"PER: {per(refs_words,hypos_words)}")