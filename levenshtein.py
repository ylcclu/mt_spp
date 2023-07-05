#! /usr/bin python3
import sys
import numpy as np

def levenshteinDistance(s: list[str], t: list[str], d: list[list[int]]) -> int:
    """
    Args:
        s: source text
        t: target text
        d: the Levenshtein matrix initialized as a zero matrix

    Returns:
        Levenshtein Distance between s and t
    """
    if len(s)+1 != len(d[0]) or len(t)+1 != len(d):
        # in case of input of incorrect format
        print(f"len(d)={len(d)}, len(s)={len(s)}, len(d[0])={len(d[0])}, len(t)={len(t)}")
        return -1
    else:
        m,n = len(s),len(t)

        # to transform source prefixes into empty string,
        # drop all characters
        for j in range(1,m+1):
            d[0][j] = j

        # to tranform empty string into target prefixes,
        # insert every character in the target prefix
        for i in range(1,n+1):
            d[i][0] = i
        
        # calculate the rest of the matrix
        for i in range(1,n+1):
            for j in range(1,m+1):
                subsituteCost = 0
                if s[j-1] != t[i-1]:
                    # if last characters do not match, add 1 substitution cost
                    subsituteCost = 1
                
                d[i][j] = min(d[i-1][j-1]+subsituteCost, # Match or Substitution
                              d[i][j-1]+1,               # Deletion
                              d[i-1][j]+1)               # Insertion
        # print(np.matrix(d))
        return d[n][m]


def backtrackChange(s: list[str], t: list[str], d: list[list[int]]) -> str:
    """Finds a combination of operations needed to turn s into t
       by decoding the Levenshtein matrix for s and t using backtracking

    Args:
        s: source text
        t: target text
        d: the calculated Levenshtein matrix

    Returns:
        a string of operations to turn s into t
    """
    change = ""
    # start from bottom right
    i,j=len(d)-1,len(d[0])-1
    while i>0 or j>0:
        diagonal = d[i-1][j-1] # Match or Substitution
        above = d[i-1][j]      # Insertion
        left = d[i][j-1]       # Deletion
        current = d[i][j]
        if (diagonal <= min(left,above) and # diagonal <= the other two
            (diagonal == current or diagonal == current-1)):
            if diagonal == current-1: # substitution
                change = f"- substitute {s[j-1]} with {t[i-1]}\n" + change
            else: # no op
                change = "- no operation\n" + change
            # move to diagonal
            i-=1
            j-=1
        elif (left <= above and # left <= above: deletion
              left == current-1):
            change = f"- delete {s[j-1]}\n" + change
            # move left
            j-=1
        else: # insertion
            change = f"- insert {t[i-1]}\n" + change
            # move up
            i-=1    
    return change

def format_input(s,t) -> tuple[list[str], list[str]]:
    # turn s and t into workable format: lists of strings
    if type(s) == str:
        s = [x for x in s]
    if type(t) == str:
        t = [x for x in t]

    return s, t

def levenshtein_with_change(s: list[str], t: list[str]) -> tuple[int,str,list[list[int]]]:
    format_input(s,t)
    # initialize the distance matrix d: a zero matrix of dimension (m+1)x(n+1)
    d = [[0 for _ in range(len(s)+1)] for _ in range(len(t)+1)]
    # calculate the Levenshtein Distance between
    lev_dist = levenshteinDistance(s,t,d)
    change = backtrackChange(s,t,d)

    return lev_dist, change, d

def levenshtein_without_change(s: list[str], t: list[str]) -> int:
    format_input(s,t)
    # initialize the distance matrix d: a zero matrix of dimension (m+1)x(n+1)
    d = [[0 for _ in range(len(s)+1)] for _ in range(len(t)+1)]
    # calculate the Levenshtein Distance between
    return levenshteinDistance(s,t,d)

if __name__ == '__main__':
    # two words are given as arguments in command line
    s,t = sys.argv[1],sys.argv[2]
    # s - source text of length m
    # t - target text of length n

    lev_dist, change, d = levenshtein_with_change(s,t)
    if lev_dist >= 0:
        print("The Levenshtein Distance between "
              f"{s} and {t} is {lev_dist}.")
        print("The Levenshtein Matrix:\n",np.matrix(d))
        print(f"Changes needed:\n{change}")
