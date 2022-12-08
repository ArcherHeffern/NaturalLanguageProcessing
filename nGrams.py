import collections
import random
from collections import defaultdict, Counter
from math import log
from typing import Sequence, Iterable, Generator, TypeVar

# hw2.py
# Version 1.1
# 9/26/2022

############################################################
# The following constants and function are provided as helpers.
# Do not modify them! The stubs for what you need to implement are later in the file.

# DO NOT MODIFY
random.seed(0)

# DO NOT MODIFY
START_TOKEN = "<start>"
# DO NOT MODIFY
END_TOKEN = "<end>"
# DO NOT MODIFY
NEG_INF = float("-inf")
# DO NOT MODIFY (needed if you copy code from HW 1)
T = TypeVar("T")


# DO NOT MODIFY
def load_tokenized_file(path: str) -> Generator[Sequence[str], None, None]:
    """Yield sentences as sequences of tokens."""
    with open(path, encoding="utf8") as file:
        for line in file:
            line = line.rstrip("\n")
            tokens = line.split(" ")
            yield tuple(tokens)


# DO NOT MODIFY
def sample(probs: dict[str, float]) -> str:
    """Return a sample from a distribution."""
    # To avoid relying on the dictionary iteration order, sort items
    # This is very slow and should be avoided in general, but we do
    # it in order to get predictable results
    items = sorted(probs.items())
    # Now split them back up into keys and values
    keys, vals = zip(*items)
    # Choose using the weights in the values
    return random.choices(keys, weights=vals)[0]


############################################################
# The stubs below this are the ones that you should fill in.
# Do not modify anything above this line other than to add any needed imports.


def bigram_probs(
    sentences: Iterable[Sequence[str]],
) -> dict[str, dict[str, float]]:
    """Return bigram probabilities computed from the provided sequences."""
    counts_dict: defaultdict[str, Counter] = defaultdict(Counter)
    for document in sentences:
        document = [START_TOKEN] + list(document) + [END_TOKEN]
        for i in range(len(document) - 1):
            bigram: tuple[str, str] = (document[i], document[i + 1])
            counts_dict[bigram[0]].update({bigram[1]})
    counts_to_probs(counts_dict)
    return convert_to_dict(counts_dict)


def trigram_probs(
    sentences: Iterable[Sequence[str]],
) -> dict[tuple[str, str], dict[str, float]]:
    """Return trigram probabilities computed from the provided sequences."""
    counts_dict: defaultdict[tuple[str, str], Counter] = defaultdict(Counter)
    for document in sentences:
        document = 2*[START_TOKEN] + list(document) + 2*[END_TOKEN]
        for i in range(len(document) - 2):
            trigram: tuple[str, str, str] = (document[i], document[i + 1], document[i + 2])
            counts_dict[(trigram[0], trigram[1])].update({trigram[2]})
    counts_to_probs(counts_dict)
    return convert_to_dict(counts_dict)


def counts_to_probs(probabilities_dict):
    for counter in probabilities_dict.values():
        total: float = counter.total()
        for element in counter:
            counter[element] /= total


def convert_to_dict(d):
    if isinstance(d, defaultdict) or isinstance(d, Counter):
        d = {k: convert_to_dict(v) for k, v in d.items()}
    return d


def sample_bigrams(probs: dict[str, dict[str, float]]) -> list[str]:
    """Generate a sequence by sampling from the provided bigram probabilities."""
    bigram = list()
    context = START_TOKEN
    while context != END_TOKEN:
        context = sample(probs[context])
        bigram.append(context)
    bigram.pop()
    return bigram


def sample_trigrams(probs: dict[tuple[str, str], dict[str, float]]) -> list[str]:
    """Generate a sequence by sampling from the provided trigram probabilities."""
    trigram = list()
    context = (START_TOKEN, START_TOKEN)
    while context[1] != END_TOKEN:
        value = sample(probs[context])
        context = (context[1], value)
        trigram.append(value)
    trigram.pop()
    return trigram


def bigram_sequence_prob(
    sequence: Sequence[str], probs: dict[str, dict[str, float]]
) -> float:
    """Compute the probability of a sequence using bigram probabilities."""

    #list of tokens
    sequence = [START_TOKEN] + list(sequence)
    net_probability = 0
    for i in range(len(sequence) - 1):
        token = sequence[i]
        if token not in probs.keys():
            return NEG_INF
        probability = probs[token][sequence[i + 1]]
        if probability == 0:
            return NEG_INF
        net_probability += log(probability)
    return net_probability


    # for element in sequence -> get probability of value
    # if probability == 0 or context not in the distribution return NEG_INF
    # probability += log value


def trigram_sequence_prob(
    sequence: Sequence[str], probs: dict[tuple[str, str], dict[str, float]]
) -> float:
    """Compute the probability of a sequence using trigram probabilities."""
    sequence = 2*[START_TOKEN] + list(sequence)
    net_probability = 0
    for i in range(len(sequence) - 2):
        token:tuple[str, str] = (sequence[i], sequence[i + 1])
        if token not in probs.keys():
            return NEG_INF
        probability = probs[token][sequence[i + 2]]
        if probability == 0:
            return NEG_INF
        net_probability += log(probability)
    return net_probability
