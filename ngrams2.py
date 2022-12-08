from collections import Counter, defaultdict

from typing import Iterable, TypeVar, Sequence

# DO NOT MODIFY
T = TypeVar("T")

# DO NOT MODIFY
START_TOKEN = "<start>"
# DO NOT MODIFY
END_TOKEN = "<end>"


def counts_to_probs(counts: Counter[T]) -> defaultdict[T, float]:
    """Return a defaultdict with the input counts converted to probabilities."""
    # get total value by iterating over the counter values

    # go over counter again and calculate probabilities
    total: int = counts.total()
    probabilities_dict: defaultdict = defaultdict(float)
    for key, value in probabilities_dict.items():
        probabilities_dict[key] = value / total
    return probabilities_dict


def bigrams(sentence: Sequence[str]) -> list[tuple[str, str]]:
    """Return the bigrams contained in a sequence."""
    padded_sentence = [START_TOKEN] + list(sentence) + [END_TOKEN]
    print(padded_sentence)

    return list(zip(padded_sentence, padded_sentence[1:]))


if __name__ == '__main__':
    print(bigrams("This is a sentence"))

def trigrams(sentence: Sequence[str]) -> list[tuple[str, str, str]]:
    """Return the trigrams contained in a sequence."""
    padded_sentence = [START_TOKEN] * 2 + list(sentence) + [END_TOKEN] * 2
    tri_list: list = list()
    size: int = len(sentence)
    for i in range(0, size + 2):
        tri_list.append((padded_sentence[i], padded_sentence[i + 1], padded_sentence[i + 2]))
    return tri_list


def count_unigrams(sentences: Iterable[list[str]], lower: bool = False) -> Counter[str]:
    """Count the unigrams in an iterable of sentences, optionally lowercasing."""
    counter: Counter = Counter()
    for sentence in sentences:
        for token in sentence:
            counter[token] += 1
    return counter


def count_bigrams(
        sentences: Iterable[list[str]], lower: bool = False) -> Counter[tuple[str, str]]:
    """Count the bigrams in an iterable of sentences, optionally lowercasing."""
    counter: Counter = Counter()
    for sentence in sentences:
        if lower:
            for key, value in enumerate(sentence):
                sentence[key] = value.lower()
        sentence_trigrams: list[tuple[str, str]] = bigrams(sentence)
        for trigram in sentence_trigrams:
            counter.update([trigram])
    return counter


def count_trigrams(
        sentences: Iterable[list[str]], lower: bool = False
) -> Counter[tuple[str, str, str]]:
    """Count the trigrams in an iterable of sentences, optionally lowercasing."""
    counter: Counter = Counter()
    for sentence in sentences:
        if lower:
            for key, value in enumerate(sentence):
                sentence[key] = value.lower()
        sentence_trigrams: list[tuple[str, str, str]] = trigrams(sentence)
        for trigram in sentence_trigrams:
            counter.update([trigram])
    return counter
