from abc import abstractmethod, ABC
from collections import Counter
from collections import defaultdict
from math import log
from operator import itemgetter
from typing import Generator, Iterable, Sequence

############################################################
# The following constants, classes, and function are provided as helpers.
# Do not modify them! The stubs for what you need to implement are later in the file.

# DO NOT MODIFY
NEG_INF = float("-inf")


# DO NOT MODIFY
class TaggedToken:
    """Store the text and tag for a token."""

    # DO NOT MODIFY
    def __init__(self, text: str, tag: str):
        self.text: str = text
        self.tag: str = tag

    # DO NOT MODIFY
    def __str__(self) -> str:
        return f"{self.text}/{self.tag}"

    # DO NOT MODIFY
    def __repr__(self) -> str:
        return f"<TaggedToken {str(self)}>"

    # DO NOT MODIFY
    @classmethod
    def from_string(cls, s: str) -> "TaggedToken":
        """Create a TaggedToken from a string with the format "token/tag".

        While the tests use this, you do not need to.
        """
        splits = s.rsplit("/", 1)
        assert len(splits) == 2, f"Could not parse token: {repr(s)}"
        return cls(splits[0], splits[1])


# DO NOT MODIFY
class Tagger(ABC):
    # DO NOT IMPLEMENT THIS METHOD HERE
    @abstractmethod
    def train(self, sentences: Iterable[Sequence[TaggedToken]]) -> None:
        """Train the part of speech tagger by collecting needed counts from sentences."""
        raise NotImplementedError

    # DO NOT IMPLEMENT THIS METHOD HERE
    @abstractmethod
    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        """Tag a sentence with part of speech tags."""
        raise NotImplementedError

    # DO NOT MODIFY
    def tag_sentences(
        self, sentences: Iterable[Sequence[str]]
    ) -> Generator[list[str], None, None]:
        """Yield a list of tags for each sentence in the input."""
        for sentence in sentences:
            yield self.tag_sentence(sentence)

    # DO NOT MODIFY
    def test(
        self, tagged_sentences: Iterable[Sequence[TaggedToken]]
    ) -> tuple[list[str], list[str]]:
        """Return a tuple containing a list of predicted tags and a list of actual tags.

        Does not preserve sentence boundaries to make evaluation simpler.
        """
        predicted: list[str] = []
        actual: list[str] = []
        for sentence in tagged_sentences:
            predicted.extend(self.tag_sentence([tok.text for tok in sentence]))
            actual.extend([tok.tag for tok in sentence])
        return predicted, actual


# DO NOT MODIFY
def safe_log(n: float) -> float:
    """Return the log of a number or -inf if the number is zero."""
    return NEG_INF if n == 0.0 else log(n)


# DO NOT MODIFY
def max_item(scores: dict[str, float]) -> tuple[str, float]:
    """Return the key and value with the highest value."""
    # PyCharm gives a false positive type warning here
    # noinspection PyTypeChecker
    return max(scores.items(), key=itemgetter(1))


# DO NOT MODIFY
def most_frequent_item(counts: Counter[str]) -> str:
    """Return the most frequent item in a Counter.

    In case of ties, the lexicographically first item is returned.
    """
    assert counts, "Counter is empty"
    return items_descending_value(counts)[0]


# DO NOT MODIFY
def items_descending_value(counts: Counter[str]) -> list[str]:
    """Return the keys in descending frequency, breaking ties lexicographically."""
    # Why can't we just use most_common? It sorts by descending frequency, but items
    # of the same frequency follow insertion order, which we can't depend on.
    # Why can't we just use sorted with reverse=True? It will give us descending
    # by count, but reverse lexicographic sorting, which is confusing.
    # So instead we used sorted() normally, but for the key provide a tuple of
    # the negative value and the key.
    # PyCharm gives a false positive type warning here
    # noinspection PyTypeChecker
    return [key for key, value in sorted(counts.items(), key=_items_sort_key)]


# DO NOT MODIFY
def _items_sort_key(item: tuple[str, int]) -> tuple[int, str]:
    # This is used by items_descending_count, but you should never call it directly.
    return -item[1], item[0]


############################################################
# The stubs below this are the ones that you should fill in.
# Do not modify anything above this line other than to add any needed imports.


class MostFrequentTagTagger(Tagger):
    def __init__(self) -> None:
        # Add an attribute to store the most frequent tag
        self.default_tag = ""

    def train(self, sentences: Iterable[Sequence[TaggedToken]]) -> None:
        tag_counts = Counter()
        for sentence in sentences:
            for token in sentence:
                tag_counts[token.tag] += 1
        self.default_tag = most_frequent_item(tag_counts)

    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        return [self.default_tag] * len(sentence)


class UnigramTagger(Tagger):
    def __init__(self) -> None:
        # Add data structures that you need here
        self.default_tag = ""
        self.most_frequent = dict()
        self.all_words = set()

    def train(self, sentences: Iterable[Sequence[TaggedToken]]):
        all_token_tag_counts = defaultdict(Counter)
        tag_counts = Counter()
        for sentence in sentences:
            for token in sentence:

                # for every token:

                all_token_tag_counts[token.text][token.tag] += 1
                tag_counts[token.tag] += 1
                self.all_words.add(token.text)
        self.default_tag = most_frequent_item(tag_counts)
        for token in all_token_tag_counts:
            self.most_frequent[token] = most_frequent_item(all_token_tag_counts[token])

    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        return [self.most_frequent[token] if token in self.all_words else self.default_tag for token in sentence]


class SentenceCounter:
    def __init__(self, k: float) -> None:

        # used for emission probs:
        self.token_given_tag_count = defaultdict(Counter)
        self.tag_counts = Counter()
        self.vocabulary_per_tag = Counter()

        self.k = k
        self.smoothed_emission_prob = defaultdict(lambda: defaultdict(float))
        self.unique_tags_list: list[str] = list()

        self.unsmoothed_transition_matrix = defaultdict(lambda: defaultdict(float))
        self.init_tag = "<init>"

    def count_sentences(self, sentences: Iterable[Sequence[TaggedToken]]) -> None:

        # unsmoothed transition counts
        unsmoothed_transition_counts = defaultdict(Counter)

        # used for vocabulary per tag
        vocabulary_per_tag = defaultdict(set)

        for sentence in sentences:
            prev_tag = self.init_tag
            for tagged_token in sentence:
                # emission probs calculations
                self.token_given_tag_count[tagged_token.tag][tagged_token.text] += 1
                self.tag_counts[tagged_token.tag] += 1
                vocabulary_per_tag[tagged_token.tag].add(tagged_token.text)

                # transition matrix counts
                unsmoothed_transition_counts[prev_tag][tagged_token.tag] += 1
                prev_tag = tagged_token.tag
        
        # unique tags
        self.unique_tags_list = items_descending_value(self.tag_counts)

        # vocab per tag
        for key, items in vocabulary_per_tag.items():
            self.vocabulary_per_tag[key] = len(items)

        # process translation matrix
        for prev_tag, next_tag_counter in unsmoothed_transition_counts.items():
            total_count = next_tag_counter.total()
            for curr_tag, curr_tag_count in next_tag_counter.items():
                self.unsmoothed_transition_matrix[prev_tag][curr_tag] = curr_tag_count / total_count

    def unique_tags(self) -> list[str]:
        return self.unique_tags_list

    def emission_prob(self, tag: str, word: str) -> float:
        numerator = self.token_given_tag_count[tag][word] + self.k
        denominator = self.tag_counts[tag] + self.vocabulary_per_tag[tag] * self.k
        if denominator == 0:
            return 0
        return numerator / denominator

    def transition_prob(self, prev_tag: str, current_tag: str) -> float:
        return self.unsmoothed_transition_matrix[prev_tag][current_tag]  #

    def initial_prob(self, tag: str) -> float:
        return self.unsmoothed_transition_matrix[self.init_tag][tag]  # done


class BigramTagger(Tagger, ABC):
    # You can add additional methods to this class if you want to share anything
    # between the greedy and Viterbi taggers. However, do not modify any of the
    # implemented methods.

    def __init__(self, k) -> None:
        # DO NOT MODIFY THIS METHOD
        self.counter = SentenceCounter(k)

    def train(self, sents: Iterable[Sequence[TaggedToken]]) -> None:
        # DO NOT MODIFY THIS METHOD
        self.counter.count_sentences(sents)

    def sequence_probability(self, sentence: Sequence[str], tags: Sequence[str]) -> float:
        """Return the probability for a sequence of tags given tokens."""
        probability = 0
        # get first item probability
        if len(sentence) >= 1 and len(tags) >= 1:
            # initial probability
            probability += safe_log(self.counter.initial_prob(tags[0]))
            probability += safe_log(self.counter.emission_prob(tags[0], sentence[0]))

            sentence_length = len(sentence)
            for i in range(1, sentence_length):
                # transition probability
                probability += safe_log(self.counter.transition_prob(tags[i - 1], tags[i]))
                # emission probability
                probability += safe_log(self.counter.emission_prob(tags[i], sentence[i]))
        return probability
        # get rest of probabilities


class GreedyBigramTagger(BigramTagger):
    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:
        sentence_length = len(sentence)
        tag_sequences = [""] * sentence_length
        max_prob = float('-inf')
        for tag in self.counter.unique_tags():
            init_prob = safe_log(self.counter.initial_prob(tag))
            emission_prob = safe_log(self.counter.emission_prob(tag, sentence[0]))
            total_prob = init_prob + emission_prob
            if total_prob > max_prob:
                max_prob = total_prob
                tag_sequences[0] = tag

        for i in range(1, sentence_length):
            max_prob = float("-inf")
            for tag in self.counter.unique_tags():
                emission_prob = safe_log(self.counter.emission_prob(tag, sentence[i]))
                transition_prob = safe_log(self.counter.transition_prob(tag_sequences[i - 1], tag))
                curr_prob = emission_prob + transition_prob
                if curr_prob > max_prob:
                    max_prob = curr_prob
                    tag_sequences[i] = tag
        return tag_sequences


class ViterbiBigramTagger(BigramTagger):
    def tag_sentence(self, sentence: Sequence[str]) -> list[str]:

        sentence_length = len(sentence)

        best_scores: list[defaultdict[str: float]] = list(defaultdict(float) for _ in range(sentence_length))
        backpointers: list[defaultdict[str: str]] = list(defaultdict(str) for _ in range(sentence_length))

        if sentence_length < 1:
            return []

        # calculate at position 1
        for tag in self.counter.unique_tags():
            best_scores[0][tag] = safe_log(self.counter.emission_prob(tag, sentence[0])) + safe_log(self.counter.initial_prob(tag))
        # calculate path probability matrix
        for step in range(1, sentence_length):
            for tag in self.counter.unique_tags():
                prev_state, best_score = self.transition_score(tag, step, sentence[step], best_scores)
                best_scores[step][tag] = best_score
                backpointers[step][tag] = prev_state

        # backtracking stage
        best_sentence: list[str] = list()
        # tag : score
        best_tag, score = max_item(best_scores[sentence_length - 1])
        best_sentence.append(best_tag)
        backpointer_tag = backpointers[sentence_length - 1][best_tag]

        for step in range(sentence_length - 2, 0, -1):
            best_sentence.append(backpointer_tag)
            backpointer_tag = backpointers[step][backpointer_tag]
        if sentence_length > 1:
            best_sentence.append(backpointer_tag)

        best_sentence.reverse()
        return best_sentence

    # returns tuple of best transition score and best previous state
    def transition_score(self, tag, step, curr_word, best_scores):
        # prev state
        scores = dict()  # prev_tag : probability_of_tag
        # emisison * prev_score * transition
        for prev_tag in self.counter.unique_tags():
            emission = safe_log(self.counter.emission_prob(tag, curr_word))
            prev_score = best_scores[step - 1][prev_tag]
            transition = safe_log(self.counter.transition_prob(prev_tag, tag))
            scores[prev_tag] = emission + prev_score + transition
        return max_item(scores)
