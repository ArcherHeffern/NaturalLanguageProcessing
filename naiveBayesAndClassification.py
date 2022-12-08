import json
from collections import defaultdict, Counter
from math import log
from typing import (
    Iterable,
    Any,
    Sequence,
    Generator,
)

############################################################
# The following classes and methods are provided as helpers.
# Do not modify them! The stubs for what you need to implement are later in the file.

# DO NOT MODIFY
START_TOKEN = "<start>"
# DO NOT MODIFY
END_TOKEN = "<end>"


# DO NOT MODIFY
class AirlineSentimentInstance:
    """Represents a single instance from the airline sentiment dataset.

    Each instance contains the sentiment label, the name of the airline,
    and the sentences of text. The sentences are stored as a tuple of
    tuples of strings. The outer tuple represents sentences, and each
    sentences is a tuple of tokens."""

    def __init__(
        self, label: str, airline: str, sentences: Sequence[Sequence[str]]
    ) -> None:
        self.label: str = label
        self.airline: str = airline
        # These are converted to tuples so they cannot be modified
        self.sentences: tuple[tuple[str, ...], ...] = tuple(
            tuple(sentence) for sentence in sentences
        )

    def __repr__(self) -> str:
        return f"<AirlineSentimentInstance: {str(self)}>"

    def __str__(self) -> str:
        return f"label={self.label}; airline={self.airline}; sentences={self.sentences}"

    # You should never call this function directly. It's called by data loading functions.
    @classmethod
    def from_dict(cls, json_dict: dict[str, Any]) -> "AirlineSentimentInstance":
        return AirlineSentimentInstance(
            json_dict["label"], json_dict["airline"], json_dict["sentences"]
        )


# DO NOT MODIFY
class SentenceSplitInstance:
    """Represents a potential sentence boundary in context.

    Each instance is labeled with whether it is ('y') or is not ('n') a sentence
    boundary, the characters to the left of the boundary token, the potential
    boundary token itself (punctuation that could be a sentence boundary), and
    the characters to the right of the boundary token."""

    def __init__(
        self, label: str, left_context: str, token: str, right_context: str
    ) -> None:
        self.label: str = label
        self.left_context: str = left_context
        self.token: str = token
        self.right_context: str = right_context

    def __repr__(self) -> str:
        return f"<SentenceSplitInstance: {str(self)}>"

    def __str__(self) -> str:
        return " ".join(
            [
                f"label={self.label};",
                f"left_context={repr(self.left_context)};",
                f"token={repr(self.token)};",
                f"right_context={repr(self.right_context)}",
            ]
        )

    # You should never call this function directly. It's called by data loading functions.
    @classmethod
    def from_dict(cls, json_dict: dict[Any, Any]) -> "SentenceSplitInstance":
        return SentenceSplitInstance(
            json_dict["label"],
            json_dict["left"],
            json_dict["token"],
            json_dict["right"],
        )


# DO NOT MODIFY
def load_sentiment_instances(
    datapath: str,
) -> Generator[AirlineSentimentInstance, None, None]:
    """Load airline sentiment instances from a JSON file."""
    with open(datapath, encoding="utf8") as infile:
        json_list = json.load(infile)
        for json_item in json_list:
            yield AirlineSentimentInstance.from_dict(json_item)


# DO NOT MODIFY
def load_segmentation_instances(
    datapath: str,
) -> Generator[SentenceSplitInstance, None, None]:
    """Load sentence segmentation instances from a JSON file."""
    with open(datapath, encoding="utf8") as infile:
        json_list = json.load(infile)
        for json_item in json_list:
            yield SentenceSplitInstance.from_dict(json_item)


    # DO NOT MODIFY
class ClassificationInstance:
    """Represents a label and features for classification."""

    def __init__(self, label: str, features: Iterable[str]) -> None:
        self.label: str = label
        # Features can be passed in as any iterable and they will be
        # stored in a tuple
        self.features: tuple[str, ...] = tuple(features)

    def __repr__(self) -> str:
        return f"<ClassificationInstance: {str(self)}>"

    def __str__(self) -> str:
        return f"label={self.label}; features={self.features}"


############################################################
# The stubs below this are the ones that you should fill in.
# Do not modify anything above this line other than to add any needed imports.


def accuracy(predictions: Sequence[str], expected: Sequence[str]) -> float:
    """Compute the accuracy of the provided predictions."""
    if len(predictions) != len(expected):
        raise ValueError(
            f"""Predictions and Expected should be the same length
        Prediction length: {len(predictions)}
        Expected length: {len(expected)}""")
    total_count = 0
    correct_count = 0
    for prediction, expect in zip(predictions, expected):
        total_count += 1
        if prediction == expect:
            correct_count += 1
    if not total_count:
        return 0
    else:
        return correct_count / total_count


def recall(predictions: Sequence[str], expected: Sequence[str], label: str) -> float:
    """Compute the recall of the provided predictions."""
    if len(predictions) != len(expected):
        raise ValueError(
            f"""Predictions and Expected should be the same length
        Prediction length: {len(predictions)}
        Expected length: {len(expected)}""")
    total_count = 0
    correct_count = 0
    for prediction, expect in zip(predictions, expected):
        if expect == label:
            total_count += 1
            if prediction == label:
                correct_count += 1
    if not total_count:
        return 0
    else:
        return correct_count / total_count


def precision(predictions: Sequence[str], expected: Sequence[str], label: str) -> float:
    """Compute the precision of the provided predictions."""
    # number of true positives divided by the total number of positive predictions
    if len(predictions) != len(expected):
        raise ValueError(
            f"""Predictions and Expected should be the same length
        Prediction length: {len(predictions)}
        Expected length: {len(expected)}""")
    true_positives = 0
    positives = 0
    for prediction, expect in zip(predictions, expected):
        if prediction == label:
            positives += 1
        if expect == label and expect == prediction:
            true_positives += 1
    if not positives:
        return 0
    else:
        return true_positives / positives


def f1(predictions: Sequence[str], expected: Sequence[str], label: str) -> float:
    """Compute the F1-score of the provided predictions."""
    temp_precision = precision(predictions, expected, label)
    temp_recall = recall(predictions, expected, label)
    return 2 * (temp_recall * temp_precision) / (temp_recall + temp_precision)


class UnigramAirlineSentimentFeatureExtractor:
    @staticmethod
    def extract_features(instance: AirlineSentimentInstance) -> ClassificationInstance:
        """Extract unigram features from an instance."""
        no_duplicates = {val.lower() for sublist in instance.sentences for val in sublist}
        return ClassificationInstance(instance.label, no_duplicates)


def bigrams(sentence: Sequence[str]) -> list[tuple[str, str]]:
    padded_sentence = [START_TOKEN] + list(sentence) + [END_TOKEN]
    bi_list: list = list()
    size: int = len(sentence)
    for i in range(0, size + 1):
        bi_list.append((padded_sentence[i].lower(), padded_sentence[i + 1].lower()))
    return bi_list


class BigramAirlineSentimentFeatureExtractor:
    @staticmethod
    def extract_features(instance: AirlineSentimentInstance) -> ClassificationInstance:
        """Extract bigram features from an instance."""
        no_duplicates = {str(val) for sublist in instance.sentences for val in bigrams(sublist)}
        return ClassificationInstance(instance.label, no_duplicates)


class BaselineSegmentationFeatureExtractor:
    @staticmethod
    def extract_features(instance: SentenceSplitInstance) -> ClassificationInstance:
        """Extract features for all three tokens from an instance."""
        # ('left_tok=22', 'split_tok=.', 'right_tok=The')
        # split = ""
        # if instance.label == "y":
        split = instance.token
        epic_tuple = ("left_tok=" + instance.left_context, "split_tok=" + split, "right_tok=" + instance.right_context)
        return ClassificationInstance(instance.label, epic_tuple)


class InstanceCounter:
    """Holds counts of the labels and features seen during training.

    See the assignment for an explanation of each method."""

    def __init__(self) -> None:
        self.label_counts = Counter()
        self.total_label_count = 0
        self.feature_count = defaultdict(Counter)
        self.feature_total_counts = Counter()
        self.unique_label_list = list()
        self.unique_feature_set = set()
        self.unique_feature_set_size = 0

    def count_instances(self, instances: Iterable[ClassificationInstance]) -> None:
        # You should fill in this loop. Do not try to store the instances!
        temp_unique_label_set = set()
        for instance in instances:
            # Given a label, return how many times that label was seen.
            # This is used to compute prior probabilities over labels.
            self.label_counts[instance.label] += 1
            # Return the number of total labels seen in the training data. (This is just the number of instances
            # seen, as each has only one label.) This is not the number of unique labels.
            self.total_label_count += 1
            # Given a feature and a label, how many times does feature show up when only counting over
            # instances of class label? For example, how many times does the unigram feature “good” appear
            # among all instances with the label “positive”?
            for element in instance.features:
                self.feature_count[instance.label][element] += 1
                self.unique_feature_set.add(element)
                self.feature_total_counts[instance.label] += 1
            # Return a list (not a set!) of the unique labels observed. For example, if the instances
            # have labels: ['y', 'n', 'y', 'y', 'n'], you should return only ['y', 'n']. The order of
            # the labels does not matter. You should just return a stored precomputed value (for example,
            # you could keep a variable called self.label_list). Note that you cannot give an instance
            # attribute the same name as a method, so you can’t call it
            temp_unique_label_set.add(instance.label)
            # Return the size of your feature vocabulary. This is the number of unique features that
            # have been seen across all instances, regardless of label.

        self.unique_label_list = list(temp_unique_label_set)
        self.unique_feature_set_size = len(self.unique_feature_set)

    def label_count(self, label: str) -> int:
        return self.label_counts[label]

    def total_labels(self) -> int:
        return self.total_label_count

    def feature_label_joint_count(self, feature: str, label: str) -> int:
        return self.feature_count[label][feature]

    def labels(self) -> list[str]:
        return self.unique_label_list

    def feature_vocab_size(self) -> int:
        return self.unique_feature_set_size

    def feature_set(self) -> set[str]:
        return self.unique_feature_set

    # For a given label, this is the total count of all the features seen with the label. Note that
    # this is not the number of unique features seen with the label.
    def total_feature_count_for_label(self, label: str) -> int:
        return self.feature_total_counts[label]
# Do not modify them! The stubs for what you need to implement are later in the file.

class NaiveBayesClassifier:
    """Perform classification using naive Bayes.

    See the assignment for an explanation of each method."""

    # DO NOT MODIFY
    def __init__(self, k: float):
        self.k: float = k
        self.instance_counter: InstanceCounter = InstanceCounter()

    # DO NOT MODIFY
    def train(self, instances: Iterable[ClassificationInstance]) -> None:
        self.instance_counter.count_instances(instances)

    def prior_prob(self, label: str) -> float:
        # count of the label divided by the total number of instances seen during training.
        return self.instance_counter.label_count(label) / self.instance_counter.total_labels()

    def likelihood_prob(self, feature: str, label) -> float:
        # that is the count of the feature with the given label (computed by feature_label_joint_count
        # plus the k value

        # divided by the total count of all the features observed with that class plus the feature vocabulary
        # size (the number of unique features observed) multiplied by k.
        inst = self.instance_counter
        k = self.k
        numerator = inst.feature_label_joint_count(feature, label) + k
        denominator = inst.total_feature_count_for_label(label) + (inst.feature_vocab_size() * k)
        return numerator / denominator

    def log_posterior_prob(self, features: Sequence[str], label: str) -> float:
        probability = 0
        for feature in features:
            # ensure feature is in features
            if feature in self.instance_counter.feature_set():
                probability += log(self.likelihood_prob(feature, label))
        return log(self.prior_prob(label)) + probability

    def classify(self, features: Sequence[str]) -> str:
        probs = []
        for label in self.instance_counter.unique_label_list:
            probs.append((self.log_posterior_prob(features, label), label))
        return max(probs)[1]

    def test(
        self, instances: Iterable[ClassificationInstance]
    ) -> tuple[list[str], list[str]]:
        true_values = list()
        predicted_values = list()
        for instance in instances:
            predicted_values.append(self.classify(instance.features))
            true_values.append(instance.label)
        # get predicted and true labels
        return predicted_values, true_values


# MODIFY THIS AND DO THE FOLLOWING:
# 1. Inherit from UnigramAirlineSentimentFeatureExtractor or BigramAirlineSentimentFeatureExtractor
#    (instead of object) to get an implementation for the extract_features method.
# 2. Set a value for self.k below based on your tuning experiments.
class TunedAirlineSentimentFeatureExtractor(UnigramAirlineSentimentFeatureExtractor):
    def __init__(self) -> None:
        self.k = .5

