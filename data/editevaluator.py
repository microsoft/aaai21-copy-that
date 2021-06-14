from collections import Counter
from difflib import SequenceMatcher
from typing import List, NamedTuple, Set, Tuple, Dict


class EditEvaluator:
    """Evaluate a (code) editing model."""

    def __init__(self):
        self.__num_samples = 0  # type: int
        self.__sum_exact_matches = 0  # type: int

        # Does the model edit the correct span of the original (the concrete edit doesn't matter)
        self.__sum_prediction_similarity_to_target = 0  # type: float
        self.__span_precision_sum = 0.0  # type: float
        self.__span_recall_sum = 0.0  # type: float
        self.__span_f1_sum = 0.0  # type: float

        # Jaccard
        self.__sum_jaccard_similarity_of_tokens = 0.  # type: float
        self.__sum_jaccard_similarity_of_added_tokens = 0.  # type: float
        self.__sum_jaccard_similarity_of_deleted_tokens = 0.  # type: float

    EditInformation = NamedTuple('EditInformation', [
        ('added_tokens', Counter),
        ('deleted_tokens', Counter),
        ('edits', List[Tuple[int, int, Tuple]]),
        ('sequence_matcher', SequenceMatcher)
    ])

    def __get_edit_information(self, before, after) -> 'EditEvaluator.EditInformation':
        seq_matcher = SequenceMatcher(None, before, after)
        added_tokens, deleted_tokens = Counter(), Counter()
        edited_spans_on_before = []  # type: List[Tuple[int, int, Tuple]]

        for tag, i1, i2, j1, j2 in seq_matcher.get_opcodes():
            if tag == 'equal':
                continue
            deleted_tokens.update(before[i1:i2])
            added_tokens.update(after[j1:j2])
            edited_spans_on_before.append((i1, i2, tuple(after[j1:j2])))

        return self.EditInformation(added_tokens, deleted_tokens, edited_spans_on_before, seq_matcher)

    def add_sample(self, original: List[str], gold_edited: List[str], predicted_edited: List[str]) -> None:
        """
        :param original: The original input that is being edited
        :param gold_edit: The gold edited version of `original`
        :param predicted_edits: The predicted edited version of the original
        """
        self.__num_samples += 1

        if gold_edited == predicted_edited:
            self.__sum_exact_matches += 1

        gold_edit_info = self.__get_edit_information(original, gold_edited)
        predicted_edit_info = self.__get_edit_information(original, predicted_edited)

        num_added_by_both_tokens = sum((gold_edit_info.added_tokens & predicted_edit_info.added_tokens).values())
        num_deleted_by_both_tokens = sum((gold_edit_info.deleted_tokens & predicted_edit_info.deleted_tokens).values())

        num_added_by_either_tokens = sum((gold_edit_info.added_tokens | predicted_edit_info.added_tokens).values())
        num_deleted_by_either_tokens = sum((gold_edit_info.deleted_tokens | predicted_edit_info.deleted_tokens).values())


        sample_jaccard = (num_added_by_both_tokens + num_deleted_by_both_tokens) / (num_added_by_either_tokens + num_deleted_by_either_tokens)
        self.__sum_jaccard_similarity_of_tokens += sample_jaccard

        if num_added_by_either_tokens > 0:
            self.__sum_jaccard_similarity_of_added_tokens += num_added_by_both_tokens / num_added_by_either_tokens
        else:
            self.__sum_jaccard_similarity_of_added_tokens += 1

        if num_deleted_by_either_tokens > 0:
            self.__sum_jaccard_similarity_of_deleted_tokens += num_deleted_by_both_tokens / num_deleted_by_either_tokens
        else:
            self.__sum_jaccard_similarity_of_deleted_tokens += 1

        # This counts how more similar the predicted is to the golden edited compared to the original, implicitly normalizing
        # for the similarity of the original->gold_edited
        # < 1 means that the prediction is worse than doing no edits
        # > 1 means that the prediction is better than doing a random edit.
        original_to_gold_ratio = gold_edit_info.sequence_matcher.ratio()
        predicted_to_gold_ratio = SequenceMatcher(None, predicted_edited, gold_edited).ratio()
        if original_to_gold_ratio > 0:
            self.__sum_prediction_similarity_to_target +=  predicted_to_gold_ratio / original_to_gold_ratio
        elif predicted_to_gold_ratio > 0: # This should be very rare. The whole original->gold changed completely.
            self.__sum_prediction_similarity_to_target += 1
        else:
            self.__sum_prediction_similarity_to_target += 0


        # How accurate are the edit locations?
        # We count the precision/recall in term of tokens in the original that are edited. Insertions (which do not
        # edit a range) are are considered to have a span of 1 in the original input.
        def find_overlap_on_original(start_pos: int, end_pos: int, edits: List[Tuple[int, int, Tuple]]) -> Tuple[int, int]:
            assert start_pos <= end_pos
            correctly_changed = 0

            if start_pos == end_pos:
                for s, e, _ in edits:
                    if s <= start_pos <= e:
                        # Insertions at the correct positions count as correct
                        return 1, 1
                return 0, 1  # An insertion in the wrong place.

            for s, e, _ in edits:
                if s <= start_pos <= e or s <= end_pos <= e or (start_pos <= s and e <= end_pos):
                    if s == e:  # An insertion in [s,e) which overlaps with [start_pos, end_pos)
                        correctly_changed += 1
                    else:
                        correctly_changed += min(e, end_pos) - max(s, start_pos)


            return correctly_changed, end_pos - start_pos

        def compute_coverage(edits: List[Tuple[int, int, Tuple]], target_edits: List[Tuple[int, int, Tuple]]):
            sum_correct_changed, sum_total_changed = 0, 0
            for predicted_edit_start, predicted_edit_end, _ in edits:
                cor, total = find_overlap_on_original(predicted_edit_start, predicted_edit_end, target_edits)
                sum_correct_changed += cor
                sum_total_changed += total
            if sum_correct_changed > 0:
                return sum_correct_changed / sum_total_changed
            else:
                return 0

        precision = compute_coverage(predicted_edit_info.edits, gold_edit_info.edits)
        recall = compute_coverage(gold_edit_info.edits, predicted_edit_info.edits)

        self.__span_precision_sum += precision
        self.__span_recall_sum += recall
        if precision + recall > 0:
            self.__span_f1_sum += 2 * (precision * recall) / (precision + recall)
        else:
            self.__span_f1_sum += 0

    def evaluation_statistics(self) -> Dict[str, float]:
        return {
            'Exact Match': self.__sum_exact_matches / self.__num_samples,
            'Jaccard Similarity of Edits': self.__sum_jaccard_similarity_of_tokens / self.__num_samples,
            'Jaccard Similarity of Edits - Addition:': self.__sum_jaccard_similarity_of_added_tokens / self.__num_samples,
            'Jaccard Similarity of Edit - Deletion': self.__sum_jaccard_similarity_of_deleted_tokens / self.__num_samples,
            'Normalized Prediction Similarity to Target': self.__sum_prediction_similarity_to_target / self.__num_samples,
            'Span Precision': self.__span_precision_sum / self.__num_samples,
            'Span Recall': self.__span_recall_sum / self.__num_samples,
            'Span F1': self.__span_f1_sum / self.__num_samples,
        }

if __name__ == '__main__':
    evaluator = EditEvaluator()
    evaluator.add_sample(['a', 'b', 'd', 'e'], ['a', 'x', 'd', 'x', 'e'], ['a', 'b', 'd', 'x', 'e'])
    print(evaluator.evaluation_statistics())