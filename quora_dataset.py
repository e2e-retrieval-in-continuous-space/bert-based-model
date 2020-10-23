from data_utils import *
from transformers.file_utils import cached_path

class QuoraDataset:
    """
    A data class for the Quora dataset at:

    https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs
    """
    def __init__(self, src_filename=None, split_fracs=[0.03, 0.97], seed=1, limit=None):
        """
            Args:
                src_filename:
                    Local file path to the Quora Question Dataset
        """
        if src_filename is None:
            src_filename = cached_path("http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv")

        # 404279 examples,  149263 positive
        examples = list(self.read_examples(src_filename, limit))
        qid2text = {
            qid: text for qid, text in
                     flatmap([[(e.qid1, e.q1_text), (e.qid2, e.q2_text)] for e in examples])
        }

        pos_examples = [e for e in examples if e.is_duplicate]

        # Build adjacency list
        graph = build_graph(pos_examples)

        # Build reachability map
        reachable = build_transitive_closure(graph)

        # Only use pos_examples because negative sampling would be used
        test_examples, train_examples = split_examples(pos_examples, split_fracs, seed=seed)

        test_data_qid = generate_data(test_examples, reachable)

        # Each element in a positive pair is considered a test query
        test_qid_set = set(flatmap(test_data_qid))

        # Filter out training examples that have qid appearing in test_query
        train_examples = [e for e in train_examples
                          if e.qid1 not in test_qid_set or e.qid2 not in test_qid_set]

        train_data_qid = generate_data(train_examples, reachable)

        self._qid2text = qid2text
        self._examples = examples
        self._reachable = reachable
        self._test_data_qid = test_data_qid
        self._train_data_qid = train_data_qid

    def get_examples(self):
        return self._examples

    def get_relevant_result(self, queries: List[str]):
        """
        Returns: List[Tuple[str, set(id)]]
            List of (query, List(candidate IDs))
        """
        return [self._reachable.get(q, set()) for q in queries]

    def get_text_for_qid(self, qid):
        return self._qid2text.get(qid, None)

    def get_train_data(self):
        """
        Returns:
             A list of tuples of positive pair of questions (question_text1, question_text2)
        """
        return [(self.get_text_for_qid(qid1), self.get_text_for_qid(qid2))
                for qid1, qid2 in self._train_data_qid]

    def get_test_data(self):
        """
        Returns:
             A list of tuples of positive pair of questions (question_text1, question_text2)
        """
        return [(self.get_text_for_qid(qid1), self.get_text_for_qid(qid2))
                for qid1, qid2 in self._test_data_qid]

    @staticmethod
    def read_examples(src_filename, limit=None):
        """
        Iterator for the Quora question dataset.

        Args:
            src_filename:
                Full path to the file to be read.
        """
        with open(src_filename, encoding="utf8") as f:
            # Skip header line
            next(f)

            for i, line in enumerate(f):
                fields = line.strip().split("\t")

                # Throw away malformed lines (e.g. containing "\n")
                if len(fields) != 6:
                    continue

                # convert is_duplicate to boolean
                fields[-1] = True if fields[-1] == "1" else False

                yield Example(*fields)
                if limit is not None and i + 1 >= limit:
                    break


if __name__ == "__main__":
    quora = QuoraDataset("quora_duplicate_questions.tsv")
    test_data = quora.get_test_data()
    print(test_data[:10])

