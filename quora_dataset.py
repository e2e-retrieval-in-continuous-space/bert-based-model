import itertools
from collections import Counter
from typing import Set, Dict

from torch.utils.data import Dataset

from data_utils import *
from transformers.file_utils import cached_path


class QuoraDataset(Dataset):
    def __init__(self, qid_pairs: List[Tuple[str, str]], qid2text: Dict[str, str]):
        self.qid_pairs = qid_pairs
        self.qid2text = qid2text

    def __len__(self):
        return len(self.qid_pairs)

    def __getitem__(self, idx):
        qid1, qid2 = self.qid_pairs[idx]
        res = (Question(qid1, self.qid2text[qid1]), Question(qid2, self.qid2text[qid2]))
        return res


class RetrievalDataset(Dataset):
    def __init__(self, query_result_list: List[Tuple[str, Set[str]]]):
        self.query_result_list = query_result_list

    def __len__(self):
        return len(self.query_result_list)

    def __getitem__(self, idx):
        return self.query_result_list[idx]


class QuoraDataUtil:
    """
    A data loader class for the Quora dataset at:

    https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs
    """
    def __init__(self, src_filename=None, limit=None):
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
        self._qid2text = qid2text
        self._examples = examples

    def get_examples(self):
        return self._examples

    def get_text_for_qid(self, qid):
        return self._qid2text.get(qid, None)

    def construct_retrieval_task(self, train_size=139306, test_size=12206, retrieval_size=9218):
        # use positive examples to form disjoint sets
        pos_examples = [e for e in self._examples if e.is_duplicate]
        pos_qids = set(flatmap([(e.qid1, e.qid2) for e in pos_examples]))
        uf = UnionFind(pos_qids)
        for e in pos_examples:
            # each positive example is considered an edge
            uf.union(e.qid1, e.qid2)

        #  disjoint_set is a map of set name to its members
        disjoint_set = defaultdict(list)
        for qid in pos_qids:
            set_name = uf.find(qid)
            disjoint_set[set_name].append(qid)

        # construct training pairs
        set_names = list(disjoint_set.keys())
        random.shuffle(set_names)
        train_qid_pairs = set()
        for sn in set_names:
            dset = disjoint_set[sn]
            pairs = itertools.combinations(dset, 2)
            train_qid_pairs.update(pairs)

            # might go over the target train_size a bit
            if len(train_qid_pairs) >= train_size:
                break

        # filter out examples whose qids are already in the train_qid_pairs
        test_examples = []
        train_qids = set(flatmap(train_qid_pairs))
        for e in self._examples:
            if not e.is_duplicate:
                test_examples.append(e)
            else:
                if e.qid1 not in train_qids and e.qid2 not in train_qids:
                    test_examples.append(e)

        # build positive test_qid_pairs
        test_qid_pairs = set()
        random.shuffle(test_examples)
        candidate_ids = set()
        query2result_list = []
        for e in test_examples:
            if e.qid1 in candidate_ids or e.qid2 in candidate_ids:
                continue

            if e.is_duplicate:
                # doesn't matter which of qid1 or qid2 to use here.  they are in the same set
                set_members = disjoint_set[uf.find(e.qid1)]
                pairs = list(itertools.combinations(set_members, 2))
                test_qid_pairs.update(pairs)

                # set members are qids which are relevant to each other
                # every member can be considered a test query
                candidate_ids.update(set_members)
                for qid in set_members:
                    result = set_members.copy()
                    # remove query from result
                    result.remove(qid)
                    query2result_list.append((qid, result))

            candidate_ids.update([e.qid1, e.qid2])
            if len(query2result_list) >= retrieval_size:
                break

        test_qids = flatmap(test_qid_pairs)
        assert len(train_qids.intersection(test_qids)) == 0, "train and test qids have overlaps"

        train_qid_pairs = list(train_qid_pairs)[:train_size]
        test_qid_pairs = list(test_qid_pairs)[:test_size]
        query2result_list = query2result_list[:retrieval_size]
        return (
            QuoraDataset(train_qid_pairs, self._qid2text),
            QuoraDataset(test_qid_pairs, self._qid2text),
            RetrievalDataset(query2result_list),
            list(candidate_ids),
            self._qid2text
        )

    def construct_retrieval_task_old(self, train_size=139306, retrieval_size=9218, split_fracs=[0.03, 0.97], seed=1):
        """
        Quora dataset is loaded into examples which are then split into test_examples, and train_examples
        They each contain positive and negative examples.

        A retrieval task can be constructed as:
            - train data:  positive qid pairs from the train_examples including additional ones from transitive closure
            - test data:  positive qid pairs from the test_examples including additional ones from transitive closure
            - test query set:  each qid in the test data is a query
            - query2result list:  a map of test query to relevant results
            - candidates: All qids from the test_examples which can be from the positive or negative examples

        Args:
            split_fracs:
                List of fractions (test_portion, train_portion) to split the examples.
                Portions must add to done
        """
        # Use all positive and negative examples for the split
        test_examples, train_examples = split_examples(self._examples, split_fracs, seed=seed)

        test_pos_examples = [(e.qid1, e.qid2) for e in test_examples if e.is_duplicate]
        # a set of positive (qid1, qid2)
        test_qid_pairs, test_reachable = generate_all_examples(test_pos_examples)

        # a set of (qid1, qid2, ...)
        test_qid_set = set(flatmap(test_qid_pairs))

        train_pos_qids = set([(e.qid1, e.qid2)for e in train_examples if e.is_duplicate])

        # a set of positive (qid1, qid2)
        train_qid_pairs, _ = generate_all_examples(train_pos_qids)

        # Filter out training examples that have qid appearing in test_question_set and keep only positive examples
        train_qid_pairs = [(qid1, qid2) for qid1, qid2 in train_qid_pairs
                      if qid1 not in test_qid_set and qid2 not in test_qid_set]

        # Each question in a positive pair is considered a test query.
        # Construct a list of (query, result)
        query2result_list = [
            (qid, test_reachable.get(qid, set()))
            for qid in flatmap([(qid1, qid2) for qid1, qid2 in test_qid_pairs])
        ]

        # Candidates are generated based on all questions in test data including both positive
        # and negative pairs
        candidate_ids = list(set(flatmap([(e.qid1, e.qid2) for e in test_examples])))
    
        if seed:
            random.seed(seed)

        if train_size:
            random.shuffle(train_qid_pairs)
            train_qid_pairs = train_qid_pairs[:train_size]

        if retrieval_size:
            random.shuffle(query2result_list)
            query2result_list = query2result_list[:retrieval_size]

        return (
            QuoraDataset(train_qid_pairs, self._qid2text),
            QuoraDataset(list(test_qid_pairs), self._qid2text),
            RetrievalDataset(query2result_list),
            candidate_ids,
            self._qid2text,
        )

    @staticmethod
    def read_examples(src_filename, limit=None):
        """
        Iterator for the Quora question dataset.

        Args:
            src_filename:
                Full path to the file to be read.

            limit:
                Load up to the given examples
        """
        with open(src_filename, encoding="utf8") as f:
            # Skip header line
            next(f)

            for i, line in enumerate(f):
                # id  qid1  qid2  question1 question2 is_duplicate
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
    quora = QuoraDataUtil("quora_duplicate_questions.tsv")
    quora.construct_retrieval_task()



