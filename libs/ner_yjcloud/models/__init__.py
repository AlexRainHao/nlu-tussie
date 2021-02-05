"""
Evaluation for
    * a) Classifier task
    * b) NER task
    
for a)
Metrics for Single-label Classifier, including
    * overall precision
    * overall f1 score
    * overall recall
    * each classes precision
    * each classes f1 score
    * each classes recall

E.X.
    >>> idx2int = {0: "default", 1: "eat", 2: "sleep", 3: "hangover", 4: "worker", 5: "clean"}
    >>> size = 100
    >>> true_label = np.random.choice(list(idx2int.keys()), size = size, replace = True)
    >>> pred_label = np.random.choice(list(idx2int.keys()), size = size, replace = True)
    >>> op = clsEvalCounter(idx2int = idx2int)
    >>> res = op.run(true_label, pred_label)
    >>> print(res)
    >>> {"precision": 0., "f1": 0., "recall": 0.,
        "cls_1": {"precision" :0., "f1": 0., "recall": 0.},
        "cls_2": {"precision" :0., "f1": 0., "recall": 0.}}

for b)
Metrics for NER, including evaluator
    * Precision
    * Recall
    * F1-score
for each entity type
Only supports for tagging schema of `BIO2` and `BIOES`

E.X.
    >>> text = list("今天，王德发在北京进行了散步活动")
    >>> tag = ["B-TIME", "I-TIME", "O", "B-PER", "I-PER", "B-PER", "O", "B-LOC", "I-LOC"] + ["O"] * 7
    >>> pred_tag = ["B-TIME", "I-TIME", "O", "B-PER", "B-PER", "I-PER", "O", "B-LOC", "I-LOC"] + ["O"] * 3 + ["B-LOC"] + ["O"] * 3
    >>> op = nerEvalCounter()
    >>> op.parser(tag, pred_tag)
    >>> op.evalutaion()
    >>> print(op.res)
    >>> {'precision': 0.4, 'f1': 0.4444444444444445, 'recall': 0.5,
         'TIME': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0},
         'PER': {'precision': 0.0, 'recall': 0.0, 'f1': 0},
         'LOC': {'precision': 0.5, 'recall': 1.0, 'f1': 0.6666666666666666}
         }
"""


from typing import List, Text, Union, Dict
from collections import defaultdict
import numpy as np
from sklearn.metrics import recall_score, f1_score, precision_score

SPACETAG = "O"
STARTTAG = ["B", "S"]
MIDTAG = ["I"]


def report_score_matr(result_score):
    label_name = ["total"] + [k for k, v in result_score.items() if isinstance(v, dict)]
    headers = ["precision", "recall", "f1"]
    
    rows = defaultdict(dict)
    for key, val in result_score.items():
        if key in headers:
            rows["total"][key] = val
        else:
            rows[key] = val
    digits = 3
    name_width = max(len(str(cn)) for cn in label_name)
    width = max(name_width, digits)
    head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)
    report = head_fmt.format('', *headers, width=width)

    report += '\n\n'
    row_fmt = '{:>{width}s} ' + ' {:>9.{digits}f}' * 3 + ' {:>9}\n'
    for row, val in rows.items():
        report += row_fmt.format(str(row), *list(val.values()), '', width=width, digits=digits)
    report += '\n'
    return report

def report_epoch_score_matr(ep_result_score: Dict):
    report = ""
    for ep, score in ep_result_score.items():
        report += "*" * 7
        report += f"Epoch: {ep}\n"
        report += report_score_matr(score)
    return report
    

def split_tag(tag):
    """convert entity tag to position tag and entity type
    E.X.
        B-LOC -> (B, LOC)
    """
    
    # if tag == "O" or tag == "[CLS]" or tag == "[SEP]":
    #     return ("O", None)
    #
    # return tag.split('-', maxsplit=1)
    stag = tag.split('-', maxsplit=1)
    
    if len(stag) == 1:
        return ("O", None)
    
    else:
        p_tag, s_tag = stag
        return p_tag, s_tag


def is_type_end(ptag: Text, ctag: Text):
    """whether previous tag span ended"""
    pfix, ptype = split_tag(ptag)
    cfix, ctype = split_tag(ctag)
    
    if pfix == SPACETAG:
        return False
    
    elif cfix == SPACETAG:
        return pfix != SPACETAG
    
    elif ptype != ctype:
        return True
    
    elif pfix in STARTTAG:
        return False if cfix in MIDTAG else True
    
    elif cfix in STARTTAG:
        return True
    else:
        return False


def is_type_start(ptag: Text, ctag: Text):
    """whether current tag start a new span"""
    pfix, ptype = split_tag(ptag)
    cfix, ctype = split_tag(ctag)
    
    if cfix == SPACETAG:
        return False
    
    elif pfix == SPACETAG:
        return cfix != SPACETAG
    
    elif ptype != ctype:
        return True
    
    elif pfix in STARTTAG:
        return False if cfix in MIDTAG else True
    
    elif cfix in STARTTAG:
        return True
    else:
        return False


class nerEvalCounter:
    """A base NER classfication evaluator class supporting tagging schema
        * BIOES
        * BIO2
    """
    
    name = "ner_evaluation_op"
    
    # tag
    def __init__(self):
        self._init_counter()
    
    def parser(self, true_tags: List, pred_tags: List):
        """
        A strict NER evaluation for single sequence,
        where each entity span TP, P, T would be calculated

        TP: only the same entity type and have the same span position in both true and predicted label sequence
            would called
        P: called once a new span started in predicted sequence
        T: called once a new span started in true sequence
        """
        true_tags, pred_tags = self._truncate_seq(true_tags, pred_tags)
        prev_true_tag, prev_pred_tag = SPACETAG, SPACETAG
        inspect_type = None  # for a span entity type need to evaluate
        
        for ttag, ptag in zip(true_tags, pred_tags):
            _, ttype = split_tag(ttag)
            _, ptype = split_tag(ptag)
            
            # if type end
            if inspect_type:
                true_end = is_type_end(prev_true_tag, ttag)
                pred_end = is_type_end(prev_pred_tag, ptag)
                if true_end ^ pred_end or ttype != ptype:  # one of end, the span faild
                    inspect_type = None
                
                elif true_end & pred_end:  # both end
                    self.typCorrCont[inspect_type] += 1
                    inspect_type = None
            
            # if new type start
            true_start = is_type_start(prev_true_tag, ttag)
            pred_start = is_type_start(prev_pred_tag, ptag)
            
            if true_start & pred_start & (ttype == ptype):
                inspect_type = ttype
            if true_start:
                self.typTrueCont[ttype] += 1
            if pred_start:
                self.typPredCont[ptype] += 1
            
            prev_true_tag, prev_pred_tag = ttag, ptag
        
        if inspect_type:
            self.typCorrCont[inspect_type] += 1
    
    def _truncate_seq(self, tseq: List, pseq: List):
        """truncate longer sequence"""
        tlen, plen = len(tseq), len(pseq)
        
        if tlen == plen:
            return tseq, pseq
        
        elif tlen < plen:
            return tseq, pseq[:tlen]
        
        else:
            return tseq[:plen], pseq
    
    def _init_counter(self):
        """initialize NER TP, P, T and results metrics scores """
        self.typCorrCont = defaultdict(int)
        self.typTrueCont = defaultdict(int)
        self.typPredCont = defaultdict(int)
        
        self.res = {
            "precision": 0.,
            "f1": 0.,
            "recall": 0.,
        }
    
    def run(self, true_tags: List, pred_tags: List):
        """
        
        Parameters
        ----------
        true_tags: 2D-List like
        pred_tags: 2D-List like

        """
        self._init_counter()
        if isinstance(true_tags[0], List):
            for true_tag, pred_tag in zip(true_tags, pred_tags):
                self.parser(true_tag, pred_tag)
        else:
            self.parser(true_tags, pred_tags)
            
        self.evalutaion()
        return self.res
    
    def evalutaion(self):
        """obtain overall metrics and each tag metrics after all dataset parsed,
        and a dict with key of tag name and value of corresponding
        metrics including `precision`, `recall`, `f1-score` return
        """
        tags = list(self.typPredCont.keys())
        
        # overall
        pres_oval, rec_oval, f1_oval = self.call_metrics(
            tp = sum(self.typCorrCont.values()),
            p = sum(self.typPredCont.values()),
            t = sum(self.typTrueCont.values())
        )
        self.res["precision"] = pres_oval
        self.res["recall"] = rec_oval
        self.res["f1"] = f1_oval
        
        # each label
        for tag in tags:
            if tag == SPACETAG:
                continue
            
            pres, rec, f1 = self.call_metrics(
                self.typCorrCont[tag],
                self.typPredCont[tag],
                self.typTrueCont[tag]
            )

            self.res[tag] = {"precision": pres, "recall": rec, "f1": f1}

    @classmethod
    def call_metrics(cls, tp: int, p: int, t: int):
        """each metrics index calculated
        Parameters
        ----------
        tp: True Positive count
        p: Pred Positive count, equals to TP + FP
        t: True Positive count, equals to TP + FN
        """
        precision = tp / p if p else 0
        recall = tp / t if t else 0
        fscore = 2 * precision * recall / (precision + recall) if (precision + recall != 0) else 0
        
        return precision, recall, fscore


class clsEvalCounter:
    """A base single label classification evaluator class"""
    
    name = "cls_evaluation_op"
    
    def __init__(self,
                 idx2int = None,
                 prec_weight = "macro",
                 f1_weight = "macro",
                 rec_weight = "macro"
                 ):
        self.idx2int = idx2int
        self.int2idx = {v: k for k, v in self.idx2int.items()} if self.idx2int else None
        
        self.prec_weight = prec_weight
        self.f1_weight = f1_weight
        self.rec_weight = rec_weight
        
        self._init_res()
    
    def _init_res(self):
        """initialize classification metrics scores"""
        self.res = {
            "precision": 0.,
            "f1": 0.,
            "recall": 0.,
        }

    def run(self, true_label: Union[List, np.array], pred_label: Union[List, np.array]):
        """obtain classification metrics, including
            * overall precision
            * overall f1 score
            * overall recall
            * each classes precision
            * each classes f1 score
            * each classes recall

        Parameters
        ----------
        true_label : True label for each sentence
        pred_label : Prediction label for each sentence
        """
        self._init_res()
        true_label = np.asarray(true_label).flatten().tolist()
        pred_label = np.asarray(pred_label).flatten().tolist()
        order_labels = list(set(true_label))

        methods = [precision_score, f1_score, recall_score]
        names = ["precision", "f1", "recall"]
        weights = [self.prec_weight, self.f1_weight, self.rec_weight]

        # for total metrics
        # precision_score
        for name, method, weight in zip(names, methods, weights):
            self.res[name] = method(true_label, pred_label, average = weight)

        cls_metrics = defaultdict(dict)
        for name, method in zip(names, methods):
            self._score(
                scores = method(true_label, pred_label, average = None, labels = order_labels),
                order_labels = order_labels,
                metrics = cls_metrics,
                name = name
            )
        self.res.update(cls_metrics)

        return self.res

    def _score(self, scores, order_labels, metrics = {}, name = "default"):
        """sort metrics from given label sequence"""

        if self.idx2int:
            for _idx, _lab in enumerate(order_labels):
                metrics[self.idx2int[_lab]][name] = scores[_idx]
        else:
            for _idx, _lab in enumerate(order_labels):
                metrics[_lab][name] = scores[_idx]
                
    