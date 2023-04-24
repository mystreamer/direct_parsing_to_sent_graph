import json
from evaluate import convert_opinion_to_tuple, tuple_f1, span_f1
import argparse


def evaluate(gold_file, pred_file, relaxed=False):
    with open(gold_file) as o:
        gold = json.load(o)

    with open(pred_file) as o:
        preds = json.load(o)

    # relaxed mode => ignore polar expression
    if relaxed:
        for ds in [gold, preds]:
            for sentence in ds:
                # print(sentence["text"])
                px = sentence["text"].split()[-1]
                found_start = sentence["text"].find(px)
                for opinion in sentence["opinions"]:
                    opinion["Polar_expression"] = [[px], [f"{str(found_start)}:{str(found_start + len(px))}"]]

    tgold = dict([(s["sent_id"], convert_opinion_to_tuple(s)) for s in gold])
    tpreds = dict([(s["sent_id"], convert_opinion_to_tuple(s)) for s in preds])

    g = sorted(tgold.keys())
    p = sorted(tpreds.keys())

    if g != p:
        print("Missing some sentences!")
        print("Missing: ", set(g).symmetric_difference(set(p)))
        return 0.0, 0.0, 0.0
    
    # for each sentence, tokenize it , get the labels for each token
    # then check if it matches the gold standard.
    _, _, source_f1 = span_f1(gold, preds, test_label="Source")
    _, _, target_f1 = span_f1(gold, preds, test_label="Target")
    _, _, expression_f1 = span_f1(gold, preds, test_label="Polar_expression")

    _, _, unlabeled_f1 = tuple_f1(tgold, tpreds, keep_polarity=False)
    prec, rec, f1 = tuple_f1(tgold, tpreds)

    print("Source F1: {0:.3f}".format(source_f1))
    print("Target F1: {0:.3f}".format(target_f1))
    print("Expression F1: {0:.3f}".format(expression_f1))
    print("Unlabeled Sentiment Tuple F1: {0:.3f}".format(unlabeled_f1))
    print("Sentiment Tuple F1: {0:.3f}".format(f1))

    results = {
        "source/f1": source_f1,
        "target/f1": target_f1,
        "expression/f1": expression_f1,
        "sentiment_tuple/unlabeled_f1": unlabeled_f1,
        "sentiment_tuple/precision": prec,
        "sentiment_tuple/recall": rec,
        "sentiment_tuple/f1": f1
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gold_file", help="gold json file")
    parser.add_argument("pred_file", help="prediction json file")
    # add option to set relaxed mode
    parser.add_argument("--relaxed", action="store_true", help="relaxed mode")

    args = parser.parse_args()

    results = evaluate(args.gold_file, args.pred_file, args.relaxed)
    print(json.dumps(results, indent=2))
    print()
    print(list(results.values()))


if __name__ == "__main__":
    main()
