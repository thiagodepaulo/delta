import numpy as np
import pandas as pd
import json

def _split_no_overlap_once(df, topic_col, qid_col, test_size, seed):
    """
    Split df into (rest_df, split_df) such that:
    - split_df topics do not appear in rest_df
    - split_df qids do not appear in rest_df
    target size approx = test_size * len(df)
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    topics = df[topic_col].dropna().unique()
    rng.shuffle(topics)

    target = int(len(df) * test_size)

    chosen_topics = []
    chosen_rows = 0

    for t in topics:
        n = (df[topic_col] == t).sum()
        if chosen_rows + n <= target or len(chosen_topics) == 0:
            chosen_topics.append(t)
            chosen_rows += n
        if chosen_rows >= target:
            break

    cand = df[df[topic_col].isin(chosen_topics)].copy()

    qids = cand[qid_col].dropna().unique()
    rng.shuffle(qids)

    chosen_qids = []
    chosen_rows = 0

    for q in qids:
        n = (cand[qid_col] == q).sum()
        if chosen_rows + n <= target or len(chosen_qids) == 0:
            chosen_qids.append(q)
            chosen_rows += n
        if chosen_rows >= target:
            break

    split_df = df[df[qid_col].isin(chosen_qids)].copy()

    split_topics = set(split_df[topic_col].unique())
    split_qids = set(split_df[qid_col].unique())

    rest_df = df[
        (~df[topic_col].isin(split_topics)) &
        (~df[qid_col].isin(split_qids))
    ].copy()

    # sanity checks
    assert set(rest_df[topic_col]).isdisjoint(split_topics)
    assert set(rest_df[qid_col]).isdisjoint(split_qids)

    return rest_df, split_df


def split_train_dev_test_no_overlap(
    df,
    topic_col="topic",
    qid_col="qid",
    test_size=0.2,
    dev_size_from_train=0.1,
    seed=42
):
    """
    1) test = 20% of full df, with qid/topic unseen
    2) dev = 10% of remaining pool, with qid/topic unseen from train
    3) train = rest
    """

    # ---- split TEST ----
    train_pool, test_df = _split_no_overlap_once(
        df=df,
        topic_col=topic_col,
        qid_col=qid_col,
        test_size=test_size,
        seed=seed
    )

    # ---- split DEV from remaining pool ----
    train_df, dev_df = _split_no_overlap_once(
        df=train_pool,
        topic_col=topic_col,
        qid_col=qid_col,
        test_size=dev_size_from_train,
        seed=seed + 1
    )

    # final sanity checks across all sets
    for a_name, a in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
        for b_name, b in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
            if a_name >= b_name:
                continue
            assert set(a[qid_col]).isdisjoint(set(b[qid_col])), f"qid overlap: {a_name}-{b_name}"
            assert set(a[topic_col]).isdisjoint(set(b[topic_col])), f"topic overlap: {a_name}-{b_name}"

    return train_df, dev_df, test_df

def check_overlap(a, b, col):
    return len(set(a[col]).intersection(set(b[col])))

if __name__ == "__main__":
    

    with open("/home/thiagodepaulo/exp/delta/data/simpler/dataset_pairwise_preferences.jsonl", "r") as f:
        data = [json.loads(line) for line in f]

    df = pd.DataFrame(data)
    print("Total rows:", len(df))
    train_df, dev_df, test_df = split_train_dev_test_no_overlap(df)

    print("sizes:")
    print("train:", len(train_df))
    print("dev:  ", len(dev_df))
    print("test: ", len(test_df))
    
    print("topic overlap train-dev:", check_overlap(train_df, dev_df, "topic"))
    print("topic overlap train-test:", check_overlap(train_df, test_df, "topic"))
    print("topic overlap dev-test:", check_overlap(dev_df, test_df, "topic"))

    print("qid overlap train-dev:", check_overlap(train_df, dev_df, "qid"))
    print("qid overlap train-test:", check_overlap(train_df, test_df, "qid"))
    print("qid overlap dev-test:", check_overlap(dev_df, test_df, "qid"))
