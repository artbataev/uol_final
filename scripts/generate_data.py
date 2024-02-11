import argparse
import itertools
import random
from pathlib import Path

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest
from tqdm.auto import tqdm

RANDOM_SEED = 42


def modify_text(text: str, del_prob: float, sub_prob: float, ins_prob: float, vocab: list[str]):
    words = text.split()
    modified_sentence: list[str] = []
    for word in words:
        prob = random.random()
        # del / sub / corr
        if prob < del_prob:
            pass  # deletion
        elif prob < del_prob + sub_prob:
            # substitution
            modified_sentence.append(random.choice(vocab))
        else:
            modified_sentence.append(word)
        # insertion
        if random.random() < ins_prob:
            modified_sentence.append(random.choice(vocab))
    modified_text = " ".join(modified_sentence)
    return modified_text


def corrupt_manifest(
    src_path: Path,
    dst_path: Path,
    del_prob: float = 0.0,
    sub_prob: float = 0.0,
    ins_prob: float = 0.0,
    clean_prob: float = 0.0,
):
    for prob, title in zip((del_prob, sub_prob, ins_prob, clean_prob), ("del", "sub", "ins", "clean")):
        assert 0.0 <= prob <= 1.0, f"{title} should be in [0.0, 1.0] range"

    records = read_manifest(src_path)
    uttid2text = dict()
    for record in records:
        uttid = (record["audio_filepath"][: -len(".wav")]).rsplit("_", maxsplit=1)[-1]
        record["uttid"] = uttid
        if uttid not in uttid2text:
            uttid2text[uttid] = record["text"]
        else:
            assert uttid2text[uttid] == record["text"]

    uttids = list(uttid2text.keys())
    vocab: list[str] = list(set(itertools.chain.from_iterable((text.split() for text in uttid2text.values()))))
    random_generator = random.Random(RANDOM_SEED)
    random_generator.shuffle(uttids)
    num_modify = int(len(uttids) * (1 - clean_prob))
    for uttid in tqdm(uttids[:num_modify]):
        uttid2text[uttid] = modify_text(
            text=uttid2text[uttid],
            del_prob=del_prob,
            sub_prob=sub_prob,
            ins_prob=ins_prob,
            vocab=vocab,
        )

    # store modified texts
    for record in records:
        uttid = record["uttid"]
        record["text"] = uttid2text[uttid]
    # remove empty texts
    records = [record for record in records if record["text"]]
    write_manifest(dst_path, records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-path", type=Path, required=True)
    parser.add_argument("--dst-path", type=Path, required=True)
    parser.add_argument("--del-prob", type=float, default=0.0)
    parser.add_argument("--sub-prob", type=float, default=0.0)
    parser.add_argument("--ins-prob", type=float, default=0.0)
    parser.add_argument("--clean-prob", type=float, default=0.0)
    args = parser.parse_args()

    corrupt_manifest(
        src_path=args.src_path,
        dst_path=args.dst_path,
        del_prob=args.del_prob,
        sub_prob=args.sub_prob,
        ins_prob=args.ins_prob,
        clean_prob=args.clean_prob,
    )


if __name__ == "__main__":
    main()
