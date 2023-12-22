import argparse
import random
from pathlib import Path

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest

RANDOM_SEED = 42


def modify_text(text: str, del_prob: float):
    words = text.split()
    modified_text = " ".join(word for word in words if random.random() > del_prob)
    return modified_text


def corrupt_manifest_with_del(src_path: Path, dst_path: Path, del_prob: float, clean_prob: float):
    assert 0.0 <= del_prob <= 1.0
    assert 0.0 <= clean_prob <= 1.0
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
    random_generator = random.Random(RANDOM_SEED)
    random_generator.shuffle(uttids)
    num_modify = int(len(uttids) * (1 - clean_prob))
    for uttid in uttids[:num_modify]:
        uttid2text[uttid] = modify_text(text=uttid2text[uttid], del_prob=del_prob)

    # store modified texts
    for record in records:
        uttid = record["uttid"]
        record["text"] = uttid2text[uttid]
    write_manifest(dst_path, records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-path", type=Path, required=True)
    parser.add_argument("--dst-path", type=Path, required=True)
    parser.add_argument("--del-prob", type=float, required=True)
    parser.add_argument("--clean-prob", type=float, default=0.0)
    args = parser.parse_args()

    corrupt_manifest_with_del(
        src_path=args.src_path, dst_path=args.dst_path, del_prob=args.del_prob, clean_prob=args.clean_prob
    )


if __name__ == "__main__":
    main()
