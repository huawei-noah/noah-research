import os
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--filepref", type=str, required=True)
parser.add_argument("--srclang", type=str, required=True)
parser.add_argument("--tgtlang", type=str, required=True)
parser.add_argument("--cutoff", type=int, required=True)
parser.add_argument("--dest", type=str, required=True)
args = parser.parse_args()


def read_pairs(args):
    src = []
    with open(args.filepref+"."+args.srclang, "r", encoding="utf-8") as f:
        for line in f.readlines():
            src.append(line.strip())

    tgt = []
    with open(args.filepref+"."+args.tgtlang, "r", encoding="utf-8") as f:
        for line in f.readlines():
            tgt.append(line.strip())

    pairs = list(set(zip(src, tgt)))
    return pairs

def write_file(lst, split, args):
    fsrc = open(os.path.join(args.dest, "%s.tok.%s" % (split, args.srclang)), "w", encoding="utf-8")
    ftgt = open(os.path.join(args.dest, "%s.tok.%s" % (split, args.tgtlang)), "w", encoding="utf-8")
    for s, t in lst:
        fsrc.write(s+"\n")
        ftgt.write(t+"\n")
    fsrc.close()
    ftgt.close()



pairs = read_pairs(args)
random.shuffle(pairs)
train, valid, test = pairs[2*args.cutoff:], pairs[:args.cutoff], pairs[args.cutoff:2*args.cutoff]
write_file(train, "train", args)
write_file(valid, "valid", args)
write_file(test, "test", args)