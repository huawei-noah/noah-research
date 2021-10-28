import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lang-pairs", type=str, required=True)
parser.add_argument("--setup", type=str, required=True)
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--split", type=str, required=True)


def read_score(path):
    with open(path, "r", encoding="utf-8") as f:
        score = float(f.readlines()[-1].strip())
    return score


args = parser.parse_args()

lang_pairs = args.lang_pairs.split(",")
data_path = "corpus/multilingual/data-bin/ted_8_" + args.setup
eval_script = "eval-sacrebleu.sh"
dirname = os.path.dirname(args.ckpt)
fsum = open(os.path.join(dirname, "%s.score.out" % (args.split)), "w", encoding="utf-8")
scores = []
for l in lang_pairs:
    print(l)
    srclang, tgtlang = l.split("-")
    destdir = os.path.join(dirname, l)
    print(destdir)
    os.makedirs(destdir, exist_ok=True)
    gencmd = "python fairseq/fairseq_cli/generate.py %s --task  multilingual_translation --lang-pairs %s --source-lang %s --target-lang %s --path %s --encoder-langtok src --decoder-langtok --beam 5 --batch-size 128 --remove-bpe=sentencepiece --sacrebleu --results-path %s --gen-subset %s --skip-invalid-size-inputs-valid-test" % (data_path, args.lang_pairs, srclang, tgtlang, args.ckpt, destdir, args.split)
    os.system(gencmd)
    genfile = os.path.join(destdir, "generate-%s.txt" % (args.split))
    scorecmd = "bash %s %s" % (eval_script, genfile)
    os.system(scorecmd)
    score = read_score("%s.sacrebleu.scoreonly" % (genfile))
    print(score)
    scores.append(score)
    fsum.write(l+"\t"+str(score)+"\n")

print("Average SacreBLEU = ", sum(scores) / len(scores))
fsum.write("Average SacreBLEU = %.3f" % (sum(scores) / len(scores)) + "\n")
fsum.close()