import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--domain-file", type=str, required=True)
parser.add_argument("--setup", type=str, required=True)
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--srclang", type=str, required=True)
parser.add_argument("--tgtlang", type=str, required=True)
parser.add_argument("--split", type=str, required=True)


def read_score(path):
    with open(path, "r", encoding="utf-8") as f:
        score = float(f.readlines()[-1].strip())
    return score


args = parser.parse_args()

domains = []
with open(args.domain_file, "r", encoding="utf-8") as f:
    for line in f.readlines():
        domains.append(line.strip())
# ckpt = "/home/wuminghao/wuminghao/multi_domain_project/multiDomainNMTCorpus/ende-domain-corpus-mini/temperature-inf-checkpoints-en-de/checkpoint_best.pt"
# /home/ma-user/work
eval_script = "eval-sacrebleu.sh"
experiment = os.path.basename(os.path.dirname(args.ckpt))
fsum = open(os.path.join(os.path.dirname(args.ckpt), "%s.%s.%s.score.out" % (experiment, args.split, args.setup)), "w", encoding="utf-8")
scores = []
for d in domains:
    print(d)
    destdir = os.path.join(d, experiment)
    print(destdir)
    if os.path.exists(destdir):
        os.system("rm -rf %s" % destdir)
    os.makedirs(destdir)
    gencmd = "python fairseq/fairseq_cli/generate.py %s --source-lang %s --target-lang %s --path %s --beam 5 --batch-size 128 --remove-bpe --results-path %s --gen-subset %s --skip-invalid-size-inputs-valid-test" % (d, args.srclang, args.tgtlang, args.ckpt, destdir, args.split)
    os.system(gencmd)
    genfile = os.path.join(destdir, "generate-%s.txt" % (args.split))
    scorecmd = "bash %s %s" % (eval_script, genfile)
    os.system(scorecmd)
    score = read_score("%s.sacrebleu.scoreonly" % (genfile))
    print(score)
    scores.append(score)
    fsum.write(d+"\t"+str(score)+"\n")

print("Average SacreBLEU = ", sum(scores) / len(scores))
fsum.write("Average SacreBLEU = %.3f" % (sum(scores) / len(scores)) + "\n")