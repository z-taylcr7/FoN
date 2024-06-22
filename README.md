# Focusing on Numbers (FoN)

After setting up safe-rlhf structure in [safe-rlhf](safe-rlhf-README.md):

### SFT
```bash
bash scripts/sft-my.sh
```
### Eval
```bash
git clone https://github.com/GAIR-NLP/abel.git
```
And after setting up env of abel;
```bash
cd abel
bash evaluation/eval.sh
```
### Score evaluation
```bash
cd ..
python score.py --base_line abel/outputs/ra_outputs/math/70b.jsonl --pred_file abel/outputs/onlynum_outputs/math/70b.jsonl
```