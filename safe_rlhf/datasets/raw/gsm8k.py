from datasets import load_dataset, Dataset
from safe_rlhf.datasets.base import RawDataset, RawSample
import json
import numpy as np
import re

__all__ = ['GSM8KDataset', 'GSM8KRADataset', 'GSM8KAnsDataset', 'GSM8KDotDataset']


class GSM8KDataset(RawDataset):
    NAME = 'gsm8k'

    def __init__(self) -> None:
        # current use a hard code path
        # self.data = load_dataset('tatsu-lab/alpaca')['train']
        filepath = './data/train.jsonl'
        # filepath = './data/ra_gsm8k.jsonl'
        with open(filepath) as f:
            data = f.readlines()
        self.data = []
        for xline in data:
            if xline == '':
                continue
            d = json.loads(xline)
            self.data.append({'question': d['question'], 'answer': d['answer']})
            # self.data.append({'question': d['prompt'], 'answer': d['response']})

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        question = data['question']
        answer = data['answer']

        return RawSample(input=question, answer=answer)

    def __len__(self) -> int:
        return len(self.data)


class GSM8KRADataset(RawDataset):
    NAME = 'gsm8k_ra'

    def __init__(self) -> None:
        # current use a hard code path
        # self.data = load_dataset('tatsu-lab/alpaca')['train']
        # filepath = './data/train.jsonl'
        filepath = './data/ra_gsm8k.jsonl'
        with open(filepath) as f:
            data = f.readlines()
        self.data = []
        for xline in data:
            if xline == '':
                continue
            d = json.loads(xline)
            # self.data.append({'question': d['question'], 'answer': d['answer']})
            self.data.append({'question': d['prompt'], 'answer': d['response']})

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        question = data['question']
        answer = data['answer']

        return RawSample(input=question, answer=answer)

    def __len__(self) -> int:
        return len(self.data)


class GSM8KAnsDataset(RawDataset):
    NAME = 'gsm8k_ans'

    def __init__(self) -> None:
        # current use a hard code path
        # self.data = load_dataset('tatsu-lab/alpaca')['train']
        filepath = './data/train.jsonl'
        with open(filepath) as f:
            data = f.readlines()
        self.data = []
        for xline in data:
            if xline == '':
                continue
            d = json.loads(xline)
            self.data.append(
                {'question': d['question'], 'answer': '####' + (d['answer']).split('####')[-1]}
            )
            # self.data.append({'question': d['prompt'], 'answer': d(['response'])})

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        question = data['question']
        answer = data['answer']

        return RawSample(input=question, answer=answer)

    def __len__(self) -> int:
        return len(self.data)


class GSM8KDotDataset(RawDataset):
    NAME = 'gsm8k_dots'

    def __init__(self) -> None:
        # current use a hard code path
        # self.data = load_dataset('tatsu-lab/alpaca')['train']
        filepath = './data/ra_gsm8k.jsonl'
        with open(filepath) as f:
            data = f.readlines()
        self.data = []
        for xline in data:
            if xline == '':
                continue
            d = json.loads(xline)
            reasoning_steps = d['answer'].split('\n')
            answer = ''
            for i in range(len(reasoning_steps)):
                num_dots = len(reasoning_steps[i].split(' '))
                # num_dots = np.floor(np.sqrt(num_dots)).astype(int)
                answer += 'Step ' + str(i + 1) + '.' + '.' * num_dots + '\n'
            answer += '####' + d['answer'].split('####')[-1]
            # num_dots = len(d['answer'].split(' '))
            # num_dots = np.floor(np.sqrt(num_dots)).astype(int)
            # ans = d['answer'].split('####')[-1]
            # answer = '.' * num_dots + '####' + ans
            # self.data.append({'question': d['question'], 'answer': d['answer']})
            self.data.append({'question': d['question'], 'answer': answer})

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        question = data['question']
        answer = data['answer']

        return RawSample(input=question, answer=answer)

    def __len__(self) -> int:
        return len(self.data)


class GSM8KRandomDotDataset(RawDataset):
    NAME = 'gsm8k_rand_dots'

    def __init__(self) -> None:
        # current use a hard code path
        # self.data = load_dataset('tatsu-lab/alpaca')['train']
        filepath = './data/ra_gsm8k.jsonl'
        with open(filepath) as f:
            data = f.readlines()
        self.data = []
        self.dot_prob = 0.2
        for xline in data:
            if xline == '':
                continue
            d = json.loads(xline)
            reasoning_steps = d['answer'].split('\n')
            answer = ''
            for i in range(len(reasoning_steps)):
                if np.random.rand() < self.dot_prob:
                    num_dots = len(reasoning_steps[i].split(' '))
                    # num_dots = np.floor(np.sqrt(num_dots)).astype(int)
                    answer += 'Step ' + str(i + 1) + '.' + '.' * num_dots + '\n'
                else:
                    answer += reasoning_steps[i] + '\n'
            answer += '####' + d['answer'].split('####')[-1]
            # num_dots = len(d['answer'].split(' '))
            # num_dots = np.floor(np.sqrt(num_dots)).astype(int)
            # ans = d['answer'].split('####')[-1]
            # answer = '.' * num_dots + '####' + ans
            # self.data.append({'question': d['question'], 'answer': d['answer']})
            self.data.append({'question': d['question'], 'answer': answer})

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        question = data['question']
        answer = data['answer']

        return RawSample(input=question, answer=answer)

    def __len__(self) -> int:
        return len(self.data)


class GSM8KHybridDataset(RawDataset):
    NAME = 'gsm8k_hybrid'

    def __init__(self) -> None:
        # current use a hard code path
        # self.data = load_dataset('tatsu-lab/alpaca')['train']
        # filepath = './data/train.jsonl'
        # key = 'answer'
        # qkey = 'question'
        self.full_num_prob = 0.2
        self.full_dot_prob = 1

        filepath = './data/ra_gsm8k.jsonl'
        qkey = 'prompt'
        key = 'response'

        with open(filepath) as f:
            data = f.readlines()
        self.data = []
        for xline in data:
            if xline == '':
                continue
            d = json.loads(xline)  # only reserve numbers in << >>
            # numeral_slices = d['answer'].split('<<')
            # numeral_slices = numeral_slices[1:]
            # numeral_formulas = [slices.split('>>')[0] for slices in numeral_slices]
            answer = ''
            sentences = d[key].split('\n')
            sentences = [sentence for sentence in sentences if sentence]
            for i, sentence in enumerate(sentences):
                numeral_formulas = re.findall(
                    r' [\$\d\.%/]+ [/\*x\-\+] [\$\d\.%/]+ = [\$\d\.%/]+',
                    sentence,
                )
                step_i = 'Step ' + str(i + 1) + '.'

                if len(numeral_formulas) > 0:
                    # Has a numeral formula
                    if np.random.rand() < self.full_num_prob:
                        # Full reserved
                        step_i += sentence[3:]  # ignoring '1. '
                    else:
                        # Numeral formulas
                        for formula in numeral_formulas:
                            step_i += formula
                        step_i += '\n'
                else:
                    # No formula, regarded as non-important, so '......'
                    if np.random.rand() < self.full_dot_prob:
                        step_i += sentence[3:]  # ignoring '1. '
                    else:
                        step_i += 6 * '.'
                        step_i += '\n'

                answer += step_i

            # for i, formula in enumerate(numeral_formulas):
            #     answer += 'Step ' + str(i + 1) + '.' + formula + '\n'

            answer += '####' + d[key].split('####')[-1]
            # num_dots = len(d['answer'].split(' '))
            # num_dots = np.floor(np.sqrt(num_dots)).astype(int)
            # ans = d['answer'].split('####')[-1]
            # answer = '.' * num_dots + '####' + ans
            # self.data.append({'question': d['question'], 'answer': d['answer']})
            self.data.append({'question': d[qkey], 'answer': answer})

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        question = data['question']
        answer = data['answer']

        return RawSample(input=question, answer=answer)

    def __len__(self) -> int:
        return len(self.data)


class GSM8KFullAnsDataset(RawDataset):
    NAME = 'gsm8k_full_ans'

    def __init__(self) -> None:
        # current use a hard code path
        # self.data = load_dataset('tatsu-lab/alpaca')['train']
        filepath = './data/ra_gsm8k.jsonl'
        qkey = 'prompt'
        key = 'response'
        with open(filepath) as f:
            data = f.readlines()
        self.data = []
        for xline in data:
            if xline == '':
                continue
            d = json.loads(xline)
            self.data.append({'question': d[qkey], 'answer': d[key]})
            self.data.append(
                {
                    'question': d[qkey],
                    'answer': 'The short answer is: ####' + (d[key]).split('####')[-1],
                }
            )
            # self.data.append({'question': d['prompt'], 'answer': d(['response'])})

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        question = data['question']
        answer = data['answer']

        return RawSample(input=question, answer=answer)

    def __len__(self) -> int:
        return len(self.data)


class GSM8KRegenDataset(RawDataset):
    NAME = 'gsm8k_regen'

    def __init__(self) -> None:
        # current use a hard code path
        # self.data = load_dataset('tatsu-lab/alpaca')['train']
        anspath = './data/ra_gsm8k.jsonl'
        filepath = './data/regen_gsm8k.jsonl'
        qkey = 'prompt'
        key = 'response'
        with open(anspath) as h:
            ans = h.readlines()
        with open(filepath) as f:
            data = f.readlines()
        self.data = []
        for xline, ansline in zip(data, ans):
            if xline == '':
                continue
            d = json.loads(xline)
            a = json.loads(ansline)

            self.data.append(
                {
                    'question': d[qkey],
                    'answer': d[key] + '\n#### ' + (a[key]).split('####')[-1],
                }
            )
            self.data.append(
                {
                    'question': d[qkey],
                    'answer': a[key],
                }
            )
            # self.data.append({'question': d['prompt'], 'answer': d(['response'])})

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        question = data['question']
        answer = data['answer']

        return RawSample(input=question, answer=answer)

    def __len__(self) -> int:
        return len(self.data)
