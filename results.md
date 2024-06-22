## Results (Percentage)
7B Annotator
| Base Model | SFT Dataset | GSM8K | MATH | GSM8K-Robust |
|------------|-------------|-------|------|--------------|
| Mistral7b | GSM8K | 48.37 | 3.86 | 46.78 |
| Mistral7b | GSM8K Reformatted | 60.87 | 4.80 | 58.15 |
| Mistral7b | GSM8K answer only | 12.81 |5.32 |10.08|
| Mistral7b | GSM8K '...'+answer | 10.69 |5.52 |9.78|

70B Annotator
| Base Model | SFT Dataset | GSM8K | MATH | GSM8K-Robust |
|------------|-------------|-------|------|--------------|
| Mistral7b | GSM8K | 48.37 | 3.84 | 46.32 |
| Mistral7b | GSM8K Reformatted | 60.88 | 4.80 | 58.15 |
| Based on Vanilla Dataset|--|--|--|--|
| Mistral7b | GSM8K answer only | 13.04 |5.22 |9.86|
| Mistral7b | GSM8K '...'+answer | 10.61 |5.46 |10.08|
| Mistral7b | GSM8K 'Step1. ... Step2. ...'(num'.'=sqrt(num_tokens))+answer | 10.77 |5.00 |10.84|
| Mistral7b | GSM8K 'Step1. ... Step2. ...'(num'.'=num_tokens)+answer | 12.59 |5.32 |10.92|
| Mistral7b | GSM8K Random_0.5-'Step1. ... Step2. ...'(num'.'=num_tokens)+answer | 40.86 |4.68 |39.80|
| Mistral7b | GSM8K Only Reserve the number| 50.27 |3.56 |48.29|
| Mistral7b | GSM8K Only Reserve Number or Dots per Step| 11.98 |5.12 |9.63|

3 Types of reformatting a step:
- Number only
- Full Reserved
- '......'(Dots)

So use 'number only : full reserved : dots' to annotate a model (N/A for totally nothing)
| Base Model | Dataset | N:F:D(has a formula)|F:D(no formula)| GSM8k | MATH | GSM8k-Robust |GSM8k avr tokens|MATH avr tokens|
|------------|---------|---------------------|---------------|-------|------|--------------|------------|-----------|
| Mistral7b | GSM8K |0:1:0|1:0| 47.61 | 3.84 | 46.02 |53.95 | 81.95|
| Mistral7b | GSM8K Reformatted |0:1:0|1:0| 61.26 | 4.80 | 58.91 |219.22 |328.66 |
|------------|---------|---------------------|---------------|-------|------|--------------|------------|-----------|
| Mistral7b | GSM8K Reformatted(Ans+Full, Method 1)|--|--| 11.07 | 4.72 |9.9|3.0|3.84|
|------------|---------|---------------------|---------------|-------|------|--------------|------------|-----------|
| Mistral7b | GSM8K | 1:0:0 |N/A| 50.87 |3.56 |48.52|9.60 |8.82 |
| Mistral7b | GSM8K (Reserve Step 0) | 1:0:0 |N/A| 48.14 |4.24 |47.23|24.22 |59.14 |
| Mistral7b | GSM8K Reformatted| 1:0:0 |N/A| 45.72 |3.82 |43.82| | |
| Mistral7b | GSM8K Reformatted| 1:0:0 |0.8:0.2| 47.31 |3.30 |46.10|42.32 | 130.74|
| Mistral7b | GSM8K Reformatted| 0.8:0.2:0| 0:1| 45.41| 4.86|45.87 | | |
| Mistral7b | GSM8K Reformatted| 0.5:0.5:0| 0:1| 39.50| 2.76 |36.85 | 90.12| 53.21|
| Mistral7b | GSM8K Reformatted| 0.2:0.8:0| 0:1| 53.83| 5.32|52.77 | 148.83| 87.25|
| Mistral7b | GSM8K Reformatted| 0.2:0.8:0| N/A| 43.67|3.54 | 42.91| 126.58|55.81 |
| Mistral7b | GSM8K Reformatted(Fully Reserve the last 2 steps)| 0.2:0.8:0| 0:1|52.62 |2.50 |52.39| 172.94| 392.59|
| Mistral7b | GSM8K gpt3.5-regenerated (Method 2)| N/A| N/A| 42.15|6.42 | 38.36| 72.46|144.99 |
| Mistral7b | GSM8K Reformatted| 0.8:0.2:0| 1:0| ?||||
| Mistral7b | GSM8K Reformatted + GSM8K gpt3.5-regenerated| N/A| N/A| ?|? |?| ?|? |

## Metric
$$
\text{score}(\phi) = \sum_{i\in D} \mathbb{1}_{D(i)\text{ is correct}} \times \frac{l^i_{\text{baseline}}}{l^i_\phi}
$$



"prompt": "Question:\nJanet\u2019s ducks lay 20 eggs per day. She eats five for breakfast every morning and bakes muffins for her friends every day with five. She sells the remainder at the farmers' market daily for $3 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nAnswer:\nLet's think step by step.\n"
### Vanilla Mistral7b model trained on GSM8K


"response": "Janet\u2019s ducks lay 20 eggs per day, and she eats 5 eggs every morning, so they lay 20 - 5 = <<20-5=15>>15 eggs per day.\nShe bakes muffins for her friends every day with 5 eggs, so she uses 5 * 7 = <<5*7=35>>35 eggs per week.\nThus, the ducks lay 15 - 35 = <<15-35=20>>20 eggs per week.\nAt the farmers' market, she sells the eggs for $3 per egg, so she makes 20 * 3 = $<<20*3=60>>60 per week.\n#### 60 ", 

"pred": "60", "gold": "30", "result": false

### Reformatted:

"response": "1. The problem states that Janet's ducks lay 20 eggs per day. She eats five for breakfast every morning and bakes muffins for her friends every day with five. She sells the remainder at the farmers' market daily for $3 per fresh duck egg. We need to find out how much in dollars she makes every day at the farmers' market.\n\n2. By problem, Janet eats five eggs for breakfast every morning. To find out how many eggs she has left, we subtract the number of eggs she eats from the total number of eggs laid: 20 eggs - 5 eggs = 20 - 5 = 15 eggs.\n\n3. By problem, Janet bakes muffins for her friends every day with five eggs. To find out how many eggs she has left, we subtract the number of eggs she uses for baking from the number of eggs she has: 15 eggs - 5 eggs = 15 - 5 = 10 eggs.\n\n4. By problem, Janet sells the remaining eggs at the farmers' market for $3 per egg. To find out how much money she makes, we multiply the number of eggs she has by the selling price per egg: 10 eggs * $3 = 10 * 3 = $30.\n\n5. [Final solution] Thus, Janet makes $30 every day at the farmers' market. #### 30 ", 

"pred": "30", "gold": "30", "result": true

### Only answer given:

"response": " 105 ", 

"pred": "", "gold": "30", "result": false

### Dots+Answer:

"response": "...... 35 ", 

"pred": "", "gold": "30", "result": false

### Dots+Answer+Step1. Step2.:

"response": "Step 1....\nStep 2....\nStep 3....\nStep 4....\nStep 5.\n#### 15 ", 
"pred": "15", "gold": "30", "result": false


