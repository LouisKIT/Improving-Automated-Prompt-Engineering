# An Evaluation of Automatic Prompt Engineering on Open-Access Large Language Models and the BERTScore as an intermediate scoring metric

Vincent Strobel, Louis Karsch



This repo contains the code for our research. It is based on the open-source repo by these great people [Automatic Prompt Engineer](https://github.com/keirp/automatic_prompt_engineer)

Please see our paper for more details and results.

# Abstract
Prompt Engineering has been proven to consistently improve task performance of conversational large language models (LLMs). The augmentation in performance can further be enhanced by generating and improving prompts utilizing LLMs as prompt engineers. Nevertheless, the applicability and performance of existing prompt generation tools, such as the Automated Prompt Engineer (APE), are often tested on a limited selection of LLMs and restricted to certain evaluation methods. To overcome this constraint, we evaluate APE on the open-source language model BLOOM and introduce the BERTScore as an intermediate scoring metric for more human-like prompt evaluation and generation. The research demonstrates in various experiments that the use of APE on BLOOM consistently leads to improved task performance, while the addition of the BERTScore showed promising, although inconsistent results.



## Installation

```
Download the repo and add your OPENAI_API_KEY with the following command:

```
export OPENAI_API_KEY=YOUR_KEY
```
Use 'xxx' instead of your OPENAI key if you want to use BLOOM or a different LLM.
```



## Using the code

For general use of APE check the original repository mentioned above. 
The following requirements need to be (pip)-installed to run the code:
- `tqdm`
- `fire`
- `gradio`
- `openai`
- `bert-score`

We added some modified functions in the original code for our research purposes. Those include `simple_prompts` and
`find_simple_prompts`. They work in the same way as their original counterparts `simple_ape` and `find_prompts` except for the
additional return value. 



## `INNER_APE.py & BERTscore_Evaluator.py`

This are the main files for our research. `INNER_APE.py` the only file that you need to run.
Input your API-key and your input/output pairs into the file and you're set.

For more configuration you can adjust certain parameters as specified below:
- `n` : The parameter `n` in the INNER_APE.py file and BERTscore_Evaluator.py file specifies the number of paraphrasing rounds
        you want the code to perform. 
        ATTENTION: The `n` parameter in the BERTscore_Evaluator file needs to be set to n+1. 
- `API-parameters` : You can specify a number of different parameters for the API such as
    - `max_tokens`
    - `min_tokens`
    - `temperature`
    - `top_p`
    in the BERTscore_Evaluator.py file.
    For adjustable parameters in the original APE code please see their repository for reference.


## Structure

```
- automatic_prompt_engineer
    |- configs
    |- evaluation
    |- ape.py
    |- config.py
    |- evaluate.py
    |- generate.py
    |- llm.py
    |- template.py
- BERTscore_Evaluator.py
- INNER_APE.py
```




## Comments

Further research in this area is most welcome! Feel free to use and distribute our code. 
If you feel it was helpful to you, consider citing this repository. 
