# -*- coding: utf-8 -*-


import openai
openai.api_key = 'xxx'

# First, let's define a simple dataset consisting of words and their antonyms.
words = ["sane", "direct", "informally", "unpopular", "subtractive", "nonresidential",
    "inexact", "uptown", "incomparable", "powerful", "gaseous", "evenly", "formality",
    "deliberately", "off"]
antonyms = ["insane", "indirect", "formally", "popular", "additive", "residential",
    "exact", "downtown", "comparable", "powerless", "solid", "unevenly", "informality",
    "accidentally", "on"]

# Now, we need to define the format of the prompt that we are using.

eval_template = \
"""Instruction: [PROMPT]
Input: [INPUT]
Output: [OUTPUT]"""

# Now, let's use APE to find prompts that generate antonyms for each word.
from automatic_prompt_engineer import ape

sum_prompts = []


#prompts, 
prompts, res, demo_fn  = ape.simple_prompts(
    dataset=(words, antonyms),
    eval_template=eval_template,
    eval_model="text-ada-001",
    prompt_gen_model = "text-ada-001"
    )
#sum_prompts.extend(prompts)
#print(res)                                  
#res_prompts = res.__dict__['prompts']
#res_scores = res.__dict__['scores']
import json

with open('prompts.json', 'w') as f:
    json.dump(res.__dict__['prompts'], f)
    
with open('scores.json', 'w') as f:
    json.dump(res.__dict__['scores'], f)
    
    





#print(len(sum_prompts))
#print(result)

# Let's see the results.


