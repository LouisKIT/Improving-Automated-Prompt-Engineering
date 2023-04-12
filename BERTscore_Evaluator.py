from automatic_prompt_engineer import ape
from bert_score import score
import json
import os
import random
import time

#This is only an evaluator and paraphrasing file. Please launch the code from the main file "INNER_APE.py."
#Don't forget to set your desired number of loops below in the BERT function.


import openai
openai.api_key = "xxx"
openai.api_base = "INSERT_API"

def BERT(initial_prompts, input_, output_refs):
    n = 2    
    eval_template = \
    """Instruction: [PROMPT]
    Input: [INPUT]
    Output: [OUTPUT]"""
    dataset=(input_, output_refs)
    data_loop = []
    dir_path = os.path.join(os.path.dirname(__file__), "Unleashed_APE", "BERT_Files")

    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)                                   
    prompts = initial_prompts
    if n>1:                        #Number of BertScore Evaluation rounds with n-1 Paraphrasing rounds
        for j in range(n-1):      #e.g. n=5 --> iterates through loop 4 times, paraphrasing again after 4th then exiting loop and evaluating 5th time
            print('Evaluating bananas...')
            score_list=BertScoreAvg(prompts, input_,output_refs) #calculate avg BERTscore for each prompt
            score_list.sort(key=lambda a: a[1], reverse=True)
            first_values = [tup[0] for tup in score_list]
            prompts=Paraphrase(first_values)
            #This second loop evaluates prompts with likelihood each round as well for comparison
            if j in range(n-2):
                data_loop.clear()
                result = ape.simple_eval(dataset,prompts,eval_template=eval_template,eval_model="text-davinci-002")
                output= result.sorted()
            
                prompts_likeli = output[0]
                scores_likeli = output[1]
            
            
                for d_prompt, d_score in zip(prompts_likeli, scores_likeli):
                    data_loop.append({"Prompt": d_prompt, "Score": d_score})
                file_name = f"Prompts_BERT_LI_{j+1}.json"
                file_path = os.path.join(dir_path, file_name)
                with open(file_path, "w") as outfile:
                    json.dump(data_loop, outfile)
                
                
    print('Evaluating bananas one more time, almost done...')
    score_list=BertScoreAvg(prompts, input_,output_refs)
    score_list.sort(key=lambda a: a[1], reverse=True)
    final_prompts = [l[0] for l in score_list]
    score_dict = [{"Prompt": x, "Score": y} for x,y, in score_list]
    file_name_final = "Final_BERT_Scores.json"
    file_path = os.path.join(dir_path, file_name_final)
    with open(file_path, 'w') as outfile:     #If you want/don't want to output the final BERT-scores prompts before
        json.dump(score_dict, outfile)        #their likelihood eval, uncomment/comment these 2 lines
    
    return final_prompts
    




def BertScoreAvg(prompts, input_, output_refs):
    scores = []
    

    for i in range(len(prompts)):
        F1_avg=0
        input_rand= random.sample(list(enumerate(input_)),5)
        for j in range(len(input_rand)):
            prompt_dummy = 'Instruction: [PROMPT]\nInput: [INPUT]\nOutput:'.replace('[PROMPT]',prompts[i]).replace('[INPUT]', input_rand[j][1])
            completion = None
            while completion is None:
                try:
                    completion = openai.Completion.create(
                    model = "text-davinci-002",              
                    prompt=prompt_dummy,  
                    min_tokens=1,                # uncomment this line when using BLOOM, comment it for OPENAI API             
                    max_tokens=5,              #adjust according to the task type at hand, especially when using smaller models           
                    temperature=0.7,            
                    echo=False,                  
                    top_p=1.0,                 
                    n=1,                        
                    logprobs=0,                 
                    )
                except Exception as e:
                        print(e)
                        print('Retrying...')
                        time.sleep(5)           
            output=completion.choices[0].text
            
            cand=[]
            refs=[]
            cand.append(output)
            refs.append(output_refs[input_rand[j][0]])
            
            P, R, F1 = score(cand, refs, lang='en', verbose = True, rescale_with_baseline=True)
        
            cand.clear()
            refs.clear()
            F1_int = F1.item()
            F1_avg += F1_int
        F1_avg=F1_avg/5
        value = (prompts[i],F1_avg)
        scores.append(value)
    return scores


def Paraphrase(prompts):
    p = prompts[0:10]  #select top10 prompts
    print('Generating paraphrased bananas...')
    prompt_list=[]
    prompt_list.extend(p)
    #Generate new prompts by keeping the top 10 and paraphrasing each of them 4 times  
    for i in range(len(p)):             
            for j in range(4):
                prompt_dummy = 'Generate a variation of the following instruction while keeping the semantic meaning: \nInput: [INPUT]'.replace('[INPUT]', p[i])
                completion = None
                while completion is None:
                    try:
                        completion = openai.Completion.create(
                        model = "text-davinci-002",              
                        prompt=prompt_dummy,  
                        min_tokens=1,               #comment this parameter out when using OPENAI
                        max_tokens=200,              
                        temperature=0.7,            
                        echo=False,                  
                        top_p=1.0,                 
                        n=1,                        
                        logprobs=0,                 
                        )
                    except Exception as e:
                        print(e)
                        print('Retrying...')
                        time.sleep(5)
                output=completion.choices[0].text
                
                prompt_list.append(output) 
    return prompt_list






if __name__ == "__main__":
    BERT()









