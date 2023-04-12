from automatic_prompt_engineer import ape
import json
import os
import openai
from BERTscore_Evaluator import BERT, Paraphrase
openai.api_key = 'xxx'
openai.api_base = "INSERT_API"


"""This is the main file of the project. Insert your input-output pairs below and change the number of loop iterations
you would like below and in the BERTScore_Evaluator file. If you are only interested in the final prompts, you
can find them in the file beginning with 'Final_BERT_LI' for the BERT pipeline and "Prompt_Scores_LI_{n}" for the likelihood one
and your set n. The files for each of the pipelines are stored in a respective folder inside the "Unleashed_APE" folder."""

"""The Likelihood_Likelihood function uses only the original scoring from the APE repo. The BERT_Likelihood function 
uses the BERTScore as an intermediate scoring metric for more context/embedding-based evaluation of outputs."""


#Input/Outputs: Define your input/output pairs here. Remember adjusting max_tokens in BERTscore_Evaluator file 
#on smaller models for better results
input_ = ['sane', 'direct', 'informally', 'unpopular', 'subtractive', 'nonresidential', 'linear', 'inexact', 'uptown', 'incomparable', 'powerful', 'gaseous', 'evenly', 'formality', 'postnatal', 'deliberately', 'off', 'uninsured', 'credit', 'float', 'untalented', 'ground', 'immobility', 'coherence', 'nationalism', 'developed', 'inoffensive', 'unattractive', 'intense', 'disarrange', 'inaction', 'rewarding', 'local', 'net', 'interrogative', 'kern', 'fill', 'confused', 'inexperience', 'inconsideration', 'unambiguous', 'immortal', 'available', 'restful', 'passing', 'justice', 'mortal', 'baptized', 'holy', 'illegitimacy', 'center', 'unworldly', 'insincere', 'disbelieve', 'middle', 'breathing', 'shared', 'minimally', 'inefficient', 'unaccompanied', 'inside', 'depreciation', 'manual', 'disembark', 'crooked', 'distant', 'association', 'sell', 'headed', 'propriety', 'lower', 'seperate', 'passionate', 'inelegant', 'last', 'hopeful', 'unfruitful', 'disorganized', 'atomistic', 'determined', 'foe', 'concealing', 'opaque', 'lowered', 'fine', 'sadness', 'unarmed', 'temperate', 'marked', 'pro', 'acute', 'exhausted', 'tender', 'immobile', 'heterogeneity', 'feed', 'unnatural', 'preventive']
output_refs = ['insane', 'indirect', 'formally', 'popular', 'additive', 'residential', 'cubic', 'exact', 'downtown', 'comparable', 'powerless', 'solid', 'unevenly', 'informality', 'prenatal', 'accidentally', 'on', 'insured', 'cash', 'sink', 'talented', 'figure', 'mobility', 'incoherence', 'internationalism', 'undeveloped', 'offensive', 'attractive', 'mild', 'arrange', 'action', 'unrewarding', 'general', 'gross', 'declarative', 'kern', 'empty', 'clearheaded', 'experience', 'consideration', 'ambiguous', 'mortal', 'unavailable', 'restless', 'running', 'injustice', 'immortal', 'unbaptized', 'unholy', 'legitimacy', 'right', 'worldly', 'sincere', 'believe', 'end', 'breathless', 'unshared', 'maximally', 'efficient', 'accompanied', 'outside', 'appreciation', 'automatic', 'embark', 'straight', 'close', 'disassociation', 'buy', 'headless', 'impropriety', 'raise', 'combined', 'passionless', 'elegant', 'first', 'hopeless', 'fruitful', 'organized', 'holistic', 'undetermined', 'friend', 'revealing', 'clear', 'raised', 'coarse', 'happiness', 'armed', 'intemperate', 'unmarked', 'contra', 'negligible', 'fit', 'tough', 'mobile', 'homogeneity', 'starve', 'natural', 'curative']

def main():
    print('Your request for the best bananas has been received. Please stand by...')
    prompts = Initial_Prompts(input_, output_refs)
    Likelihood_Likelihood(prompts, input_, output_refs)
    BERT_Likelihood(prompts, input_, output_refs)
    return


# Use APE for initial prompt generation and first likelihood evaluation
def Initial_Prompts(input_, output_refs):
    print('Generating initial bananas...')
    eval_template = \
    """Instruction: [PROMPT]
    Input: [INPUT]
    Output: [OUTPUT]"""
    p, res, demo_fn  = ape.simple_prompts(
        dataset=(input_, output_refs),
        eval_template=eval_template,
        eval_model="text-davinci-002",
        prompt_gen_model = "text-davinci-002"
        )
    
    prompts = res.prompts
    scores = res.scores
    data = [{'Prompt': prompt, 'Score': score} for prompt, score in zip(prompts, scores)]
    data = sorted(data, key=lambda x: x['Score'], reverse=True)


    sorted_final= [d['Prompt'] for d in data]
    script_dir = os.path.dirname(__file__)

    
    dir_path = os.path.join(script_dir, "Unleashed_APE")
    file_path = os.path.join(dir_path, "Initial.json")

    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    
    with open(file_path, "w") as outfile:
        json.dump(data, outfile)
    return sorted_final



def Likelihood_Likelihood(initial_prompts, input_, output_refs):
    
    eval_template = \
    """Instruction: [PROMPT]
    Input: [INPUT]
    Output: [OUTPUT]"""
    dataset=(input_, output_refs)
    dir_path = os.path.join(os.path.dirname(__file__), "Unleashed_APE", "Likelihood_Files")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    n = 1 #Number of n evaluation rounds with n Paraphrasing rounds
    prompts = initial_prompts
    data_loop = []
    if n>=1:
        for i in range(n):      
            
            prompts_para=(Paraphrase(prompts))
            print('Evaluating bananas...')
            res2 = ape.simple_eval(dataset,prompts_para,eval_template=eval_template,eval_model="text-davinci-002") 
            output= res2.sorted()
            
            prompts = output[0]
            scores = output[1]
            
            
            for d_prompt, d_score in zip(prompts, scores):
                data_loop.append({"Prompt": d_prompt, "Score": d_score})
            file_name = f"Prompt_Scores_LIKELI_{i+1}.json"
            file_path = os.path.join(dir_path, file_name)
                 
            with open(file_path, 'w') as outfile:
                json.dump(data_loop, outfile)
            
            data_loop.clear()       
    return

def BERT_Likelihood(prompts, input_, output_refs):
    prompt_list = BERT(prompts, input_, output_refs)  #set number of eval rounds in BERTscore file
    print('BERTscore evaluation finished. Applying final APE likelihood scoring...')
    dir_path = os.path.join(os.path.dirname(__file__), "Unleashed_APE", "BERT_Files")

    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    eval_template = \
    """Instruction: [PROMPT]
    Input: [INPUT]
    Output: [OUTPUT]"""
    dataset=(input_, output_refs)  
    #Take final BERT-evaluated prompts and eval them using Likelihood 
    
    res = ape.simple_eval(dataset,prompt_list,eval_template=eval_template,eval_model="text-davinci-002")    
    
    output= res.sorted()
    _prompts = output[0]
    _scores = output[1]

    data = []
    for d_prompt, d_score in zip(_prompts, _scores):
        data.append({"Prompt": d_prompt, "Score": d_score})
    file_name = "Final_BERT_LI.json"
    file_path = os.path.join(dir_path, file_name)
    with open(file_path, "w") as outfile:
        json.dump(data, outfile)
    return





if __name__ == "__main__":
    main()
