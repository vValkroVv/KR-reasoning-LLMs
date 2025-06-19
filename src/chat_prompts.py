import warnings
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer
import json
import pickle
import numpy as np
from .prompt import generate_combined_prompts_one
from .prompt_reasoning import generate_combined_prompts_one as generate_combined_prompts_one_reasoning
from .prompt_with_cot_add import generate_combined_prompts_one as generate_combined_prompts_one_with_cot

with open("/data/home/vkropoti/sql_data/mini-dev-index", "rb") as fp:   # Unpickling
    mini_dev_index = np.array(pickle.load(fp))
    # mini_dev_index = set(mini_dev_index)
    
def get_chat_messages(inp,system_prompt):
    return [{"role":"system", "content":system_prompt},
    {"role":"user", "content": f"{inp}"}]

def load_json(dir):
    with open(dir, "r") as j:
        contents = json.loads(j.read())
    return contents

def generate_prompts_sql(path_dev_json,path_sql_dbs,model_name, system_prompt,use_reasoning=False, paths_to_cot=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if paths_to_cot is not None:
        cot = []
        for path_to_cot in paths_to_cot:
            with open(path_to_cot, "rb") as fp:   # Unpickling
                cot.append(pickle.load(fp))
            
    data = load_json(path_dev_json)
    batch_messages = []
    for i,k in enumerate(range(len(data))):
    # for i,k in enumerate(mini_dev_index):
        question = data[k]['question']
        db = data[k]['db_id']
        sql_dialect = "SQLite"
        knowledge = data[k]['evidence']
        
        if paths_to_cot is not None:
            cots = [x[i] for x in cot]
            prompt = generate_combined_prompts_one_with_cot(f'{path_sql_dbs}{db}/{db}.sqlite', question, sql_dialect, cots, knowledge, use_reasoning)
        else:
            if use_reasoning:
                prompt = generate_combined_prompts_one_reasoning(f'{path_sql_dbs}{db}/{db}.sqlite', question, sql_dialect, knowledge)
            else:
                prompt = generate_combined_prompts_one(f'{path_sql_dbs}{db}/{db}.sqlite', question, sql_dialect, knowledge)
            
        batch_messages.append(get_chat_messages(prompt,system_prompt))
        
    formatted_prompts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,enable_thinking=use_reasoning) #enable_thinking=False
        for messages in batch_messages
    ]
    return formatted_prompts

def generate_bd_list(path_dev_json):


    data = load_json(path_dev_json)
    bd_list = []
    for k in range(len(data)):
    # for k in mini_dev_index:
        db = data[k]['db_id']
        bd_list.append(db)

    return  bd_list

def generate_sql_gt_list(path_dev_json):

    data = load_json(path_dev_json)
    sql_gt_list = []
    for k in range(len(data)):
    # for k in mini_dev_index:
        gt_sql = data[k]['SQL']
        sql_gt_list.append(gt_sql)

    return  sql_gt_list

