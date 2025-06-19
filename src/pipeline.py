from .model_and_sampling import get_sampling_func
from .chat_prompts import generate_prompts_sql, generate_bd_list, generate_sql_gt_list
from .process_and_save_answers import answers_process_pipeline
from .score_sql import print_and_save
from .process_and_save_answers import answers_process_pipeline

from transformers import AutoTokenizer

import warnings
warnings.filterwarnings("ignore")

def get_system_prompt(use_reasoning,use_cot):
    if use_cot:
        if use_reasoning:
            system_prompt = """
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine: SQLite

You will be given an additional analysis of other models, please take it into account, so the plan is
1. Analyse given addition analysis from other models (find in which analysis the same and differ, how to use that to answer question) (additiona reasoning around 0.5-1k words)
2. Combine best from previous part and think how to solve the problem (analyze tables, take care what columns avaliable in each table and so on, check the syntacsis) (additiona reasoning around 0.5-1k words)
3. Create the answers. The answer should be only sql query.

Carefully follow the instruction and add <|im_end|> and the end of the final answer"""
        else:
#             , choose think, combine them to answer the most accurate way.
# Firstly think how to solve the problem, analyze tables, take care what columns avaliable in each table and so on, check the syntacsis.
# Secondly, create the answers. The answer should be only sql query.
        #reasoning
            system_prompt = """
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine: SQLite

You will be given an additional analysis of other models, please take it into account, so the plan is
1. Analyse given addition analysis from other models (find in which analysis the same and differ, how to use that to answer question)
2. Combine best from previous part and think how to solve the problem (analyze tables, take care what columns avaliable in each table and so on, check the syntacsis)
3. Create the answers. The answer should be only sql query.

Carefully follow the instruction and add <|im_end|> and the end of the final answer"""
    else:
        if use_reasoning:
            #reasoning
            system_prompt = """
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine: SQLite

Firstly think how to solve the problem, analyze tables, take care what columns avaliable in each table and so on, check the syntacsis (do not otherthink 2k words is a limit, no more!).
Secondly, create the answer based on thoughts. The answer should be only sql query.

Carefully follow the instruction and add <|im_end|> and the end of the final answer.
"""
        else:
            system_prompt = "You are helpfull assistant solving text to sql. Carefully answer the question and add <|im_end|> and the end of it."
    return system_prompt
    
def pipeline(llm, model_config, prompts_config, save_config):
    # print("Загружаем модель...")
    
    sampling_params = get_sampling_func(
        max_new_tokens = model_config['max_new_tokens'],
        temperature=model_config['temperature'],
                                          top_k=model_config['top_k'],
                                          top_p=model_config['top_p'],
                                          seed=model_config['seed'],
                                     )
    
    # print("Модель успешно загружена")
    
    print("Загружаем чаты с данными...")
    system_prompt = get_system_prompt(prompts_config['use_reasoning'],prompts_config['paths_to_cot'] is not None)
    batch_prompts = generate_prompts_sql(path_dev_json=prompts_config['path_dev_json'],
                                        path_sql_dbs=prompts_config['path_sql_dbs'],
                                         model_name=model_config['model'],
                                         system_prompt=system_prompt,
                                         use_reasoning = prompts_config['use_reasoning'],
                                         paths_to_cot = prompts_config['paths_to_cot'],
                                        )
    bd_list = generate_bd_list(path_dev_json=prompts_config['path_dev_json'])
    sql_gt_list = generate_sql_gt_list(path_dev_json=prompts_config['path_dev_json'])
    print("Данные успешно загружены")
    
    print("Начали генерацию SQL запросов")
    outputs = llm.generate(batch_prompts,sampling_params=sampling_params)
    sql_predict = answers_process_pipeline(outputs,
                                          path_sql_save=save_config['path_sql_save'],
                                           path_think_save=save_config['path_think_save'],
                                           path_sql_answers_save=save_config['path_sql_answers_save'],
                                          )
    print("Закончили генерацию SQL запросов")
    
    path_sql_dbs = prompts_config['path_sql_dbs']
    # "/data/home/vkropoti/sql_data/dev_databases/"
    print("Начали скорр полученных SQL запросов")
    print_and_save(model_config['model'].split("/")[-1],path_sql_dbs,bd_list,sql_gt_list,sql_predict,
                   path_to_save_scores=save_config['path_to_save_scores'],
                   path_to_save_executed = save_config['path_to_save_executed'],
                  )

def get_batch_example(model_config, prompts_config):
    system_prompt = get_system_prompt(prompts_config['use_reasoning'],prompts_config['paths_to_cot'] is not None)
    batch_prompts = generate_prompts_sql(path_dev_json=prompts_config['path_dev_json'],
                                        path_sql_dbs=prompts_config['path_sql_dbs'],
                                         model_name=model_config['model'],
                                         system_prompt=system_prompt,
                                         use_reasoning = prompts_config['use_reasoning'],
                                         paths_to_cot = prompts_config['paths_to_cot'],
                                        )

    #v2
    tokenizer = AutoTokenizer.from_pretrained(model_config['model'])
    arr_lens = []
    # print(batch_prompts[0])
    for i in range(len(batch_prompts)):
        res = tokenizer.tokenize(batch_prompts[i])
        arr_lens.append(len(res))
    # break
    
    return batch_prompts[0], max(arr_lens)



    
    