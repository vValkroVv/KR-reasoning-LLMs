def create_think_configs(model, seed, version):
    model_config = {
        'model':   f"/data/home/vkropoti/models/{model}",
         # f"/data/home/models/{model}",
        # f"/data/home/vkropoti/models/{model}",
        'tensor_parallel_size': 2,
        'max_model_len' : 25000,
        'max_new_tokens' : 15000,
        'temperature' : 0.0,
        'top_k' : 20,
        'top_p' : 0.95,
        'gpu_memory_utilization' : 0.9,
        'trust_remote_code' : False,
        'seed' : seed
    }


    prompts_config = {
        'path_dev_json' :  '/home/vkropoti/vllm/dev.json', # bird
        # "/data/home/vkropoti/sql_data/dev_spider.json",
        # '/home/vkropoti/vllm/dev.json', # bird
        'path_sql_dbs' :  "/data/home/vkropoti/sql_data/dev_databases/",
        # "/data/home/vkropoti/sql_data/data_bases_spider/" ,
        # "/data/home/vkropoti/sql_data/dev_databases/",
        
        'use_reasoning' : True,
        'paths_to_cot' : None
    }
    
    save_config = {
        'path_sql_save' : f"/data/home/vkropoti/sql_llm_answers/bird_dev_m_schema/sql_all_{model}-reasoning-v{version}",
        'path_think_save' : f"/data/home/vkropoti/sql_llm_answers/bird_dev_m_schema/sql_think_{model}-reasoning-v{version}",
        'path_sql_answers_save' : f"/data/home/vkropoti/sql_llm_answers/bird_dev_m_schema/sql_predict_{model}-reasoning-v{version}",
        'path_to_save_scores' :  f"/data/home/vkropoti/sql_llm_answers/bird_dev_m_schema/sql_results_{model}-reasoning-v{version}",
        'path_to_save_executed' :  f"/data/home/vkropoti/sql_llm_answers/bird_dev_m_schema/sql_executed_{model}-reasoning-v{version}",
        
        # 'path_sql_save' : f"/data/home/vkropoti/sql_llm_answers/spider_dev_m_schema/sql_all_{model}-reasoning-v{version}",
        # 'path_think_save' : f"/data/home/vkropoti/sql_llm_answers/spider_dev_m_schema/sql_think_{model}-reasoning-v{version}",
        # 'path_sql_answers_save' : f"/data/home/vkropoti/sql_llm_answers/spider_dev_m_schema/sql_predict_{model}-reasoning-v{version}",
        # 'path_to_save_scores' :  f"/data/home/vkropoti/sql_llm_answers/spider_dev_m_schema/sql_results_{model}-reasoning-v{version}",
        # 'path_to_save_executed' :  f"/data/home/vkropoti/sql_llm_answers/spider_dev_m_schema/sql_executed_{model}-reasoning-v{version}",
    }
    return model_config, prompts_config, save_config

def create_think_train_configs(model, seed, version):
    model_config = {
        'model':   f"/data/home/vkropoti/models/{model}",
        'tensor_parallel_size': 2,
        'max_model_len' : 25000,
        'max_new_tokens' : 15000,
        'temperature' : 0.95,
        'top_k' : 50,
        'top_p' : 0.95,
        'gpu_memory_utilization' : 0.9,
        'trust_remote_code' : False,
        'seed' : seed
    }


    prompts_config = {
        'path_dev_json' :  '/data/home/vkropoti/sql_data/train/train.json', # bird
        'path_sql_dbs' :  "/data/home/vkropoti/sql_data/train/train_databases/",        
        'use_reasoning' : True,
        'paths_to_cot' : None
    }

    
    save_config = {
        'path_sql_save' : f"/data/home/vkropoti/sql_llm_answers/bird_train_m_schema/sql_all_{model}-reasoning-v{version}",
        'path_think_save' : f"/data/home/vkropoti/sql_llm_answers/bird_train_m_schema/sql_think_{model}-reasoning-v{version}",
        'path_sql_answers_save' : f"/data/home/vkropoti/sql_llm_answers/bird_train_m_schema/sql_predict_{model}-reasoning-v{version}",
        'path_to_save_scores' :  f"/data/home/vkropoti/sql_llm_answers/bird_train_m_schema/sql_results_{model}-reasoning-v{version}",
        'path_to_save_executed' :  f"/data/home/vkropoti/sql_llm_answers/bird_train_m_schema/sql_executed_{model}-reasoning-v{version}",
    }
    return model_config, prompts_config, save_config
    
def create_cot_configs(model,seed, version, type_cot, n, use_think=False, num_observations=1):
    if use_think:
        model_config = {
            'model': f"/data/home/vkropoti/models/{model}",
            'tensor_parallel_size': 2,
            'max_model_len' : 40000,
            'max_new_tokens' : 10000,
            'temperature' : 0.6,
            'top_k' : 20,
            'top_p' : 0.95,
            'gpu_memory_utilization' : 0.9,
            'trust_remote_code' : False,
            'seed' : seed
        }
    else:
        model_config = {
            'model': f"/data/home/vkropoti/models/{model}",
            'tensor_parallel_size': 2,
            'max_model_len' : 40000,
            'max_new_tokens' : 1500,
            'temperature' : 0,
            'top_k' : 1,
            'top_p' : 0.8,
            'gpu_memory_utilization' : 0.9,
            'trust_remote_code' : False,
            'seed' : seed
        }

    
    paths_to_cot = []
    for i in range(num_observations):
        paths_to_cot.append(f"/data/home/vkropoti/sql_llm_answers/spider_dev_m_schema/sql_{type_cot}_{model}-reasoning-v{(version+i)%n}",)
        
        # f"/data/home/vkropoti/sql_llm_answers/base_reasoning/sql_{type_cot}_{model}-reasoning-v{(version+1)%n}",
        # f"/data/home/vkropoti/sql_llm_answers/base_reasoning/sql_{type_cot}_{model}-reasoning-v{(version+2)%n}",
        # f"/data/home/vkropoti/sql_llm_answers/base_reasoning/sql_{type_cot}_{model}-reasoning-v{(version+3)%n}",
    
    prompts_config = {
        'path_dev_json' :  "/data/home/vkropoti/sql_data/dev_spider.json",
        # "/data/home/vkropoti/sql_data/dev_spider.json",
        # '/home/vkropoti/vllm/dev.json',
        'path_sql_dbs' :  "/data/home/vkropoti/sql_data/data_bases_spider/" ,
        # "/data/home/vkropoti/sql_data/data_bases_spider/" ,
        # "/data/home/vkropoti/sql_data/dev_databases/",
        
        'use_reasoning' : use_think,
        'paths_to_cot' : paths_to_cot
    }

    
    save_config = {
        
        # 'path_sql_save' :  f"/data/home/vkropoti/sql_llm_answers/bird_dev_m_schema/sql_all_{model}-cot_{type_cot}_think=={use_think}_nobs{num_observations}-v{version}",
        # 'path_think_save' : None,
        # # f"/data/home/vkropoti/sql_llm_answers/base_reasoning/sql_think_{model}-cot_{type_cot}_think=={use_think}-v{version}",
        # 'path_sql_answers_save' : f"/data/home/vkropoti/sql_llm_answers/bird_dev_m_schema/sql_predict_{model}-cot_{type_cot}_think=={use_think}_nobs{num_observations}-v{version}",
        # 'path_to_save_scores' : f"/data/home/vkropoti/sql_llm_answers/bird_dev_m_schema/sql_results_{model}-cot_{type_cot}_think=={use_think}_nobs{num_observations}-v{version}",
        # 'path_to_save_executed' : f"/data/home/vkropoti/sql_llm_answers/bird_dev_m_schema/sql_executed_{model}-cot_{type_cot}_think=={use_think}_nobs{num_observations}-v{version}",

        'path_sql_save' :  f"/data/home/vkropoti/sql_llm_answers/spider_dev_m_schema/sql_all_{model}-cot_{type_cot}_think=={use_think}_nobs{num_observations}-v{version}",
        'path_think_save' : None,
        # f"/data/home/vkropoti/sql_llm_answers/base_reasoning/sql_think_{model}-cot_{type_cot}_think=={use_think}-v{version}",
        'path_sql_answers_save' : f"/data/home/vkropoti/sql_llm_answers/spider_dev_m_schema/sql_predict_{model}-cot_{type_cot}_think=={use_think}_nobs{num_observations}-v{version}",
        'path_to_save_scores' : f"/data/home/vkropoti/sql_llm_answers/spider_dev_m_schema/sql_results_{model}-cot_{type_cot}_think=={use_think}_nobs{num_observations}-v{version}",
        'path_to_save_executed' : f"/data/home/vkropoti/sql_llm_answers/spider_dev_m_schema/sql_executed_{model}-cot_{type_cot}_think=={use_think}_nobs{num_observations}-v{version}",
    }
    return model_config, prompts_config, save_config


def create_cot_configs_deepcoder(model_base, model_add, seed, version, type_cot, n, use_think=False, num_observations=1):
    if use_think:
        model_config = {
        'model': f"/data/home/vkropoti/models/{model_base}",
        'tensor_parallel_size': 2,
        'max_model_len' : 40000,
        'max_new_tokens' : 10000,
        'temperature' : 0.4,
        'top_k' : 20,
        'top_p' : 0.95,
        'gpu_memory_utilization' : 0.9,
        'trust_remote_code' : False,
        'seed' : seed
    }
    else:
        model_config = {
        'model': f"/data/home/vkropoti/models/{model_base}",
        'tensor_parallel_size': 2,
        'max_model_len' : 40000,
        'max_new_tokens' : 1500,
        'temperature' : 0,
        'top_k' : 1,
        'top_p' : 0.95,
        'gpu_memory_utilization' : 0.9,
        'trust_remote_code' : False,
        'seed' : seed
    }

    
    paths_to_cot = []
    for i in range(num_observations):
        # if i%2==0:
        paths_to_cot.append(f"/data/home/vkropoti/sql_llm_answers/bird_dev_m_schema/sql_{type_cot}_{model_add}-reasoning-v{(version+i)%n}")
        # else:
        #     paths_to_cot.append(f"/data/home/vkropoti/sql_llm_answers/bird_dev_m_schema/sql_{type_cot}_{model_base}-reasoning-v{(version+i)%n}")
    
    prompts_config = {
        'path_dev_json' : '/home/vkropoti/vllm/dev.json',
        'path_sql_dbs' : "/data/home/vkropoti/sql_data/dev_databases/",
        
        'use_reasoning' : use_think,
        'paths_to_cot' : paths_to_cot
    }
    
    save_config = {
            'path_sql_save' : f"/data/home/vkropoti/sql_llm_answers/bird_dev_m_schema/sql_all_{model_base}-cot_withadd_OMNI_{type_cot}_think=={use_think}_nobs{num_observations}-v{version}",
        'path_think_save' : None,
    # f"/data/home/vkropoti/sql_llm_answers/base_reasoning/sql_think_{model_base}-cot_withadd_{type_cot}_think=={use_think}-v{version}",
        'path_sql_answers_save' :  f"/data/home/vkropoti/sql_llm_answers/bird_dev_m_schema/sql_predict_{model_base}-cot_withadd_OMNI_{type_cot}_think=={use_think}_nobs{num_observations}-v{version}",
        'path_to_save_scores' :  f"/data/home/vkropoti/sql_llm_answers/bird_dev_m_schema/sql_results_{model_base}-cot_withadd_OMNI_{type_cot}_think=={use_think}_nobs{num_observations}-v{version}",
        'path_to_save_executed' : f"/data/home/vkropoti/sql_llm_answers/bird_dev_m_schema/sql_executed_{model_base}-cot_withadd_OMNI_{type_cot}_think=={use_think}_nobs{num_observations}-v{version}",
    }
    return model_config, prompts_config, save_config