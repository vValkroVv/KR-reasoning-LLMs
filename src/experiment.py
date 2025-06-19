import warnings
warnings.filterwarnings("ignore")

from .pipeline import pipeline, get_batch_example
from .model_and_sampling import get_vllm_model
import gc
import torch
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment

# from tqdm import tqdm
import logging
# logging.getLogger('vllm').setLevel(logging.WARNING)
# logging.getLogger('vllm').setLevel(logging.ERROR)
logging.getLogger('vllm').setLevel(logging.CRITICAL)

from .config import create_think_configs
from .config import create_think_train_configs
from .config import create_cot_configs
from .config import create_cot_configs_deepcoder

def run_think_experiment(model, arr_seeds, arr_versions):

    print("Загружаем модель...")
    model_config, prompts_config, save_config = create_think_configs(model,arr_seeds[0],arr_versions[0])
    llm = get_vllm_model(model=model_config['model'],
                                      tensor_parallel_size=model_config['tensor_parallel_size'],
                                          max_model_len=model_config['max_model_len'],
                                          gpu_memory_utilization=model_config['gpu_memory_utilization'],
                                          trust_remote_code=model_config['trust_remote_code'],
                                     )
    print("Модель успешно загружена")
    print("model_config",model_config)
    print()
    print("prompts_config",prompts_config)
    print()
    print("save_config",save_config)
    print()
    print("Модель успешно загружена")
    batch_example, max_tokens = get_batch_example(model_config, prompts_config)
    print('Пример промпта')
    print(batch_example)
    print()
    print()
    print(f"Максимальное количество токенов в промптах: {max_tokens}")
    print()
    print('начинаем скорринг')
    
    for seed, version in zip(arr_seeds,arr_versions):
        model_config, prompts_config, save_config = create_think_configs(model,seed,version)
        pipeline(llm, model_config, prompts_config, save_config)
        
        print("="*100)
        print()
        print()
    destroy_distributed_environment()
    destroy_model_parallel()
    del llm.llm_engine
    del llm
    gc.collect()
    torch.cuda.empty_cache()

def run_think_train_experiment(model, arr_seeds, arr_versions):

    print("Загружаем модель...")
    model_config, prompts_config, save_config = create_think_train_configs(model,arr_seeds[0],arr_versions[0])
    llm = get_vllm_model(model=model_config['model'],
                                      tensor_parallel_size=model_config['tensor_parallel_size'],
                                          max_model_len=model_config['max_model_len'],
                                          gpu_memory_utilization=model_config['gpu_memory_utilization'],
                                          trust_remote_code=model_config['trust_remote_code'],
                                     )
    print("Модель успешно загружена")
    print("model_config",model_config)
    print()
    print("prompts_config",prompts_config)
    print()
    print("save_config",save_config)
    print()
    print("Модель успешно загружена")
    batch_example, max_tokens = get_batch_example(model_config, prompts_config)
    print('Пример промпта')
    print(batch_example)
    print()
    print()
    print(f"Максимальное количество токенов в промптах: {max_tokens}")
    print()
    print('начинаем скорринг')
    
    for seed, version in zip(arr_seeds,arr_versions):
        model_config, prompts_config, save_config = create_think_train_configs(model,seed,version)
        pipeline(llm, model_config, prompts_config, save_config)
        
        print("="*100)
        print()
        print()
    destroy_distributed_environment()
    destroy_model_parallel()
    del llm.llm_engine
    del llm
    gc.collect()
    torch.cuda.empty_cache()

def run_cot_experiment(model, arr_seeds, arr_versions, type_cot, use_think=False,num_observations=1):

    print("Загружаем модель...")
    model_config, prompts_config, save_config = create_cot_configs(model,arr_seeds[0],arr_versions[0],type_cot, len(arr_versions), use_think=use_think,num_observations=num_observations)
    llm = get_vllm_model(model=model_config['model'],
                                      tensor_parallel_size=model_config['tensor_parallel_size'],
                                          max_model_len=model_config['max_model_len'],
                                          gpu_memory_utilization=model_config['gpu_memory_utilization'],
                                          trust_remote_code=model_config['trust_remote_code'],
                                     )
    print("Модель успешно загружена")
    print("model_config",model_config)
    print()
    print("prompts_config",prompts_config)
    print()
    print("save_config",save_config)
    print()
    print("Модель успешно загружена")
    batch_example, max_tokens = get_batch_example(model_config, prompts_config)
    print('Пример промпта')
    print(batch_example)
    print()
    print()
    print(f"Максимальное количество токенов в промптах: {max_tokens}")
    print()
    print('начинаем скорринг')
    
    
    for seed, version in zip(arr_seeds,arr_versions):
        model_config, prompts_config, save_config = create_cot_configs(model,seed,version,type_cot, len(arr_versions),use_think=use_think,num_observations=num_observations)
        pipeline(llm,model_config, prompts_config, save_config)
        
        print("="*100)
        print()
        print()
    destroy_distributed_environment()
    destroy_model_parallel()
    del llm.llm_engine
    del llm
    gc.collect()
    torch.cuda.empty_cache()

def run_cot_experiment_deepcoder(model, model_add, arr_seeds, arr_versions, type_cot, use_think=False,num_observations=1):
    print("Загружаем модель...")
    model_config, prompts_config, save_config = create_cot_configs_deepcoder(model,model_add,arr_seeds[0],arr_versions[0],type_cot, len(arr_versions), use_think=use_think, num_observations=num_observations)
    llm = get_vllm_model(model=model_config['model'],
                                      tensor_parallel_size=model_config['tensor_parallel_size'],
                                          max_model_len=model_config['max_model_len'],
                                          gpu_memory_utilization=model_config['gpu_memory_utilization'],
                                          trust_remote_code=model_config['trust_remote_code'],
                                     )
    print("model_config",model_config)
    print("prompts_config",prompts_config)
    print("save_config",save_config)
    print("Модель успешно загружена")
    batch_example, max_tokens = get_batch_example(model_config, prompts_config)
    print('Пример промпта')
    print(batch_example)
    print()
    print()
    print(f"Максимальное количество токенов в промптах: {max_tokens}")
    print()
    print('начинаем скорринг')
    

    for seed, version in zip(arr_seeds,arr_versions):
        model_config, prompts_config, save_config = create_cot_configs_deepcoder(model,model_add,seed,version,type_cot, len(arr_versions), use_think=use_think, num_observations=num_observations)
        pipeline(llm, model_config, prompts_config, save_config)
        
        print("="*100)
        print()
        print()
        
    destroy_distributed_environment()
    destroy_model_parallel()
    del llm.llm_engine
    del llm
    gc.collect()
    torch.cuda.empty_cache()
        