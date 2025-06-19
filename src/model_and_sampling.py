import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"
# "2,3"
# "1"
# "0,1"
os.environ["TOKENIZERS_PARALLELISM"] = 'False'

import warnings
warnings.filterwarnings("ignore")

from vllm import LLM
from transformers import AutoTokenizer
from vllm import SamplingParams

# rope_config = {
#     "rope_type": "yarn",
#     "factor": 1.38,
#     "original_max_position_embeddings": 32768
# }

rope_config = {
    "rope_type": "yarn",
    "factor": 4,
    "original_max_position_embeddings": 32768
}

def get_vllm_model(model, 
                   max_model_len=5000, 
                   gpu_memory_utilization = 0.9,
                   tensor_parallel_size=1,
                   trust_remote_code=False,
                  ):

    if trust_remote_code:
        llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,  # Needed for some models
            dtype='float16',
        )
    else:
        llm = LLM(
                model=model,
                tensor_parallel_size=2,
                # max_model_len=max_model_len,
                max_model_len=40960,
                # max_model_len=32768,
                # max_model_len=8192,
                gpu_memory_utilization=0.9,
                # trust_remote_code=True,  # Needed for some models
                # quantization="fp8"
                # dtype='fp8',
                # rope_scaling=rope_config,
                dtype='float16',
            )

        # print("stop_token_ids:", stop_token_ids)
        
        # max_model_len = 8192 # used to allocate KV cache memory in advance
        # max_input_len = 6144
        # max_output_len = 2048 # (max_input_len + max_output_len) must <= max_model_len
        
        # print("max_model_len:", max_model_len)
        # print("temperature:", 0)
        
        # sampling_params = SamplingParams(
        #     temperature = 0.0, 
        #     max_tokens = max_output_len,
        #     # top_k=20,
        #     # top_p=0.95,
        #     n = 1,
        #     stop_token_ids = stop_token_ids
        # )
        
        # llm = LLM(
        #     model = model,
        #     dtype = "bfloat16", 
        #     tensor_parallel_size = 2,
        #     max_model_len = max_model_len,
        #     gpu_memory_utilization = 0.92,
        #     swap_space = 42,
        #     enforce_eager = True,
        #     disable_custom_all_reduce = True,
        #     trust_remote_code = True
        # )

    return llm

def get_sampling_func(max_new_tokens=1000,
                  temperature=0,
                  top_k=20,
                   top_p=0.95,
                   seed=42
                  ):
        
    sampling_params = SamplingParams(max_tokens=max_new_tokens,
                                     seed=seed,
                                     stop = ["<|im_end|>"],
                                     temperature=temperature,
                                    top_k = top_k,
                                    top_p = top_p,
                                    )
    # sampling_params = SamplingParams(
    #         temperature = 0.8, 
    #         max_tokens = 2048,
    #         top_k=20,
    #         top_p=0.95,
    #         n = 1,
    #         stop_token_ids = [151645]
    #     )
    return sampling_params