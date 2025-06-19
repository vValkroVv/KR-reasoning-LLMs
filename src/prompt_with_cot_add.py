from .table_schema import generate_schema_prompt

def generate_add_reasoning_prompt(reasoning_infos):
    if len(reasoning_infos)==0:
        return ""
        
    init_ = [f"Here is addition analysis from other models which can be helpfull, please take this into account but be careful (choose the best from each analysis, combine them to make the most proper answer):\n"]
    for i in range(len(reasoning_infos)):
        init_.append(f"model {i} analysis:\n {reasoning_infos[i]}\n")
    return "\n".join(init_)

def generate_comment_prompt(question, sql_dialect, knowledge=None):
    base_prompt = f"-- Using valid {sql_dialect}"
    knowledge_text = " and understanding External Knowledge" if knowledge else ""
    knowledge_prompt = f"-- External Knowledge: {knowledge}" if knowledge else ""

    combined_prompt = (
        f"{base_prompt}{knowledge_text}, answer the following questions for the tables provided above.\n"
        f"-- {question}\n"
        f"{knowledge_prompt}"
    )
    return combined_prompt


def generate_cot_prompt(sql_dialect):
    return f"\nGenerate the {sql_dialect} for the above question after thinking step by step: "


# def generate_instruction_prompt(sql_dialect):
#     return f"""
# \nIn your response, you do not need to mention your intermediate steps. 
# Do not include any comments in your response.
# Do not need to start with the symbol ```
# You only need to return the result {sql_dialect} SQL code
# start from SELECT
        # """

def generate_instruction_prompt(sql_dialect, question, think=False):
    if not think:
        return f"""
Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.
- Do not misuse column names, keep them in quotes `
- Carefully understand what colomns belond to what tables
- Use only initial column names, change names prohibited
        
Output Format:
In your answer, please enclose the generated SQL query in a code block:
```
-- Your SQL query
```    
and nothing else, no exaplains, no comments (asnwer should be less than 512 characters)

Take a deep breath and think step by step to find the correct SQL query in SQLite format.

Question repetition: {question}

Be care! Include only what i want it questiona and nothing else"""
    else:
        return f"""
Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please:
    1. Analyse given addition analysis from other models (find in which analysis the same and differ, how to use that to answer question) (additiona reasoning around 0.5-1k words)
    2. Combine best from previous part and think how to solve the problem (analyze tables, take care what columns avaliable in each table and so on, check the syntacsis) (additiona reasoning around 0.5-1k words)
- In your response, you should provide thoughts and answer (thoughts are separated with <think>).
- Do not misuse column names, keep them in quotes `
- Carefully understand what colomns belond to what tables
- Use only initial column names, change names prohibited
        
Output Format:
In your answer, please enclose the generated SQL query in a code block:
```
-- Your SQL query
```    
and nothing else, no exaplains, no comments (asnwer should be less than 512 characters)

Take a deep breath and think step by step to find the correct SQL query in SQLite format.

Question repetition: {question}

Be care! Include only what i want it questiona and nothing else"""
        
# import json
# def load_json(dir):
#     with open(dir, "r") as j:
#         contents = json.loads(j.read())
#     return contents
    
# new_dataset = load_json('/home/vkropoti/vllm/src/new_dataset.json')
# data = load_json("/home/vkropoti/vllm/dev.json")
# d = {}
# for i in range(len(data)):
#     d[data[i]['question']] = new_dataset[i]['input_seq'] 
#     # data = load_json(path_dev_json)
#     # batch_messages = []
#     # bd_gt_list = []
#     # for k in tqdm(range(len(data))): #len(data)
#     #     question = data[k]['question']
#     #     db = data[k]['db_id']
#     #     sql_dialect = "SQLite"
#     #     knowledge = data[k]['evidence']
#     #     gt_sql = data[k]['SQL']
#     #     # prompt = generate_combined_prompts_one(f'{path_sql_dbs}{db}/{db}.sqlite',question,sql_dialect,knowledge)
#     #     batch_messages.append(get_chat_messages(new_dataset[k]['input_seq']))
        
# def generate_schema_prompt(sql_dialect, db_path,question):
#     return d[question]
    
def generate_combined_prompts_one(db_path, question, sql_dialect, reasoning_info,knowledge=None, think=False):
    reasoning_prompt = generate_add_reasoning_prompt(reasoning_info)
    schema_prompt = generate_schema_prompt(sql_dialect, db_path)
    # schema_prompt = generate_schema_prompt(sql_dialect, db_path,question)
    comment_prompt = generate_comment_prompt(question, sql_dialect, knowledge)
    cot_prompt = generate_cot_prompt(sql_dialect)
    instruction_prompt = generate_instruction_prompt(sql_dialect, question, think)

    combined_prompts = "\n\n".join(
        [reasoning_prompt, schema_prompt, comment_prompt, cot_prompt, instruction_prompt]
    )
    return combined_prompts