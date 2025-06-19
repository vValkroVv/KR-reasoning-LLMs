import pickle
import warnings
warnings.filterwarnings("ignore")

import re
def parse_response(response):
    pattern = r"```sql\s*(.*?)\s*```"
    
    sql_blocks = re.findall(pattern, response, re.DOTALL)

    if sql_blocks:
        # Extract the last SQL query in the response text and remove extra whitespace characters
        last_sql = sql_blocks[-1].strip()
        return last_sql
    else:
        # print("No SQL blocks found.")
        return response
        
def process_answers(outputs, path=None):
    answers = []
    for i in range(len(outputs)):
        answers.append(outputs[i].outputs[0].text)
    if path is not None:
        with open(path, "wb") as fp:
            pickle.dump(answers, fp)
    return answers


def extract_think_from_query(sql_predict, path=None):
    think_arr = []
    num_errors = 0
    for i in range(len(sql_predict)):
        try:
            answer = sql_predict[i].split("</think>")[0]
            think_arr.append(answer[7::].strip())
        except:
            think_arr.append("")
            num_errors += 1

    if path is not None:
        with open(path, "wb") as fp:
            pickle.dump(think_arr, fp)
    print('extract_think количество ошибок: ',num_errors)
    return think_arr

def extract_sql_query(sql_predict, path=None):
    queries = []
    num_errors = 0
    for i in range(len(sql_predict)):
        try:
            answer = sql_predict[i].split("</think>")[-1].strip()
            while answer[0]=='`':
                answer = answer[1:]
            while answer[-1]=='`':
                answer = answer[:-1]
            # answer = answer.split("sql")[1]
            queries.append(answer.strip())
        except:
            queries.append("")
            num_errors += 1
            
    if path is not None:
        with open(path, "wb") as fp:
            pickle.dump(queries, fp)
            
    print('extract_sql количество ошибок: ',num_errors)
    return queries


# def extract_sql_query(sql_predict, path=None):
#     queries = []
#     num_errors = 0
#     for i in range(len(sql_predict)):
#         answer = parse_response(sql_predict[i])   
#         queries.append(answer.strip())
#     if path is not None:
#         with open(path, "wb") as fp:
#             pickle.dump(queries, fp)
            
#     print('extract_sql количество ошибок: ',num_errors)
#     return queries

def answers_process_pipeline(sql_predict,path_sql_save=None,path_think_save=None,path_sql_answers_save=None):
    sql_predict = process_answers(sql_predict,path_sql_save)
    think = extract_think_from_query(sql_predict,path_think_save)
    answer = extract_sql_query(sql_predict, path_sql_answers_save)
    return answer