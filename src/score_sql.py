import sqlite3
import numpy as np
import time
import pickle
from func_timeout import func_timeout, FunctionTimedOut
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# with open("/data/home/vkropoti/sql_data/mini-dev-index", "rb") as fp:   # Unpickling
#     mini_dev_index = np.array(pickle.load(fp))
    
def calculate_ex(predicted_res, ground_truth_res):
    res = 0
    if set(predicted_res) == set(ground_truth_res):
        res = 1
    return res

# def calculate_ex_values(path_sql_dbs, bd_list, sql_gt_list, sql_predict):
#     result = []
#     executed = []
#     for i in range(len(bd_list)):
#         if len(sql_predict[i])<=512:
#             try:
#                 def sqlite_operation():
#                     db = bd_list[i]
#                     sql_gt = sql_gt_list[i]
#                     conn = sqlite3.connect(f'{path_sql_dbs}{db}/{db}.sqlite')
#                     cursor = conn.cursor()
            
#                     with conn:
#                         cursor.execute(sql_predict[i])
#                         results_pred = cursor.fetchall()
#                         cursor.execute(sql_gt)
#                         results_gt = cursor.fetchall()
                        
#                     cursor.close()
#                     conn.close()
                    
#                     return results_pred, results_gt
#                 try:
#                     results_pred, results_gt = func_timeout(5, sqlite_operation)
#                     executed.append(1)
#                 except:
#                     s = 'time out'
#                     results_pred, results_gt = 0, 1
#                     executed.append(0)
                    
#                 result.append(calculate_ex(results_pred,results_gt))

#             except Exception as e:
#                 # print(e)
#                 result.append(0)
#                 executed.append(0)
#         else:
#             result.append(0)
#             executed.append(0)
#         # time.sleep(0.01)
#     return result, executed

def sql_worker(args):
    i, path_sql_dbs, db, sql_gt, sql_predict = args
    if len(sql_predict) > 512:
        return i, 0, 0
    
    try:
        
        def execute_query(path_sql_dbs,db,sql_predict,sql_gt):
            conn = sqlite3.connect(f'{path_sql_dbs}/{db}/{db}.sqlite')
            cursor = conn.cursor()
            with conn:
                cursor.execute(sql_predict)
                pred = cursor.fetchall()
                
                cursor.execute(sql_gt)
                real = cursor.fetchall()
            cursor.close()
            conn.close()
            return pred, real

        
        results_pred, results_gt = func_timeout(5, execute_query, args=(path_sql_dbs, db, sql_gt, sql_predict,))
        # results_gt = func_timeout(5, execute_query, args=(sql_gt,))
        
        
        return i, calculate_ex(results_pred, results_gt), 1
    
    except (FunctionTimedOut, Exception) as e:
        # print(e)
        return i, 0, 0

def calculate_ex_values(path_sql_dbs, bd_list, sql_gt_list, sql_predict):
    result = [0] * len(bd_list)
    executed = [0] * len(bd_list)
    
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                sql_worker,
                (i, path_sql_dbs, db, sql_gt_list[i], sql_predict[i])
            )
            for i, db in enumerate(bd_list)
            if len(sql_predict[i]) <= 512
        ]
        
        for future in as_completed(futures):
            try:
                i, ex_val, exec_flag = future.result(timeout=10)
                result[i] = ex_val
                executed[i] = exec_flag
            except Exception as e:
                continue
    
    return result, executed
    
def print_and_save(name, path_sql_dbs, bd_list, sql_gt_list, sql_predict, path_to_save_scores=None, path_to_save_executed=None):
    result, executed = calculate_ex_values(path_sql_dbs, bd_list, sql_gt_list, sql_predict)

    # print(f"{name} Финальный результат EX: {np.mean(result):.3f}")
    # print(f"{name} gроцент запросов, которые успешно выполнились: {np.mean(executed)*100:.2f}%")
    # print(f"{name} Mini DEV Финальный результат EX: {np.mean(np.array(result)[mini_dev_index]):.3f}")
    # print(f"{name} процент запросов, которые успешно выполнились Mini DEV: {np.mean(np.array(executed)[mini_dev_index])*100:.2f}%")

    print(f"{name} DEV Финальный результат EX: {np.mean(result)*100:.2f}")
    print(f"{name} процент запросов, которые успешно выполнились DEV: {np.mean(executed)*100:.2f}%")

    if path_to_save_scores is not None:
        with open(path_to_save_scores, "wb") as fp:   #Pickling
            pickle.dump(result, fp)
            
    if path_to_save_executed is not None:
        with open(path_to_save_executed, "wb") as fp:   #Pickling
            pickle.dump(executed, fp)

