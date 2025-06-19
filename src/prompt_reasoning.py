from .table_schema import generate_schema_prompt


def generate_comment_prompt(question, sql_dialect, knowledge=None):
    base_prompt = f"-- Using valid {sql_dialect}"
    knowledge_text = " and understanding External Knowledge" if knowledge else ""
    knowledge_prompt = f"-- External Knowledge: {knowledge}" if knowledge else ""

    combined_prompt = (
        f"{base_prompt}{knowledge_text}, answer the following questions for the tables provided above.\n"
        f"-- question: {question}\n"
        f"{knowledge_prompt}"
    )
    return combined_prompt


def generate_cot_prompt(sql_dialect):
    return f"\nGenerate the {sql_dialect} for the above question after thinking step by step: "


# def generate_instruction_prompt(sql_dialect):
#     return f"""
#         \nIn your response, you do not need to mention your intermediate steps. 
#         Do not include any comments in your response.
#         Do not need to start with the symbol ```
#         You only need to return the result {sql_dialect} SQL code
#         start from SELECT
#         """
def generate_instruction_prompt(sql_dialect, question):
    return f"""
Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.
- In your response, you should provide thoughts and answer (thoughts are separated with <think>).
- Do not misuse column names, keep them in quotes `
- Carefully understand what colomns belond to what tables
- Use only initial column names, change names prohibited
        
Output Format:
In your answer, please enclose the generated SQL query in a code block:
```
```    
and nothing else, no exaplains, no comments (asnwer should be less than 512 characters)

Take a deep breath and think step by step to find the correct SQL query in SQLite format.

Question repetition: {question}

Be care! Include only what i want it questiona and nothing else"""

# input_prompt_template = '''Task Overview:
# You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

# Database Engine:
# SQLite

# Database Schema:
# {db_details}
# This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

# Question:
# {question}

# Instructions:
# - Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
# - The generated query should return all of the information asked in the question without any missing or extra information.
# - Before generating the final SQL query, please think through the steps of how to write the query.

# Output Format:
# In your answer, please enclose the generated SQL query in a code block:
# ```
# -- Your SQL query
# ```

# Take a deep breath and think step by step to find the correct SQL query.'''

def generate_combined_prompts_one(db_path, question, sql_dialect, knowledge=None):
    schema_prompt = generate_schema_prompt(sql_dialect, db_path)
    comment_prompt = generate_comment_prompt(question, sql_dialect, knowledge)
    cot_prompt = generate_cot_prompt(sql_dialect)
    instruction_prompt = generate_instruction_prompt(sql_dialect,question)

    combined_prompts = "\n\n".join(
        [schema_prompt, comment_prompt, cot_prompt, instruction_prompt]
    )
    return combined_prompts