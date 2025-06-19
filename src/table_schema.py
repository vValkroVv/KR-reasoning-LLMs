import sqlite3
from sqlalchemy import create_engine
from .schema_engine import SchemaEngine
from .db_prompts import db_prompts
def nice_look_table(column_names: list, values: list):
    rows = []
    # Determine the maximum width of each column
    widths = [
        max(len(str(value[i])) for value in values + [column_names])
        for i in range(len(column_names))
    ]

    # Print the column names
    header = "".join(
        f"{column.rjust(width)} " for column, width in zip(column_names, widths)
    )
    # print(header)
    # Print the values
    for value in values:
        row = "".join(f"{str(v).rjust(width)} " for v, width in zip(value, widths))
        rows.append(row)
    rows = "\n".join(rows)
    final_output = header + "\n" + rows
    return final_output


# def generate_schema_prompt_sqlite(db_path, num_rows=None):
#     # extract create ddls
#     """
#     :param root_place:
#     :param db_name:
#     :return:
#     """
#     full_schema_prompt_list = ["here is avaliable tables below\n"]
#     conn = sqlite3.connect(db_path)
#     # Create a cursor object
#     cursor = conn.cursor()
#     cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
#     tables = cursor.fetchall()
#     schemas = {}
#     for table in tables:
#         if table == "sqlite_sequence":
#             continue
#         cursor.execute(
#             "SELECT sql FROM sqlite_master WHERE type='table' AND name='{}';".format(
#                 table[0]
#             )
#         )
#         create_prompt = f"Table: {table[0]}\n{cursor.fetchone()[0]}"
#         schemas[table[0]] = create_prompt
#         if num_rows:
#             cur_table = table[0]
#             if cur_table in ["order", "by", "group"]:
#                 cur_table = "`{}`".format(cur_table)

#             cursor.execute("SELECT * FROM {} LIMIT {}".format(cur_table, num_rows))
#             column_names = [description[0] for description in cursor.description]
#             values = cursor.fetchall()
#             rows_prompt = nice_look_table(column_names=column_names, values=values)
#             verbose_prompt = "/* \n {} example rows: \n SELECT * FROM {} LIMIT {}; \n {} \n */".format(
#                 num_rows, cur_table, num_rows, rows_prompt
#             )
#             schemas[table[0]] = "{} \n {}".format(create_prompt, verbose_prompt)

#     for k, v in schemas.items():
#         full_schema_prompt_list.append(v)

#     schema_prompt = "\n\n".join(full_schema_prompt_list)

#     return schema_prompt

def generate_schema_prompt_sqlite(db_path, num_rows=None):
    # print('path',f"sqlite://{db_path}")
    # db_engine = create_engine(f"sqlite:///{db_path}")
    # schema_engine = SchemaEngine(engine=db_engine, db_name=db_path.split('/')[-1].split('.')[0],schema=None)
    # mschema = schema_engine.mschema
    # mschema_str = mschema.to_mschema()

    return "here is avaliable tables and their structure below\n" + db_prompts[db_path.split('/')[-1].split('.')[0]]
    # + db_prompts[db_path.split('/')[-1].split('.')[0]]

def generate_schema_prompt(sql_dialect, db_path=None, num_rows=None):
    if sql_dialect == "SQLite":
        return generate_schema_prompt_sqlite(db_path, num_rows)
    else:
        raise ValueError("Unsupported SQL dialect: {}".format(sql_dialect))