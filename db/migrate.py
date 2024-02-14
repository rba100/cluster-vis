import psycopg2
from psycopg2.extras import execute_batch

# Database connection parameters
source_db_config = {
    'dbname': 'postgres',
    'user': 'vectoruser',
    'password': '',
    'host': '',
    'port': ''
}

# requiressl=True
target_db_config = {
    'dbname': 'themestation',
    'user': 'citus',
    'password': '',
    'host': '',
    'port': '5432',
    'sslmode': 'require'    
}

# Batch size for reading and writing
BATCH_SIZE = 1000

def migrate_table(source_conn_params, target_conn_params, table_name):
    source_conn = psycopg2.connect(**source_conn_params)
    target_conn = psycopg2.connect(**target_conn_params)
    
    try:
        with source_conn.cursor(name='fetch_large_result') as source_cursor, \
             target_conn.cursor() as target_cursor:
             
            source_cursor.itersize = BATCH_SIZE
            
            # Adjust the SELECT query if you need specific columns
            source_cursor.execute(f'SELECT word, embedding, isCommon FROM {table_name}')
            
            while True:
                records = source_cursor.fetchmany(BATCH_SIZE)
                if not records:
                    break  # No more records to fetch
                
                # Adjust the INSERT query according to your table structure
                execute_batch(target_cursor,
                              f'INSERT INTO {table_name} (word, embedding, isCommon) VALUES (%s, %s, %s)',
                              records)
                
                target_conn.commit()  # Commit after each batch
                print(".", end="")
                
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        source_conn.close()
        target_conn.close()

# Call the function
migrate_table(source_db_config, target_db_config, 'words')
print("Migration complete")
