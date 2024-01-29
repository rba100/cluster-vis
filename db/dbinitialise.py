import psycopg2
import argparse
import sys
import os
from schema import create_extension, create_table_words, create_table_embeddings, create_index_embeddings, create_table_cache, create_user

# Define your functions here (init_all, create_extension, etc.)

def main():
    # Define the command-line arguments
    parser = argparse.ArgumentParser(description="Database management script")
    parser.add_argument('command', choices=['create_table_embeddings', 'create_index_embeddings', 'create_table_cache', 'create_user'], help="Command to execute")
    parser.add_argument('--username', help="Username for creating a user")
    parser.add_argument('--password', help="Password for the user")
    parser.add_argument('--table', help="Table name for embedding operations")
    parser.add_argument('--size', type=int, help="Embedding size")

    # Parse the arguments
    args = parser.parse_args()
    
    connectionString = os.environ['POSTGRESDB'] if 'POSTGRESDB' in os.environ else "host=localhost dbname=postgres user=postgres password=postgres"
    conn = psycopg2.connect(connectionString)

    if args.username and args.password and args.command == 'create_user':
        create_user(conn, args.username, args.password)
    if args.table and args.size and args.command == 'create_table_embeddings':
        create_table_embeddings(conn, args.table, args.size)
    if args.table and args.command == 'create_index_embeddings':
        create_index_embeddings(conn, args.table)
    if args.table and args.size and args.command == 'create_table_cache':
        create_table_cache(conn, args.table, args.size)

    conn.close()

if __name__ == "__main__":
    main()
