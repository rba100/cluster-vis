import streamlit as st

embeddingsTablename = "openai3large256"
similarityOperator = "<#>"

@st.cache_data(max_entries=50)
def get_closest_words(embedding, _cursor, preferCommonWords=True, k=5):
    embedding_str = ','.join(map(str, embedding))
    embedding_str = f'[{embedding_str}]'

    # Queries now join the words table with the openai3large256 table
    query = f"""
    WITH CommonWords AS (
        SELECT w.word
        FROM words w
        INNER JOIN {embeddingsTablename} o ON w.id = o.id
        WHERE w.isCommon = true
        ORDER BY o.embedding {similarityOperator} %s
        LIMIT {k}
    ),
    UncommonWords AS (
        SELECT w.word
        FROM words w
        INNER JOIN {embeddingsTablename} o ON w.id = o.id
        WHERE w.isCommon = false
        ORDER BY o.embedding {similarityOperator} %s
        LIMIT {k}
    )

    SELECT word
    FROM (
        SELECT word FROM CommonWords
        UNION ALL
        SELECT word FROM UncommonWords
    ) AS Combined
    LIMIT {k};
    """

    queryAll = f"""
    SELECT w.word FROM words w
    INNER JOIN {embeddingsTablename} o ON w.id = o.id
    ORDER BY o.embedding {similarityOperator} %s
    LIMIT {k};
    """
    
    if preferCommonWords:
        _cursor.execute(query, (embedding_str, embedding_str))
    else:
        _cursor.execute(queryAll, (embedding_str,))
    results = _cursor.fetchall()[:k]
    return [result[0] for result in results]
