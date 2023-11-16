import streamlit as st

@st.cache_data(max_entries=50)
def get_closest_words(embedding, _cursor, preferCommonWords=True, k=5):
    embedding_str = ','.join(map(str, embedding))
    embedding_str = f'[{embedding_str}]'
    query = f"""
    WITH CommonWords AS (
    SELECT word
    FROM words WHERE isCommon = true
    ORDER BY embedding <=> %s
    LIMIT {k}
),
UncommonWords AS (
    SELECT word
    FROM words WHERE isCommon = false
    ORDER BY embedding <=> %s
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
SELECT word FROM words
ORDER BY embedding <=> %s
LIMIT {k};
"""
    if(preferCommonWords):
        _cursor.execute(query, (embedding_str,embedding_str))
    else:
        _cursor.execute(queryAll, (embedding_str,))
    results = _cursor.fetchall()[:k]
    return [result[0] for result in results]
