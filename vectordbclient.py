import psycopg2

def get_closest_words(embedding, cursor, preferCommonWords=True):
    embedding_str = ','.join(map(str, embedding))
    embedding_str = f'[{embedding_str}]'
    query = """
    WITH CommonWords AS (
    SELECT word
    FROM words WHERE isCommon = true
    ORDER BY embedding <=> %s
    LIMIT 10
),
UncommonWords AS (
    SELECT word
    FROM words WHERE isCommon = false
    ORDER BY embedding <=> %s
    LIMIT 10
)

SELECT word
FROM (
    SELECT word FROM CommonWords
    UNION ALL
    SELECT word FROM UncommonWords
) AS Combined
LIMIT 10;
    """

    queryAll = """
SELECT word FROM words
ORDER BY embedding <=> %s
LIMIT 10;
"""
    if(preferCommonWords):
        cursor.execute(query, (embedding_str,embedding_str))
    else:
        cursor.execute(queryAll, (embedding_str,))
    results = cursor.fetchall()[:5]
    return [result[0] for result in results]

def reflect_across_vector(normal, target):
    """
    Reflects the target vector across a plane orthogonal to the normal vector.
    Both vectors are assumed to be normalized and within an n-dimensional space.

    Parameters:
    normal (np.array): The normalized vector that defines the reflection plane.
    target (np.array): The normalized vector to be reflected.

    Returns:
    np.array: The reflected vector, normalized and in the same n-dimensional space.
    """
    return normal + (normal - target)