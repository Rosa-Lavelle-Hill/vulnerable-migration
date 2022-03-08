from Functions.Database import select

def select_feature_data(q_path, first_id, last_id, id_rep=1):
    q = open(q_path, 'r').read()
    if id_rep == 2:
        q = q.format(first_id,last_id,first_id,last_id)
    else:
        q = q.format(first_id, last_id)
    df = select(q)
    return df

