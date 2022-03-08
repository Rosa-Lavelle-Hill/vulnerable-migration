def Create_Spreadsheet(df, first_id, last_id,
                       save_path,
                       cols=['id_num', 'customer_id'],
                       new_cols=['Moved', 'University', 'Revisit', 'Comments']):
    spreadsheet = df[cols]
    spreadsheet.drop_duplicates(inplace=True)
    spreadsheet = spreadsheet.reindex(columns=cols + new_cols)
    spreadsheet.to_csv(save_path + "ID_{}_to_{}.csv".format(first_id, last_id), index=False)
    return