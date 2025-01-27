import pandas as pd

def raw_to_transition_record():
    df = pd.read_csv("./Chicago_data/real_raw_data.csv", header=0)
    
    df['Inspection_Date'] = pd.to_datetime(df['Inspection_Date'],format="%Y-%m-%d")
    df['timeSinceLast'] = df["timeSinceLast"] * 12.0
    df = df.sort_values(by=['License', 'Inspection_Date'])
    # df = df[df["firstRecord"] == 0]
    df = df.drop(columns = ["criticalCount",'Test', 'Train', 'Unnamed: 0', 'score',"pass_flag","fail_flag"])
    labels_avg = df['criticalFound'].mean()
    # df_one = df.groupby("License").filter(lambda x: len(x) ==1)
    # df = df.groupby('License').filter(lambda x: len(x) >= 2)
    df = df.reset_index()
    
    features = list(df.columns)
    features.append("last_check")
    trans_df = []
    last_license = -1
    timeSinceLast_avg = df["timeSinceLast"].mean()
    # deal with at least twice ins establsihments 
    for i in range(0, len(df)):
        if last_license == df.loc[i]["License"]:
            time_diff = (df.loc[i]["Inspection_Date"] - df.loc[i-1]["Inspection_Date"]).days
            new_row = df.loc[i].copy().to_dict()
            new_row['timeSinceLast'] = int (time_diff / 30)
            new_row['last_check'] = df.loc[i-1]["criticalFound"]
            trans_df.append(new_row)

        else:
            new_row = df.loc[i].copy().to_dict()
            new_row['timeSinceLast'] = 10.4
            new_row['last_check'] = labels_avg
            trans_df.append(new_row)
            last_license = df.loc[i]["License"]
    
        

    trans_df = pd.DataFrame.from_dict(trans_df)
    trans_df = trans_df.drop(columns = ['index', 'Inspection_Date', "Business_ID"])
    features = list(trans_df.columns)
    return trans_df


trans_df = raw_to_transition_record()
trans_df.to_csv("./Chicago_data/transition_data_all.csv")