#----------------------------------------------------------------------------
#   Get data
#----------------------------------------------------------------------------

# External tools
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import librosa
from scipy.stats import kurtosis
from scipy.stats import skew

# Own tools
from utils import get_feature_stats, extract_features, envelope

#----------------------------------------------------------------------------
#   Function
#----------------------------------------------------------------------------

def get_data(save_locally = False,
            sub_segments = True, num_segments=5,
            cleaning = True, threshold = 0.005):
    """
    Get the data

    Adjust parameters :
    # save_locally : save the dataframe
    # sub_segments : split the audio into segments
    # num_segments : number of segments
    # cleaning : whether to apply a enveloppe mask
    # threshold : threshold of the enveloppe
    """

    arr_features=[]

    # Get data path
    path = os.path.join(os.getcwd(),'..','data')

    # Get list of genres
    genres = [f for f in os.listdir(path)]
    genres = genres

    for idx,genre in tqdm(enumerate(genres),total=len(genres)):
        # Get genre pathing
        genre_path = os.path.join(path,genre)

        for fname in os.listdir(genre_path):
            # Get file pathing
            file_path = os.path.join(genre_path,fname)
            y, sr = librosa.load(file_path, duration=30)

            # Cleaning option
            if cleaning == True :
                mask = envelope(y,sr,threshold)
                y = y[mask]

            # Split songs into sub segments
            if sub_segments == True :

                # Get subsegments
                track_duration = round(len(y)/sr)
                samples_per_track = sr * track_duration
                samples_per_segment = int(samples_per_track / num_segments)

                for d in range(num_segments):
                    start = samples_per_segment * d
                    finish = start + samples_per_segment
                    new_y = y[start:finish]
                    # Get features
                    dict_features=extract_features(y=new_y,sr=sr)
                    # Get label
                    dict_features['label']=idx
                    # Keep name
                    dict_features['id']=f"{fname}_{d}"
                    # Total track duration
                    dict_features['total_duration']=track_duration
                    dict_features['sub_duration']= round((finish-start)/sr)
                    # Append to list
                    arr_features.append(dict_features)

             # Don't split songs into subsegment
            else:
                # Get features
                dict_features=extract_features(y=y,sr=sr)
                # Get label
                dict_features['label']=idx
                # Keep name
                dict_features['id']=f"{fname}"
                # Append to list
                arr_features.append(dict_features)

    # Create a dataframe with the features
    df=pd.DataFrame(data=arr_features)

    # ----------------------------------------------------------
    # Print final output details
    # ----------------------------------------------------------

    if cleaning == True :
        print('# Envelope used for data cleaning')

    if sub_segments == True:
        print(f'# Split each sound into {num_segments} sub segments')
    else :
        print('# Sounds not split into sub segments')

    if save_locally == False:
        print('# Dataset generated but not saved')
        print('# Shape of the dataset',df.shape)
    else:
        print('# Dataset generated')
        print('# Shape of the dataset',df.shape)
        df.to_csv(os.path.join(os.getcwd(),'..','dataset','clean_df.csv'),index=False)
        print('# Dataset saved')

    return df


#----------------------------------------------------------------------------
#   Execute
#----------------------------------------------------------------------------

if __name__ == '__main__':
  df = get_data(save_locally = True, sub_segments = True, num_segments=5,
                cleaning = True, threshold = 0.005)
