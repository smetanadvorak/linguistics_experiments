import pandas as pd
import numpy as np
import copy

from eyeoi.babibo.aoi import XMLProcessor
from eyeoi.dataset import Dataset

def read_experiment_csv(file_path, all=False):
    # Read the Excel file
    df = pd.read_csv(file_path, dtype=str)

    if "P1" in file_path:
        df = df.iloc[:, 3:20]
        df = df.iloc[9:]

    elif "P2" in file_path:
        df = df.iloc[:, :17]
        df = df.iloc[5:]

    df = df.reset_index(drop=True)

    rows_to_drop = list(range(24, len(df), 25))
    df = df.drop(rows_to_drop)
    df = df.reset_index(drop=True)

    if not all:
        rows_to_drop = list(range(2, len(df), 4))
        df = df.drop(rows_to_drop)
        df = df.reset_index(drop=True)
        rows_to_drop = list(range(2, len(df), 3))
        df = df.drop(rows_to_drop)
        df = df.reset_index(drop=True)
        assert(len(df) == 36)
    else:
        assert(len(df) == 72)

    # Convert to list of dictionaries
    dict_list = df.to_dict(orient='records')
    for d in dict_list:
        try:
            d['item_type'] = d['Video_name'].split('/')[1].split('.')[0]
        except:
            pass
    return dict_list


def get_item_order(file_path):
    dict_list = read_experiment_csv(file_path, all=True)
    item_list = []
    for d in dict_list:
        if d['Video_name'] is not None:
            item_list.append(d['Video_name'])
    return item_list

def load_excel(file_path):
    df = pd.read_excel(file_path)
    rows_to_drop = list(range(2, len(df), 4))
    df = df.drop(rows_to_drop)
    df = df.reset_index(drop=True)
    rows_to_drop = list(range(2, len(df), 3))
    df = df.drop(rows_to_drop)
    df = df.reset_index(drop=True)
    return df

def get_xy(aoi):
    keyframes = aoi.find('KeyFrames').findall('KeyFrame')
    origin = [0,0]
    for kf in keyframes:
        points_list = kf.find('Points').findall('Point')
        point = points_list[0]
        origin[0] = int(point.find('X').text)
        origin[1] = int(point.find('Y').text)
    return origin


def load_reference(
        aoi_path="data_example/G10207EN-scrrec (AOIs) merged.xml", 
        box_aoi_path="data_example/G10207EN-scrrec (AOIs) - Left Right.xml", 
        xl_path1="data_example/Timeslots_AOIs_SJ_Part 1.xlsx", 
        xl_path2="data_example/Timeslots_AOIs_SJ_Part 2.xlsx"):

    xl1 = load_excel(xl_path1)
    xl2 = load_excel(xl_path2)

    combined_df = pd.concat([xl1, xl2], axis=0)
    combined_df = combined_df.reset_index(drop=True)
    print(combined_df)

    template_xml = XMLProcessor(aoi_path)
    template_xml.read_xml(drop_first_chars=0)
    print(len(template_xml.master_dict.keys()), template_xml.master_dict.keys())

    box_xml = XMLProcessor(box_aoi_path)
    box_xml.read_xml(drop_first_chars=0)
    print(len(box_xml.master_dict.keys()), box_xml.master_dict.keys())

    l_origin = get_xy(box_xml.master_dict["LEFT"])
    r_origin = get_xy(box_xml.master_dict["RIGHT"])

    print(f"Left origin: {l_origin}, Right origin: {r_origin}")

    aoi_ref_dict = {}
    for index, row in combined_df.iterrows():
        aoi_l = template_xml.master_dict[row["AOI_LEFT"]]
        aoi_r = template_xml.master_dict[row["AOI_RIGHT"]]
        aoi_ref = {
            row["Video_RIGHT"]: {
                "LEFT": XMLProcessor.shift_aoi(copy.deepcopy(aoi_r), l_origin, r_origin),
                "RIGHT": copy.deepcopy(aoi_r),
            },
            row["Video_LEFT"]: {
                "LEFT": copy.deepcopy(aoi_l), 
                "RIGHT": XMLProcessor.shift_aoi(copy.deepcopy(aoi_l), r_origin, l_origin),
            }
        }

        if row["Video_name"] in aoi_ref_dict:
            raise Exception(f"{row["Video_name"]} already in ") 
        
        aoi_ref_dict[row["Video_name"]] = aoi_ref

    [print(f"{k}:{v}\n") for k,v in aoi_ref_dict.items()]

    return aoi_ref_dict
    # return combined_df, template_xml, (l_origin, r_origin)


    


'''
python3 load_reference.py "data_example/G10207EN-scrrec (AOIs) merged.xml" "data_example/psycho/G10207EN_P1.csv" "data_example/psycho/G10207EN_P2.csv"

python3 load_reference.py "data_example/G10207EN-scrrec (AOIs) merged.xml" "data_example/Timeslots_AOIs_SJ_Part 1.xlsx" "data_example/Timeslots_AOIs_SJ_Part 2.xlsx"
'''