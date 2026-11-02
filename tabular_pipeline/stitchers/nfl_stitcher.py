import os
import glob
import pandas as pd
import numpy as np

def stitch_nfl_dataset(
    nfl_dir: str = "nfl-big-data-bowl-2026-prediction",
    output_dir: str = "cases/nfl_big_data_bowl_2026",
    max_weeks: int = 3,
):
    """
    Transforms multi-week spatio-temporal NFL tracking files into clean tabular matrices.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Stitching NFL dataset from {nfl_dir} (Max weeks: {max_weeks})...")
    
    # 1. Process Train Weeks
    train_dir = os.path.join(nfl_dir, "train")
    input_files = sorted(glob.glob(os.path.join(train_dir, "input_2023_w*.csv")))[:max_weeks]
    
    train_dfs = []
    for inp_p in input_files:
        week_num = os.path.basename(inp_p).replace("input_2023_", "").replace(".csv", "")
        out_p = os.path.join(train_dir, f"output_2023_{week_num}.csv")
        if not os.path.exists(out_p):
            continue
            
        print(f"  Processing {week_num}...")
        inp_df = pd.read_csv(inp_p)
        out_df = pd.read_csv(out_p)
        
        # Filter strictly for players that need prediction
        p_inp = inp_df[inp_df["player_to_predict"] == True].copy()
        
        # Extract the last observed frame (T_0 state)
        last_states = (
            p_inp.sort_values("frame_id")
            .groupby(["game_id", "play_id", "nfl_id"])
            .last()
            .reset_index()
        )
        
        # Join with future frame targets
        merged = out_df.merge(
            last_states,
            on=["game_id", "play_id", "nfl_id"],
            suffixes=("_future", "_last")
        )
        train_dfs.append(merged)
        
    train_tabular = pd.concat(train_dfs, ignore_index=True)
    
    # Rename target columns cleanly
    train_tabular.rename(columns={
        "frame_id_future": "future_frame_id",
        "x_future": "target_x",
        "y_future": "target_y",
        "x_last": "x_0",
        "y_last": "y_0",
        "s": "s_0",
        "a": "a_0",
        "dir": "dir_0",
        "o": "o_0",
    }, inplace=True)
    
    train_tabular["id"] = (
        train_tabular["game_id"].astype(str) + "_" + 
        train_tabular["play_id"].astype(str) + "_" + 
        train_tabular["nfl_id"].astype(str) + "_" + 
        train_tabular["future_frame_id"].astype(str)
    )
    
    train_out_path = os.path.join(output_dir, "train_tabular.csv")
    train_tabular.to_csv(train_out_path, index=False)
    print(f"Train tabular saved: {train_out_path} ({len(train_tabular):,} rows)")
    
    # 2. Process Test Dataset
    test_path = os.path.join(nfl_dir, "test.csv")
    test_input_path = os.path.join(nfl_dir, "test_input.csv")
    
    test_out_path = None
    if os.path.exists(test_path) and os.path.exists(test_input_path):
        print("  Processing test dataset...")
        test_df = pd.read_csv(test_path)
        test_in_df = pd.read_csv(test_input_path)
        
        p_test_in = test_in_df[test_in_df["player_to_predict"] == True].copy()
        last_test_states = (
            p_test_in.sort_values("frame_id")
            .groupby(["game_id", "play_id", "nfl_id"])
            .last()
            .reset_index()
        )
        
        test_tabular = test_df.merge(
            last_test_states,
            on=["game_id", "play_id", "nfl_id"],
            suffixes=("_target", "_last")
        )
        
        test_tabular.rename(columns={
            "frame_id_target": "future_frame_id",
            "x": "x_0",
            "y": "y_0",
            "s": "s_0",
            "a": "a_0",
            "dir": "dir_0",
            "o": "o_0",
        }, inplace=True)
        
        test_out_path = os.path.join(output_dir, "test_tabular.csv")
        test_tabular.to_csv(test_out_path, index=False)
        print(f"Test tabular saved: {test_out_path} ({len(test_tabular):,} rows)")
        
    return train_out_path, test_out_path

if __name__ == "__main__":
    stitch_nfl_dataset()
