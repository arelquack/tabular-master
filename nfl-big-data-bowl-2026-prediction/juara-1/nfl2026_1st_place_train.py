# %% [markdown]
# This is a subset of the training code from the 1st Place Solution.  
# The differences from the final model training code are as follows:
# * In the final setup, 5-fold splits are generated with three different seeds, resulting in a total of 15 models, whereas this code uses only seed = 0.
# * ~~In the final model, multiple models are trained with different configurations of dynamic features, but this code uses only a single configuration.~~
# * Now you can choice from 3 configurations. 
# 
# Overall, the CV score should be around 0.47.
# 
# 
# 

# %% [markdown]
# ~~### notice~~
# + ~~If you want to change the CV split for each seed when training models, please upgrade your scikit-learn version.~~
# + ~~For some reason, in the notebook environment, I was no longer able to upgrade scikit-learn to version 1.7.0.~~
# + Changing the session option to use the latest version increased the default version and resolved the issue.

# %% [code] {"execution":{"iopub.status.busy":"2026-01-20T09:49:34.187374Z","iopub.execute_input":"2026-01-20T09:49:34.187598Z","iopub.status.idle":"2026-01-20T09:49:42.379176Z","shell.execute_reply.started":"2026-01-20T09:49:34.187575Z","shell.execute_reply":"2026-01-20T09:49:42.377930Z"}}

# !pip install scikit-learn==1.7.0

# %% [markdown]
# 

# %% [code] {"execution":{"iopub.status.busy":"2026-01-20T09:49:42.381691Z","iopub.execute_input":"2026-01-20T09:49:42.383721Z","iopub.status.idle":"2026-01-20T09:49:42.395023Z","shell.execute_reply.started":"2026-01-20T09:49:42.383675Z","shell.execute_reply":"2026-01-20T09:49:42.393622Z"}}
from sklearn.model_selection import GroupKFold
import json
import numpy as np
# import wandb
import pandas as pd
import os
import random
import pathlib
from tqdm import tqdm
from timm.utils import ModelEmaV3
import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import sklearn
print(sklearn.__version__)

# %% [markdown]
# ## Constant values and pre-computed feature statistics

# %% [code] {"execution":{"iopub.status.busy":"2026-01-20T09:49:42.395957Z","iopub.execute_input":"2026-01-20T09:49:42.396293Z","iopub.status.idle":"2026-01-20T09:49:42.414700Z","shell.execute_reply.started":"2026-01-20T09:49:42.396258Z","shell.execute_reply":"2026-01-20T09:49:42.413867Z"}}

# constant values
FIELD_WIDTH: float = 120.0
FIELD_HEIGHT: float = 53.3
MAX_PLAYER_NUM: int = 22
MAX_IN_PLAYER_NUM: int = 22
MAX_OUT_PLAYER_NUM: int = 22
player_roles = ['Defensive Coverage', 'Targeted Receiver', 'Passer', 'Other Route Runner']
player_role_to_idx = {role: idx for idx, role in enumerate(player_roles)}
src_feat_names = ["x", "y", "s", "a", "dir", "o"]
src_feat_idx = {name: i for i, name in enumerate(src_feat_names)}

# Pre-computed feature statistics
feature_mean = {'x': 60.82776974002471,
                'y': 26.66586819309761,
                'ori_sin': -0.028771674664162084,
                'ori_cos': -0.0860770011312027,
                'distance_to_ball': 19.420012294946886,
                'distance_to_receiver': 13.905649795712955,
                'ball_land_x_dx': 4.807716986443858,
                'ball_land_y_dy': -0.6598223350763681,
                'receiver_dx': -3.034959448458881,
                'receiver_dy': -0.3004955149788581,
                'dir_sin_s': -0.04365514987349996,
                'dir_cos_s': 1.7474138934015886,
                'dir_sin_a': -0.022489013627792452,
                'dir_cos_a': 1.0584119108147563,
                'diff_dir_o_sin': -0.0064997680282648995,
                'diff_dir_o_cos': 0.1363704308055112,
                'player_role_encoded': 1.1193698534538628,
                'num_frame_output': 11.64147,
                'passer_x': 49.05013399503722,
                'passer_y': 26.528289259127966,
                'ball_land_x': 65.33169440307206,
                'ball_land_y': 26.221190352646918,
                'rot_x': -12.97965036,
                'rot_y': 0.54228895,
                'dir_sin': -0.008201030098231301,
                'dir_cos': 0.45047703362132063,
                'a': 2.118334968043749,
                's': 3.0198784693373515,
                'diff_x': 0.1789712431138102,
                'diff_y': -0.004446990066127387,
                'player_last_x': 60.82776974002471,
                'player_last_y': 26.66586819309761,
                'passer_last_x': 49.05013399503722,
                'passer_last_y': 26.528289259127966,
                }


feature_std = {'x': 23.479941313323085,
               'y': 10.00749618625406,
               'ori_sin': 0.543936436139763,
               'ori_cos': 0.8342039901959803,
               'distance_to_ball': 10.76441761817005,
               'distance_to_receiver': 8.820358107267353,
               'ball_land_x_dx': 12.124398080266454,
               'ball_land_y_dy': 17.957146201379818,
               'receiver_dx': 7.586012961533925,
               'receiver_dy': 14.293949140615915,
               'dir_sin_s': 2.29624100410567,
               'dir_cos_s': 2.399019308047086,
               'dir_sin_a': 1.5753721593310632, 'dir_cos_a': 1.699787386078158,
               'diff_dir_o_sin': 0.6532995066440124,
               'diff_dir_o_cos': 0.7446884008166831,
               'player_role_encoded': 1.3341219458609044,
               'num_frame_output': 5.3315372,
               'passer_x': 23.15646164494253,
               'passer_y': 4.039143779870031,
               'ball_land_x': 24.721183980071213,
               'ball_land_y': 15.253264763883333,
               'rot_x': 12.98279418,
               'rot_y': 12.47606231,
               'dir_sin': 0.6209186583613782,
               'dir_cos': 0.6414541045817329,
               'a': 1.4157939116699676,
               's': 2.227938568313042,
               'diff_x': 0.24064030768643296,
               'diff_y': 0.23067443165381496,
               'player_last_x': 23.479941313323085,
               'player_last_y': 10.00749618625406,
               'passer_last_x': 23.15646164494253,
               'passer_last_y': 4.039143779870031,
               }


target_mean = {'x': 69.50731170150776,
               'y': 26.49108062017707,
               'dx': 2.4863890033680556,
               'dy': -0.0003338034874301969,
               'x0': 67.02092269813976,
               'y0': 26.49141442366451,
               'ball_land_x_dx': 2.1796843109462185,
               'ball_land_y_dy': 0.01456915004472861,
               'sx': 0.24357714,
               'sy': -0.00259809,
               'ax': -0.00855656,
               'ay': 0.00024983,
               }
target_std = {'x': 23.435529312124537,
              'y': 13.445279179127226,
              'dx': 3.9917733048457364,
              'dy': 3.728388769750037,
              'x0': 22.991527135040435,
              'y0': 11.787700462682832,
              'ball_land_x_dx': 7.819356803570175,
              'ball_land_y_dy': 7.527393631406031,
              'sx': 0.30963882,
              'sy': 0.42498858,
              'ax': 0.02965129,
              'ay': 0.0291162,

              }


# %% [markdown]
# ## Util functions

# %% [code] {"execution":{"iopub.status.busy":"2026-01-20T09:49:42.415480Z","iopub.execute_input":"2026-01-20T09:49:42.415770Z","iopub.status.idle":"2026-01-20T09:49:42.427203Z","shell.execute_reply.started":"2026-01-20T09:49:42.415737Z","shell.execute_reply":"2026-01-20T09:49:42.426501Z"}}
def fix_seed(seed=0):

    os.environ['PYTHONHASHSEED'] = str(seed)
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calc_score(valid_df):
    diff = valid_df[["x", "y"]].values - valid_df[["pred_x", "pred_y"]].values
    score = (diff**2).mean()**0.5
    return score

# %% [markdown]
# ## Data load and preprocess functions

# %% [code] {"execution":{"iopub.status.busy":"2026-01-20T09:49:42.428462Z","iopub.execute_input":"2026-01-20T09:49:42.428775Z","iopub.status.idle":"2026-01-20T09:49:42.451914Z","shell.execute_reply.started":"2026-01-20T09:49:42.428750Z","shell.execute_reply":"2026-01-20T09:49:42.451236Z"}}
def unify_play_direction_input(df):
    df = df.copy()
    play_direction_mask = df["play_direction"] == "left"
    df.loc[play_direction_mask, "x"] = FIELD_WIDTH - df.loc[play_direction_mask, "x"]
    df.loc[play_direction_mask, "y"] = FIELD_HEIGHT - df.loc[play_direction_mask, "y"]
    df.loc[play_direction_mask, "ball_land_x"] = FIELD_WIDTH - df.loc[play_direction_mask, "ball_land_x"]
    df.loc[play_direction_mask, "ball_land_y"] = FIELD_HEIGHT - df.loc[play_direction_mask, "ball_land_y"]
    df.loc[play_direction_mask, "dir"] = (df.loc[play_direction_mask, "dir"] + 180.0) % 360.0
    df.loc[play_direction_mask, "o"] = (df.loc[play_direction_mask, "o"] + 180.0) % 360.0
    return df


def unify_play_direction_output(df):
    df = df.copy()
    play_direction_mask = df["play_direction"] == "left"
    df.loc[play_direction_mask, "x"] = FIELD_WIDTH - df.loc[play_direction_mask, "x"]
    df.loc[play_direction_mask, "y"] = FIELD_HEIGHT - df.loc[play_direction_mask, "y"]
    return df


def read_csvs(TRAIN_PATH):

    csv_cols = ['game_id', 'play_id', 'player_to_predict', 'nfl_id', 'frame_id', 'play_direction', 'player_side', 'player_role',
                'x', 'y', 's', 'a', 'dir', 'o', 'num_frames_output', 'ball_land_x', 'ball_land_y']
    df_dtype = {'game_id': str, "play_id": str, "nfl_id": str}
    input_paths = sorted(TRAIN_PATH.glob("input_2023_w*.csv"))
    input_df = pd.concat((pd.read_csv(p, dtype=df_dtype, usecols=csv_cols) for p in input_paths), ignore_index=True)

    # Convert angle-related values to match the mathematical convention
    # (0 degrees at 12 o’clock & clockwise → 0 degrees at 3 o’clock & counterclockwise)
    input_df[["dir", "o"]] = (90.0 - input_df[["dir", "o"]]) % 360.0

    # Align play direction to the right
    input_df = unify_play_direction_input(input_df)

    output_paths = sorted(TRAIN_PATH.glob("output_2023_w*.csv"))
    output_df = pd.concat((pd.read_csv(p, dtype=df_dtype) for p in output_paths), ignore_index=True)

    input_df["game_play_id"] = input_df["game_id"] + "_" + input_df["play_id"]
    output_df["game_play_id"] = output_df["game_id"] + "_" + output_df["play_id"]

    # merge output_df and subset of input_df
    tmp_df = input_df[['game_id', 'play_id', 'nfl_id', 'player_to_predict', 'play_direction', 'x', 'y']].copy()
    tmp_df = tmp_df.query("player_to_predict==True").drop("player_to_predict", axis=1).reset_index(drop=True)
    tmp_df = tmp_df.rename({"x": "x_last", "y": "y_last"}, axis=1)
    tmp_df = tmp_df.drop_duplicates(subset=["game_id", "play_id", "nfl_id"], keep="last").reset_index(drop=True)
    output_df = pd.merge(output_df, tmp_df, on=["game_id", "play_id", "nfl_id"], how="inner", suffixes=("_exp", "_out"))

    # Align play direction to the right
    output_df = unify_play_direction_output(output_df)

    return input_df, output_df

# %%

def make_fold_df(input_df, n_fold):
    fold_df = input_df[["game_id", "play_id", "player_to_predict"]].copy()
    fold_df = fold_df.query("player_to_predict==True").reset_index(drop=True)

    fold_df = fold_df[["game_id", "play_id"]]
    fold_df = fold_df.drop_duplicates().reset_index(drop=True)
    fold_df["game_play_id"] = fold_df["game_id"] + "_" + fold_df["play_id"]

    # remove too long play
    too_long_ids = ["2023091100_3167", "2023122100_1450"]
    # remove no passer play
    no_passer_ids = ["2023091001_3216", "2023112606_4180", "2023121009_3594"]
    exclude_ids = too_long_ids + no_passer_ids
    fold_df = fold_df[~fold_df["game_play_id"].isin(exclude_ids)].reset_index(drop=True)


    for seed in range(10):
        # If you want to change the CV split for each seed when training models, please upgrade your scikit-learn version.
        # For some reason, in the notebook environment, I was no longer able to upgrade scikit-learn to version 1.7.0.
        # !pip install scikit-learn==1.7.0        
        gkf = GroupKFold(n_splits=n_fold, shuffle=True, random_state=seed)
        # gkf = GroupKFold(n_splits=n_fold)
        fold_df[f"fold{seed}"] = -1
        for fold, (train_idx, val_idx) in enumerate(gkf.split(fold_df, groups=fold_df["game_id"])):
            fold_df.loc[val_idx, f"fold{seed}"] = fold
    
    return fold_df

def make_data_dict(input_df, output_df):
    # Create a dictionary of per-play input/output data indexed by game_play_id.
    # Although keeping the dictionary contents as DataFrames would make processing
    # inside the Dataset easier, NumPy arrays are used instead for performance reasons.
    # For each play, this function groups input and output DataFrames, reshapes
    # features into (player, frame, feature) arrays, aligns player ordering so
    # that output-target players come first, and collects auxiliary information
    # (player metadata, receiver/passer positions, and ball landing position)
    # needed for model training and inference.

    input_groups = input_df.groupby("game_play_id")
    output_groups = output_df.groupby("game_play_id")

    game_play_ids = sorted(input_df["game_play_id"].unique().tolist())
    data_dict = {}
    for game_play_id in tqdm(game_play_ids):

        i_df = input_groups.get_group(game_play_id).reset_index(drop=True)
        o_df = output_groups.get_group(game_play_id).reset_index(drop=True)

        i_info_df = i_df[["nfl_id", "player_role", "player_to_predict"]].drop_duplicates().reset_index(drop=True)
        i_nfl_ids = i_df["nfl_id"].drop_duplicates().values.copy()
        o_nfl_ids = o_df["nfl_id"].drop_duplicates().values.copy()

        # Generate indexes to reorder nfl_ids
        # The reordering rules are as follows:
        # - IDs that match those in o_nfl_ids take the same order as in o_nfl_ids
        # - IDs that do not match are appended afterward while preserving their original order
        # This allows us to simply take the first len(o_nfl_ids) elements
        sort_order = [i for i, x in enumerate(i_nfl_ids) if x in set(o_nfl_ids)]
        sort_order += [i for i, x in enumerate(i_nfl_ids) if x not in set(o_nfl_ids)]

        # Get sizes
        num_player_input = len(i_nfl_ids)
        num_frame_input = len(i_df) // num_player_input
        num_frame_output = o_df["frame_id"].max()
        num_player_output = len(o_nfl_ids)

        i_data = i_df[src_feat_names].values.astype(np.float32)
        o_data = o_df[["x", "y"]].values.astype(np.float32)
        i_data = i_data.reshape(num_player_input, num_frame_input, -1)
        o_data = o_data.reshape(num_player_output, num_frame_output, -1)

        receiver_xy = i_df[i_df["player_role"] == "Targeted Receiver"][["x", "y"]].values.astype(np.float32)
        passer_xy = i_df[i_df["player_role"] == "Passer"][["x", "y"]].values.astype(np.float32)
        ball_land_xy = i_df.loc[0, ["ball_land_x", "ball_land_y"]].values.astype(np.float32)

        # Sort players in the order: target players first, then others
        i_data = i_data[sort_order]
        i_nfl_ids = [i_nfl_ids[i] for i in sort_order]
        i_info_df = i_info_df.iloc[sort_order].reset_index(drop=True)

        data_dict[game_play_id] = {
            "i_data": i_data,
            "o_data": o_data,
            "o_df": o_df,
            "i_info_df": i_info_df,
            "receiver_xy": receiver_xy,
            "passer_xy": passer_xy,
            "ball_land_xy": ball_land_xy,
        }
    return data_dict




# %% [markdown]
# ## Dataset and Dataloader

# %% [code] {"execution":{"iopub.status.busy":"2026-01-20T09:49:42.453000Z","iopub.execute_input":"2026-01-20T09:49:42.453779Z","iopub.status.idle":"2026-01-20T09:49:42.507258Z","shell.execute_reply.started":"2026-01-20T09:49:42.453749Z","shell.execute_reply":"2026-01-20T09:49:42.506533Z"}}
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self,
                 df,
                 data_dict,
                 CFG,
                 data_repeat_factor=1,
                 is_train=False,
                 ):
        pass
        self.df = df.copy()
        self.data_dict = data_dict.copy()
        self.data_repeat_factor = data_repeat_factor
        self.input_sequence_size = CFG.input_sequence_size
        self.output_sequence_size = CFG.output_sequence_size
        self.is_train = is_train
        self.ext_feat_names = CFG.ext_feat_names.copy()
        self.ext_feat_idx = {name: i for i, name in enumerate(self.ext_feat_names)}

        # augumentation parametors
        self.x_bias_std = CFG.x_bias_std
        self.y_bias_std = CFG.y_bias_std
        self.flipud_rate = CFG.flipud_rate

        self.angle_uni_rate = CFG.angle_uni_rate
        self.angle_std_rate = CFG.angle_std_rate
        self.angle_std = CFG.angle_std

        self.max_shift_size = CFG.max_shift_size
        self.least_input_frame = 5
        self.frame_shift_rate = CFG.frame_shift_rate

        # feature/target statistics
        self.feat_mean = np.array([feature_mean[name] for name in self.ext_feat_names])
        self.feat_std = np.array([feature_std[name] for name in self.ext_feat_names])
        self.target_mean = np.array([target_mean[name] for name in ("dx", "dy")])
        self.target_std = np.array([target_std[name] for name in ("dx", "dy")])

        self.target_sa_mean = np.array([target_mean[name] for name in ("sx", "sy", "ax", "ay")])
        self.target_sa_std = np.array([target_std[name] for name in ("sx", "sy", "ax", "ay")])

        self.player_last_xy_mean = np.array([feature_mean[name] for name in ("player_last_x", "player_last_y")])
        self.player_last_xy_std = np.array([feature_std[name] for name in ("player_last_x", "player_last_y")])

        self.passer_last_xy_mean = np.array([feature_mean[name] for name in ("passer_last_x", "passer_last_y")])
        self.passer_last_xy_std = np.array([feature_std[name] for name in ("passer_last_x", "passer_last_y")])

        self.ball_land_xy_mean = np.array([feature_mean[name] for name in ("ball_land_x", "ball_land_y")])
        self.ball_land_xy_std = np.array([feature_std[name] for name in ("ball_land_x", "ball_land_y")])

        self.num_frame_output_mean = np.array([feature_mean[name] for name in ("num_frame_output",)])
        self.num_frame_output_std = np.array([feature_std[name] for name in ("num_frame_output",)])

        # dataframe for evaluation
        valid_df = []
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            data = self.data_dict[row["game_play_id"]]
            valid_df.append(data["o_df"].copy())

        self.valid_df = pd.concat(valid_df).reset_index(drop=True)

    def __len__(self):
        return len(self.df) * self.data_repeat_factor

    def target_scaling(self, target):
        return (target - self.target_mean) / self.target_std

    def feature_scaling(self, feature):
        return (feature - self.feat_mean) / self.feat_std

    def target_rescaling(self, target):
        return (target * self.target_std) + self.target_mean


    def feat_pad(self, x, max_num_player=MAX_PLAYER_NUM, input_sequence_size=20, val=0.0):

        assert x.shape[0] <= max_num_player
        assert x.shape[1] <= input_sequence_size

        pad0 = max_num_player - x.shape[0]
        pad1 = input_sequence_size - x.shape[1]
        x = np.pad(x,
                   ((0, pad0), (pad1, 0), (0, 0)),
                   mode='constant',
                   constant_values=val)
        return x

    def target_pad(self, x, max_num_player=MAX_PLAYER_NUM, output_sequence_size=48, val=0.0):

        assert x.shape[0] <= max_num_player
        assert x.shape[1] <= output_sequence_size

        pad0 = max_num_player - x.shape[0]
        pad1 = output_sequence_size - x.shape[1]
        x = np.pad(x,
                   ((0, pad0), (0, pad1), (0, 0)),
                   mode='constant',
                   constant_values=val)
        return x



    def __flipud__(self, i_data, o_data, ball_land_xy):

        if self.flipud_rate and random.random() < self.flipud_rate:
            i_data = i_data.copy()
            o_data = o_data.copy()
            ball_land_xy = ball_land_xy.copy()
            i_data[:, :, src_feat_idx["y"]] = FIELD_HEIGHT - i_data[:, :, src_feat_idx["y"]]  # y
            i_data[:, :, src_feat_idx["dir"]] = (360.0 - i_data[:, :, src_feat_idx["dir"]]) % 360.0  # dir
            i_data[:, :, src_feat_idx["o"]] = (360.0 - i_data[:, :, src_feat_idx["o"]]) % 360.0  # o
            o_data[:, :, 1] = FIELD_HEIGHT - o_data[:, :, 1]  # y
            ball_land_xy[1] = FIELD_HEIGHT - ball_land_xy[1]  # y
        return i_data, o_data, ball_land_xy

    def __rotation__(self, i_data, o_data, ball_land_xy):

        angle_rad = 0.0
        # Probability of applying rotation (Uniform + Gaussian)
        apply_rate = self.angle_std_rate + self.angle_uni_rate

        # Do not apply rotation at all
        if apply_rate == 0.0:
            return i_data, o_data, ball_land_xy, angle_rad

        # Decide whether to apply rotation based on a random value in [0, 1]
        r = random.random()
        if r >= apply_rate:
            return i_data, o_data, ball_land_xy, angle_rad

        if r < self.angle_uni_rate:
            # Region where Uniform (0–360°) rotation is applied
            angle = np.random.uniform(0.0, 360.0)
        else:
            # Region corresponding to Gaussian rotation (self.angle_std_rate)
            angle = np.random.normal(loc=0, scale=self.angle_std)

        angle %= 360.0
        angle_rad = np.deg2rad(angle)

        # The rotation center is the centroid of player positions at the final input time step
        center = i_data[:, -1][:, [src_feat_idx["x"], src_feat_idx["y"]]].mean(axis=0)

        R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)], [np.sin(angle_rad), np.cos(angle_rad)]])

        # INPUT
        xy = i_data[:, :, :][:, :, [src_feat_idx["x"], src_feat_idx["y"]]]

        # rotation corrdinate
        rot_xy = (xy - center) @ R.T + center
        i_data[:, :, src_feat_idx["x"]] = rot_xy[:, :, 0]
        i_data[:, :, src_feat_idx["y"]] = rot_xy[:, :, 1]
        
        # rotation angle
        i_data[:, :, src_feat_idx["dir"]] = (i_data[:, :, src_feat_idx["dir"]] + angle) % 360.0
        i_data[:, :, src_feat_idx["o"]]   = (i_data[:, :, src_feat_idx["o"]]   + angle) % 360.0


        # OUTPUT
        xy = o_data[:, :, :][:, :, [0, 1]]  #
        rot_xy = (xy - center) @ R.T + center
        o_data[:, :, 0] = rot_xy[:, :, 0]
        o_data[:, :, 1] = rot_xy[:, :, 1]

        # BALL
        ball_land_xy = (ball_land_xy - center) @ R.T + center

        return i_data, o_data, ball_land_xy, angle_rad

    def __frame_shift__(self, i_data, o_data):

        num_frame_output = o_data.shape[1]
        num_frame_input = i_data.shape[1]
        shift_size = 0
        if self.frame_shift_rate and random.random() < self.frame_shift_rate:
            # Maximum sequence length that can be extracted from the input
            max_size_from_input = min(self.max_shift_size, max(0, num_frame_input - self.least_input_frame))
            # Maximum length that can be concatenated to the output
            # (-1 is not necessary, but kept for compatibility)
            max_size = min(max_size_from_input - 1, self.output_sequence_size - num_frame_output)
            if max_size >= 1:
                shift_size = abs(np.random.normal(0, max_size / 3.0))
                shift_size = int(np.clip(np.round(shift_size), 0, max_size))

                if shift_size > 0:
                    input_to_output_xy = i_data[:len(o_data), -shift_size:, :2]
                    i_data = i_data[:, :-shift_size]
                    o_data = np.concatenate(
                        (input_to_output_xy.astype(o_data.dtype), o_data),
                        axis=1
                    )
        return i_data, o_data, shift_size


    def __xy_shift__(self, i_data, o_data, ball_land_xy):
        # Apply a random global shift in the x-direction if x_bias_std is specified
        if self.x_bias_std:
            x_shift = np.random.normal(loc=0.0, scale=self.x_bias_std)
            i_data[:, :, src_feat_idx["x"]] = i_data[:, :, src_feat_idx["x"]] + x_shift
            o_data[:, :, 0] = o_data[:, :, 0] + x_shift
            ball_land_xy[0] = ball_land_xy[0] + x_shift
        # Apply a random global shift in the y-direction if y_bias_std is specified
        if self.y_bias_std:
            y_shift = np.random.normal(loc=0.0, scale=self.y_bias_std)
            i_data[:, :, src_feat_idx["y"]] = i_data[:, :, src_feat_idx["y"]] + y_shift
            o_data[:, :, 1] = o_data[:, :, 1] + y_shift
            ball_land_xy[1] = ball_land_xy[1] + y_shift
        # Return shifted input data, output data, and ball landing position
        return i_data, o_data, ball_land_xy

    def __getitem__(self, i):
        # The dataset size is virtually increased using data_repeat_factor.
        # Therefore, a modulo operation is required.
        i = i % len(self.df)
        row = self.df.iloc[i]

        # data copy
        data = self.data_dict[row["game_play_id"]].copy()
        i_data = data["i_data"].copy()
        o_data = data["o_data"].copy()
        i_info_df = data["i_info_df"].copy()
        ball_land_xy = data["ball_land_xy"].copy()
        receiver_idx = i_info_df.index[i_info_df["player_role"] == "Targeted Receiver"].tolist()[0]
        passer_idx = i_info_df.index[i_info_df["player_role"] == "Passer"].tolist()[0]


        # Data augumentaiton start
        frame_shift_size = 0.0
        if self.is_train:
            i_data, o_data, frame_shift_size = self.__frame_shift__(i_data, o_data)
            i_data, o_data, ball_land_xy, _ = self.__rotation__(i_data, o_data, ball_land_xy)
            i_data, o_data, ball_land_xy = self.__flipud__(i_data, o_data, ball_land_xy)
            i_data, o_data, ball_land_xy = self.__xy_shift__(i_data, o_data, ball_land_xy)
        # Data augumentaiton end


        # get size
        num_player_input, num_frame_input, _ = i_data.shape
        num_player_output, num_frame_output, _ = o_data.shape

        #
        receiver_xy = i_data[receiver_idx][:, [src_feat_idx["x"], src_feat_idx["y"]]]
        passer_xy = i_data[passer_idx][ :, [src_feat_idx["x"], src_feat_idx["y"]]]

        # extruct features
        ext_feat_idx = self.ext_feat_idx
        ext_feat_names = self.ext_feat_names
        ext_data = np.zeros((num_player_input, num_frame_input, len(ext_feat_names)), dtype=np.float32)
        if "x" in self.ext_feat_names:
            ext_data[:, :, ext_feat_idx["x"]] = i_data[:, :, src_feat_idx["x"]].copy()
        if "y" in self.ext_feat_names:
            ext_data[:, :, ext_feat_idx["y"]] = i_data[:, :, src_feat_idx["y"]].copy()
        if "ori_sin" in self.ext_feat_names:
            ext_data[:, :, ext_feat_idx["ori_sin"]] = np.sin(np.deg2rad(i_data[:, :, src_feat_idx["o"]]))
        if "ori_cos" in self.ext_feat_names:
            ext_data[:, :, ext_feat_idx["ori_cos"]] = np.cos(np.deg2rad(i_data[:, :, src_feat_idx["o"]]))
        if "dir_sin_s" in self.ext_feat_names:
            ext_data[:, :, ext_feat_idx["dir_sin_s"]] = np.sin(np.deg2rad(i_data[:, :, src_feat_idx["dir"]])) * i_data[:, :, src_feat_idx["s"]]
        if "dir_cos_s" in self.ext_feat_names:
            ext_data[:, :, ext_feat_idx["dir_cos_s"]] = np.cos(np.deg2rad(i_data[:, :, src_feat_idx["dir"]])) * i_data[:, :, src_feat_idx["s"]]
        if "ball_land_x_dx" in self.ext_feat_names:
            ext_data[:, :, ext_feat_idx["ball_land_x_dx"]] = ball_land_xy[0] - i_data[:, :, src_feat_idx["x"]]
        if "ball_land_y_dy" in self.ext_feat_names:
            ext_data[:, :, ext_feat_idx["ball_land_y_dy"]] = ball_land_xy[1] - i_data[:, :, src_feat_idx["y"]]
        if "receiver_dx" in self.ext_feat_names:
            ext_data[:, :, ext_feat_idx["receiver_dx"]] = receiver_xy[:,0] - i_data[:, :, src_feat_idx["x"]]
        if "receiver_dy" in self.ext_feat_names:
            ext_data[:, :, ext_feat_idx["receiver_dy"]] = receiver_xy[:, 1] - i_data[:, :, src_feat_idx["y"]]

        # Vector from ball landing point to the passer
        # Angle of that vector relative to the x-axis
        # Rotate coordinates so the passer direction aligns with the positive x-axis
        rot_passer_xy = passer_xy[-1] - ball_land_xy # パサーの最終座標とボール座標の差
        theta = np.arctan2(rot_passer_xy[1], rot_passer_xy[0])
        tmp_xy = i_data[:, :][:,:,[src_feat_idx["x"],src_feat_idx["y"]]].copy()
        shift_xy = tmp_xy - ball_land_xy
        rot_x = np.cos(theta) * shift_xy[:,:,0] + np.sin(theta) * shift_xy[:,:,1]
        rot_y = -np.sin(theta) * shift_xy[:,:, 0] + np.cos(theta) * shift_xy[:,:, 1]
        if "rot_x" in self.ext_feat_names:
            ext_data[:, :, ext_feat_idx["rot_x"]] = rot_x
        if "rot_y" in self.ext_feat_names:
            ext_data[:, :, ext_feat_idx["rot_y"]] = rot_y

        # The target is defined as the offset from the final input coordinates
        last_xy = i_data[:num_player_output, -1:, :]
        last_xy = last_xy[:, :, [src_feat_idx["x"], src_feat_idx["y"]]]
        target_xy = o_data - last_xy

        # Extract the last 20 frames
        ext_data = ext_data[:, -self.input_sequence_size:].copy()

        # Target data for auxiliary loss､speed and acceleration
        last_xy_input = i_data[:len(o_data), -2:][:,:,[src_feat_idx["x"], src_feat_idx["y"]]].copy()
        targets_s = np.diff(np.concatenate((last_xy_input, o_data), axis=1), axis=1)
        targets_a = np.diff(targets_s, axis=1)
        targets_s = targets_s[:, 1:].copy()
        target_sa = np.concatenate((targets_s, targets_a), axis=2)


        # nomarlize
        ext_data = self.feature_scaling(ext_data)
        target_xy = self.target_scaling(target_xy)
        target_sa = (target_sa - self.target_sa_mean) / self.target_sa_std

        # fill nan to 0
        ext_data = np.nan_to_num(ext_data)
        target_xy = np.nan_to_num(target_xy)
        target_sa = np.nan_to_num(target_sa)

        # make constant shapes
        ext_data = self.feat_pad(ext_data)
        target_xy = self.target_pad(target_xy)
        target_sa = self.target_pad(target_sa)

        # make static features start
        player_role = [player_role_to_idx[role] for role in i_info_df["player_role"]]
        player_role_onehot = np.identity(len(player_role_to_idx))[player_role]

        num_frame_output_feat = (num_frame_output - self.num_frame_output_mean) / self.num_frame_output_std

        player_last_xy = i_data[:, -1][:,[src_feat_idx["x"], src_feat_idx["y"]]].copy()
        player_last_xy = (player_last_xy - self.player_last_xy_mean) / self.player_last_xy_std

        passer_last_xy = passer_xy[-1]
        passer_last_xy = (passer_last_xy - self.passer_last_xy_mean) / self.passer_last_xy_std
        ball_land_xy = (ball_land_xy - self.ball_land_xy_mean) / self.ball_land_xy_std

        static_feat = np.zeros((MAX_PLAYER_NUM, 12), dtype=np.float32)
        static_feat[:len(player_role_onehot), :4] = player_role_onehot
        static_feat[:, 4] = num_frame_output_feat # # 全プレイヤーに同じ num_frame_output_feat をブロードキャストする設計
        static_feat[:, 5] = (frame_shift_size - 3.0) / 3.0
        static_feat[:, 6:8] = passer_last_xy # 全プレイヤーに同じ passer_xy をブロードキャストする設計
        static_feat[:, 8:10] = ball_land_xy
        static_feat[:len(player_last_xy), 10:12] = player_last_xy
        # make static features end

        # to Channel, Player, Length
        dynamic_feat = np.transpose(ext_data, (2, 0, 1))
        target_xy = np.transpose(target_xy, (2, 0, 1))  # 2pl
        target_sa = np.transpose(target_sa, (2, 0, 1))  # 2pl

        # make mask for loss func
        target_mask = np.zeros_like(target_xy)
        target_mask[:, :num_player_output, :num_frame_output] = 1.0


        dataset = {"dynamic_feat": torch.tensor(dynamic_feat).float(),
                   "static_feat": torch.tensor(static_feat).float(),
                   "target_xy": torch.tensor(target_xy).float(),
                   "target_sa": torch.tensor(target_sa).float(),
                   "target_mask": torch.tensor(target_mask).float(),
                   "num_frame_output": torch.tensor(num_frame_output).long(),
                   "num_player_output": torch.tensor(num_player_output).long(),
                   "num_player_input": torch.tensor(num_player_input).long(),
                   }

        return dataset


"""
T1 = TrainDataset(fold_df, data_dict, CFG, is_train = True)
self = T1
i=0
dataset = T1[i]
"""

# %%


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_loader(df, data_dict, CFG, i_fold=0):

    df1 = df.query("fold!=@i_fold").reset_index(drop=True)
    df2 = df.query("fold==@i_fold").reset_index(drop=True)

    generator = torch.Generator()
    generator.manual_seed(CFG.seed)


    T1 = TrainDataset(df1, data_dict, CFG, data_repeat_factor=CFG.train_data_repeat_factor, is_train=True)
    T2 = TrainDataset(df2, data_dict, CFG, is_train=False)


    I1 = DataLoader(T1, batch_size=CFG.batchsize1, num_workers=CFG.num_workers1, shuffle=True, drop_last=True,
                    worker_init_fn=seed_worker,
                    generator=generator,
                    persistent_workers=CFG.persistent_workers,
                    pin_memory=CFG.pin_memory)

    I2 = DataLoader(T2, batch_size=CFG.batchsize2, num_workers=CFG.num_workers2, shuffle=False, drop_last=False,
                    worker_init_fn=seed_worker,
                    generator=generator,
                    persistent_workers=CFG.persistent_workers,
                    pin_memory=CFG.pin_memory)
    return T1, I1, df1, T2, I2, df2

# %% [markdown]
# ## Model

# %% [code] {"execution":{"iopub.status.busy":"2026-01-20T09:49:42.510570Z","iopub.execute_input":"2026-01-20T09:49:42.510916Z","iopub.status.idle":"2026-01-20T09:49:42.542567Z","shell.execute_reply.started":"2026-01-20T09:49:42.510890Z","shell.execute_reply":"2026-01-20T09:49:42.541869Z"}}
class Block(nn.Module):
    def __init__(self, dim, kernel_size=3, groups=1, padding=0, actf=nn.SiLU()):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            actf,
            nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=groups, padding=padding, bias=False),
            nn.BatchNorm1d(dim),
            actf,
            nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=groups, padding=padding, bias=False),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, groups=1, actf=nn.SiLU()):
        padding = (kernel_size - 1) // 2
        super().__init__()
        self.block = Block(
            dim=dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
            actf=actf
        )

    def forward(self, x):
        return x + self.block(x)


class SequenceStem(nn.Module):
    def __init__(self, input_dim, nbai, kernel_size=3):
        super().__init__()
        dim = input_dim * nbai

        # Depthwise convolution
        self.stem = nn.Conv1d(
            input_dim, dim,
            kernel_size=kernel_size,
            groups=input_dim,
            bias=False
        )

        # Three residual blocks (with different group settings)
        self.blocks = nn.ModuleList([
            Block(dim, kernel_size, padding=0, groups=input_dim),
            Block(dim, kernel_size, padding=0, groups=input_dim),
            Block(dim, kernel_size, padding=0, groups=input_dim // 2),
        ])

        # No padding is used in the convolution,
        # so with kernel_size=3 the output length is reduced by 2
        # (this is related to x[:, :, 4:] in the forward pass)
        self.crop = (kernel_size - 1) * 2  # e.g., 3 → 4

    def forward(self, x):
        x = self.stem(x)

        # Residual blocks (add after cropping)
        for block in self.blocks:
            x = x[:, :, self.crop:] + block(x)

        return x


class MyTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 batch_first=True,
                 norm_first=True):

        super().__init__()

        self.batch_first = batch_first
        self.norm_first = norm_first

        # --- Multi-Head Attention ---
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
        )

        self.linear1_w1 = nn.Linear(d_model, dim_feedforward)
        self.linear1_w2 = nn.Linear(d_model, dim_feedforward)

        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # --- LayerNorm ---
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # --- Dropout ---
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _ffn_swiglu(self, x):
        """
        SwiGLU(x) = (W1 x) * silu(W2 x)
        """
        a = self.linear1_w1(x)
        b = F.silu(self.linear1_w2(x))
        x = a * b
        return self.linear2(self.dropout(x))

    def forward(self, src, src_mask=None, src_key_padding_mask=None):

        # ============================
        #      PRE-NORM
        # ============================
        if self.norm_first:
            # --- LayerNorm → MHA → res ---
            src2 = self.norm1(src)
            attn_out, _ = self.self_attn(
                src2, src2, src2,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=False,
            )
            src = src + self.dropout1(attn_out)

            # --- LayerNorm → SwiGLU FFN → res ---
            src2 = self.norm2(src)
            src2 = self._ffn_swiglu(src2)
            src = src + self.dropout2(src2)

        # ============================
        #      POST-NORM
        # ============================
        else:
            # --- MHA → res → LayerNorm ---
            attn_out, _ = self.self_attn(
                src, src, src,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=False,
            )
            src = self.norm1(src + self.dropout1(attn_out))

            # --- SwiGLU FFN → res → LayerNorm ---
            src2 = self._ffn_swiglu(src)
            src = self.norm2(src + self.dropout2(src2))

        return src




class MyTransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return x


class Head(nn.Module):
    def __init__(self, dim2, aux_target_dim, kernel_size=3):
        super().__init__()
        self.dim2 = dim2
        self.aux_target_dim = aux_target_dim

        self.model = nn.Sequential(
            ResBlock(self.dim2, kernel_size=kernel_size),
            ResBlock(self.dim2, kernel_size=kernel_size),
        )
        self.head_mu = nn.Sequential(
            nn.Conv1d(self.dim2, 2, kernel_size=1)
        )
        self.head_var = nn.Sequential(
            nn.Conv1d(self.dim2, 2, kernel_size=1)
        )
        self.head_aux_mu = nn.Sequential(
            nn.Conv1d(self.dim2, self.aux_target_dim, kernel_size=1)
        )
        self.head_aux_var = nn.Sequential(
            nn.Conv1d(self.dim2, self.aux_target_dim, kernel_size=1)
        )

    def __reshape_permute(self, x, bsize, num_player):
        # b*p,c,l -> b,c,p,l
        _, C, L = x.shape
        x = torch.permute(torch.reshape(x, (bsize, num_player, C, L)), (0, 2, 1, 3))  # b 2 p l
        return x

    def forward(self, z, bsize, num_player):

        z = self.model(z)  # bp c L
        pred_m = self.head_mu(z)  # bp 2 L
        pred_v = self.head_var(z)  # bp 2 L
        auxi_m = self.head_aux_mu(z)
        auxi_v = self.head_aux_var(z)

        pred_m = self.__reshape_permute(pred_m, bsize, num_player)
        pred_v = self.__reshape_permute(pred_v, bsize, num_player)
        auxi_m = self.__reshape_permute(auxi_m, bsize, num_player)
        auxi_v = self.__reshape_permute(auxi_v, bsize, num_player)

        return pred_m, pred_v, auxi_m, auxi_v


class Model(nn.Module):
    def __init__(self, input_dim, output_sequence_size, loss_fn, model_param):
        super().__init__()
        self.dim1 = model_param.dim1
        self.dim2 = model_param.dim2
        self.input_dim_multiplier = model_param.input_dim_multiplier
        self.tfm_nhead = model_param.tfm_nhead
        self.dim_feedforward = model_param.dim_feedforward
        self.tfm_dropout = model_param.tfm_dropout
        self.num_layers = model_param.num_layers
        self.dropout0 = model_param.dropout0
        self.dropout1 = model_param.dropout1
        self.dropout2 = model_param.dropout2
        self.aux_loss_weight = model_param.aux_loss_weight
        self.aux_target_dim = model_param.aux_target_dim
        self.static_feature_dim = model_param.static_feature_dim


        dim0 = input_dim * self.input_dim_multiplier + self.input_dim_multiplier

        self.sequence_stem = SequenceStem(input_dim, self.input_dim_multiplier)
        self.feature_stem = nn.Sequential(nn.Conv1d(self.static_feature_dim, self.input_dim_multiplier, kernel_size=1, bias=False),
                                          nn.BatchNorm1d(self.input_dim_multiplier),
                                          nn.SiLU(),
                                          )


        self.merge_layer = nn.Sequential(
            nn.Conv1d(dim0, self.dim1, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.dim1),
            ResBlock(self.dim1, kernel_size=1),
            ResBlock(self.dim1, kernel_size=1),
            nn.BatchNorm1d(self.dim1),
        )

        encoder_layer = MyTransformerEncoderLayer(d_model=self.dim1, nhead=self.tfm_nhead,
                                                  dim_feedforward=self.dim_feedforward,
                                                  dropout=self.tfm_dropout, batch_first=True)
        self.encoder = MyTransformerEncoder(encoder_layer, num_layers=self.num_layers)



        self.pred_sequence_layer = nn.Sequential(
            nn.Conv1d(self.dim1, output_sequence_size * self.dim2, kernel_size=1)
        )

        self.head = Head(self.dim2, self.aux_target_dim)
        self.output_sequence_size = output_sequence_size
        self.loss_fn = loss_fn
        self.min_var = model_param.min_var


    def __calc_loss__(self, pred_m, pred_v, auxi_m, auxi_v, target_xy, target_sa, target_mask):

        pred_loss = self.loss_fn(pred_m, target_xy, pred_v)
        pred_loss = (pred_loss * target_mask).sum() / target_mask.sum()


        aux_loss = self.loss_fn(auxi_m, target_sa, auxi_v)
        aux_loss = aux_loss * (2.0 / aux_loss.shape[1])  # Scale to keep the balance with the main loss
                                                         # even if the number of auxiliary losses increases
        aux_loss = (aux_loss * target_mask[:, :1]).sum() / target_mask.sum()

        loss = (1.0 - self.aux_loss_weight) * pred_loss + self.aux_loss_weight * aux_loss

        return loss

    def __proc(self, data):
        # self = model
                # self = model
        x_dynamic = data["dynamic_feat"]  # b c p l
        x_static = data["static_feat"]  # b p c

        bsize, c, num_player, l = x_dynamic.shape
        x_dynamic = torch.permute(x_dynamic, (0, 2, 1, 3))  # b p c l
        x_dynamic = torch.reshape(x_dynamic, (bsize * num_player, c, l))  # bp c l

        x_static = torch.reshape(x_static, (bsize * num_player, -1, 1))  # b p c -> bp c 1

        x_dynamic = self.sequence_stem(x_dynamic)
        x_static = self.feature_stem(x_static)

        # crop last frame
        x_dynamic = x_dynamic[:, :, -1:]  # bp c 1
        x = torch.concat((x_dynamic, x_static), dim=1)
        x = self.merge_layer(x)  # bp c 1
        x = x.reshape(bsize, num_player, self.dim1)  # b num_player c

        key_padding_mask = torch.arange(num_player, device=x.device).expand(bsize, num_player) >= data["num_player_input"].unsqueeze(1).to(device=x.device)  # shape (B, L)
        x = x.contiguous()
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = x.reshape(bsize * num_player, self.dim1, 1)  # b p c

        x = self.pred_sequence_layer(x)  # bp C 1
        z = x.reshape(-1, self.dim2, self.output_sequence_size)  # bp c L

        pred_m, pred_v, auxi_m, auxi_var = self.head(z, bsize, num_player)
        return pred_m, pred_v, auxi_m, auxi_var


    def forward(self, data, calc_loss=True, return_pred=True):
        # self = model

        pred_m, pred_v, auxi_m, auxi_var = self.__proc(data)
        
        output_dict = {}
        if calc_loss:
            target_xy = data["target_xy"]  # b c p l
            target_sa = data["target_sa"]  # b c p l
            target_mask = data["target_mask"]  # b c p l

            pred_v = torch.nn.Softplus()(pred_v) + self.min_var
            auxi_var = torch.nn.Softplus()(auxi_var) + self.min_var
            loss = self.__calc_loss__(pred_m, pred_v, auxi_m, auxi_var, target_xy, target_sa, target_mask)
            output_dict["loss"] = loss

        # Collect per-player predictions with the original output frame length
        preds = []
        if return_pred:
            bsize = len(pred_m)
            for i in range(bsize):
                # Retrieve the number of output frames and players for this sample
                num_frame_output = data["num_frame_output"][i]
                num_player_output = data["num_player_output"][i]
                for j in range(num_player_output):
                    # Extract valid frames only, detach from the graph, and move to CPU
                    pred = pred_m[i, :, j, :num_frame_output].detach().cpu().numpy()
                    preds.append(pred)
        # Store collected predictions in the output dictionary
        output_dict["preds"] = preds
        return output_dict

    def inference(self, data):
        pred_m, pred_v, auxi_m, auxi_var = self.__proc(data)
        return pred_m

# %% [markdown]
# ## train and validation functions

# %% [code] {"execution":{"iopub.status.busy":"2026-01-20T09:49:42.543299Z","iopub.execute_input":"2026-01-20T09:49:42.543609Z","iopub.status.idle":"2026-01-20T09:49:42.557605Z","shell.execute_reply.started":"2026-01-20T09:49:42.543576Z","shell.execute_reply":"2026-01-20T09:49:42.556643Z"}}
def train_epoch(
    model,
    loader,
    optimizer,
    epoch,
    scheduler=None,
    ema_model=None,
    device="cuda",
    grad_max_norm=None,
    grad_max_value=None,
):

    model.train()
    model.to(device)
    if ema_model is not None:
        ema_model.to(device)

    device_type = "cpu"
    if "cuda" in device:
        device_type = "cuda"
    grad_scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)

    amp_dtype_ = torch.float16


    batch_num = len(loader)
    mean_loss = 0
    n = 0
    losses = []
    for batch_idx, data in enumerate((loader)):

        data = {k: v.to(device) for k, v in data.items()}

        optimizer.zero_grad()
        with torch.autocast(device_type=device_type, dtype=amp_dtype_):
            output_dict = model(data, return_pred=False)
        loss = output_dict["loss"]
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_max_norm)

        if grad_norm >= grad_max_norm:
            print(f"{epoch} {batch_idx} clip_grad_norm_{grad_norm}")
        grad_scaler.step(optimizer)
        grad_scaler.update()

        if ema_model is not None:
            ema_model.update(model)

        loss = loss.detach().cpu().numpy()
        losses.append(loss)
        num = data["num_frame_output"].sum().detach().cpu().item() * 2  # xyなので2倍
        mean_loss += loss * num
        n += num

        iteration = epoch * batch_num + batch_idx
        if scheduler is not None:
            scheduler.step_update(iteration)

    mean_loss = mean_loss / n
    return mean_loss


def valid_epoch(
    model,
    loader,
    act_func=nn.Identity(),
    device="cuda:0",
    use_tqdm=False,
):

    model.eval()
    model.to(device)
    mean_loss = 0
    n = 0

    preds = []
    for data in (loader):
        data = {k: v.to(device) for k, v in data.items()}

        with torch.inference_mode():
            output_dict = model(data, calc_loss=True, return_pred=True)

        loss = output_dict["loss"]
        loss = loss.detach().float().cpu().numpy()

        num = data["num_frame_output"].sum().detach().cpu().item() * 2
        mean_loss += loss * num
        n += num

        for pred in output_dict["preds"]:
            preds.append(pred)
    mean_loss = mean_loss / n
    preds = np.concatenate(preds, axis=1).T


    return mean_loss, preds

# %% [markdown]
# ## Configuration

# %% [code] {"execution":{"iopub.status.busy":"2026-01-20T09:49:42.560032Z","iopub.execute_input":"2026-01-20T09:49:42.560286Z","iopub.status.idle":"2026-01-20T09:49:42.573288Z","shell.execute_reply.started":"2026-01-20T09:49:42.560256Z","shell.execute_reply":"2026-01-20T09:49:42.572778Z"}}
# %%
class DictDotNotation(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


CFG = DictDotNotation(
    batchsize1=64,
    batchsize2=64,
    num_workers1=2,
    num_workers2=2,
    epoch=35,
    train_data_repeat_factor=6,
    start_valid_epoch=8,
    n_fold=5,
    seed=0,
    device="cuda:0",

    # amp1=True,
    # amp1_dtype="float16",
    # amp2=False,

    input_sequence_size=20,
    output_sequence_size=48,

    model_param=DictDotNotation(
        dim1=256,
        dim2=32,
        dropout0=0.0,
        dropout1=0.0,
        dropout2=0.0,
        input_dim_multiplier=64,
        # register_num=0,
        # register_sigma=1.0,
        tfm_nhead=8,
        tfm_dropout=0.05,
        dim_feedforward=1024,
        num_layers=3,
        min_var=0.001,
        aux_loss_weight=0.8,
        aux_target_dim=4,
        static_feature_dim=12,
    ),


    # optimizer  params
    lr=0.0005,
    weight_decay=1e-4,
    grad_max_norm=200.0,
    grad_max_value=None,
    ema_decay=0.9995,
    ema_update_after_epoch=5,

    # data augumentation params
    angle_std_rate=0.0,
    angle_uni_rate=0.5,
    angle_std=1.0,
    x_bias_std=1.0,
    y_bias_std=1.5,

    flipud_rate=0.5,
    max_shift_size=20,
    frame_shift_rate=0.675,

    frame_scale_rate=0.0,
    frame_scale_std=0.0,

    persistent_workers=False,
    pin_memory=True,
    save_model=True,

    target_names=["dx",
                  "dy",
                  ],
    fold_seeds = (0,),  # If computational resources allow, it is better to evaluate with (0, 1, 2)
)


# %%

# %% [markdown]
# ## Select dynamic features

# %% [code] {"execution":{"iopub.status.busy":"2026-01-20T09:49:42.573903Z","iopub.execute_input":"2026-01-20T09:49:42.574231Z","iopub.status.idle":"2026-01-20T09:49:42.592568Z","shell.execute_reply.started":"2026-01-20T09:49:42.574206Z","shell.execute_reply":"2026-01-20T09:49:42.592026Z"}}

# select dynamic features here

# Best feature set(base)
# Private: 0.46471 wo tta, 5fold seed emsenble
CFG.ext_feat_names=["x", "y",
                    "ori_sin", "ori_cos",
                    "ball_land_x_dx", "ball_land_y_dy",  # Distance to the ball landing position
                    "receiver_dx", "receiver_dy",        # Distance to the receiver
                    "dir_sin_s", "dir_cos_s",            # Velocity vector
                  # "rot_x", "rot_y",                    # pass-aligned coordinate
                ]
model_name = "base"


# Base features excluding ball and receiver coordinates
# Private: 0.46511 wo tta, 5fold seed emsenble
"""
CFG.ext_feat_names=["x", "y",
                    "ori_sin", "ori_cos",
                    # "ball_land_x_dx", "ball_land_y_dy",  # Distance to the ball landing position
                    # "receiver_dx", "receiver_dy",        # Distance to the receiver
                    "dir_sin_s", "dir_cos_s",              # Velocity vector
                    # "rot_x", "rot_y",                    # pass-aligned coordinate
                ]
model_name = "norb"
"""


# Base features with pass-aligned coordinates added
# Private: 0.46946 wo tta, 5fold seed emsenble
"""
CFG.ext_feat_names=["x", "y",
                    "ori_sin", "ori_cos",
                    "ball_land_x_dx", "ball_land_y_dy",  # Distance to the ball landing position
                    "receiver_dx", "receiver_dy",        # Distance to the receiver
                    "dir_sin_s", "dir_cos_s",            # Velocity vector
                    "rot_x", "rot_y",                    # pass-aligned coordinate
                ]
model_name = "rot"

"""


# %% [code] {"execution":{"iopub.status.busy":"2026-01-20T09:49:42.593499Z","iopub.execute_input":"2026-01-20T09:49:42.593753Z","iopub.status.idle":"2026-01-20T09:49:42.606262Z","shell.execute_reply.started":"2026-01-20T09:49:42.593733Z","shell.execute_reply":"2026-01-20T09:49:42.605636Z"}}
TRAIN_PATH = pathlib.Path("/kaggle/input/nfl-big-data-bowl-2026-prediction/train")
WORK_PATH = pathlib.Path("/kaggle/working/")
best_model_dir = WORK_PATH / f"models_{model_name}" 
result_dir = WORK_PATH / "results"

os.makedirs(best_model_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)
print(best_model_dir)

# %% [code] {"execution":{"iopub.status.busy":"2026-01-20T09:49:42.607251Z","iopub.execute_input":"2026-01-20T09:49:42.607929Z","iopub.status.idle":"2026-01-20T09:51:11.570271Z","shell.execute_reply.started":"2026-01-20T09:49:42.607897Z","shell.execute_reply":"2026-01-20T09:51:11.569547Z"}}
#%%
input_df, output_df = read_csvs(TRAIN_PATH)
fold_df = make_fold_df(input_df, CFG.n_fold)
data_dict = make_data_dict(input_df, output_df)


# %% [code] {"execution":{"iopub.status.busy":"2026-01-20T09:51:11.571211Z","iopub.execute_input":"2026-01-20T09:51:11.571670Z","iopub.status.idle":"2026-01-20T09:53:30.553799Z","shell.execute_reply.started":"2026-01-20T09:51:11.571644Z","shell.execute_reply":"2026-01-20T09:53:30.552784Z"}}
valid_dfs = pd.DataFrame()
for seed in CFG.fold_seeds:
    fold_df["fold"] = fold_df[f"fold{seed}"]
    for i_fold in range(CFG.n_fold):

        fix_seed(CFG.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        T1, I1, df1, T2, I2, df2 = get_loader(fold_df, data_dict, CFG, i_fold)

        model = Model(
            input_dim=len(CFG.ext_feat_names),
            output_sequence_size=CFG.output_sequence_size,
            loss_fn=torch.nn.GaussianNLLLoss(eps=0.0, reduction='none'),
            model_param=CFG.model_param,
        )

        """
        self = model
        data = next(iter(I1))
        output = model(data)
        """

        model = torch.compile(model)
        optimizer = torch.optim.RAdam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, decoupled_weight_decay=True)

        best_score = 1e6
        model_ema = None
        for epoch in tqdm(range(CFG.epoch)):
            if epoch == CFG.ema_update_after_epoch:
                model_ema = ModelEmaV3(model, CFG.ema_decay, device=CFG.device)
                print("start ema trainning!!")

            tr_loss = train_epoch(model, I1, optimizer, epoch, ema_model=model_ema, scheduler=None,
                                  device=CFG.device, grad_max_norm=CFG.grad_max_norm)
            if epoch < CFG.start_valid_epoch:
                continue
            vl_loss, preds = valid_epoch(model_ema.module, I2, device=CFG.device)

            valid_df = T2.valid_df.copy()
            preds_xy = T2.target_rescaling(preds) # denormalize
            preds_xy = preds_xy + valid_df[["x_last", "y_last"]].values # add last input frame coordinate

            valid_df["seed"] = seed
            valid_df["i_fold"] = i_fold
            valid_df["epoch"] = epoch
            valid_df["mean_loss"] = vl_loss
            valid_df[["pred_x", "pred_y"]] = preds_xy
            score = calc_score(valid_df)
            wandb_log = {}
            wandb_log["epoch"] = epoch
            wandb_log[f"tr_loss/s{seed}_f{i_fold}"] = float(tr_loss)
            wandb_log[f"loss_val/s{seed}_f{i_fold}"] = float(vl_loss)
            wandb_log[f"score_val/s{seed}_f{i_fold}"] = score
            print(json.dumps(wandb_log, indent=4, ensure_ascii=False))
            # wandb.log(wandb_log)
            valid_dfs = pd.concat((valid_dfs, valid_df))

            if score < best_score:
                best_score = score
                best_model_path = best_model_dir / f"{seed}_{i_fold}_best.pt"
                model.eval()
                torch.save(model_ema.module._orig_mod.state_dict(), best_model_path)

    for epoch in range(CFG.start_valid_epoch, CFG.epoch):
        valid_df = valid_dfs.query("seed==@seed and epoch==@epoch").reset_index(drop=True)
        score = calc_score(valid_df)
        mean_loss = valid_df["mean_loss"].mean()
        wandb_log = {}
        wandb_log["epoch"] = epoch
        wandb_log[f"score_val/s{seed}_all"] = score
        wandb_log[f"loss_val/s{seed}_all"] = float(mean_loss)
        print(json.dumps(wandb_log, indent=4, ensure_ascii=False))
        # wandb.log(wandb_log)

# %% [markdown]
# ## Summarize results

# %% [code] {"execution":{"iopub.status.busy":"2026-01-20T09:53:30.554723Z","iopub.status.idle":"2026-01-20T09:53:30.554990Z","shell.execute_reply.started":"2026-01-20T09:53:30.554865Z","shell.execute_reply":"2026-01-20T09:53:30.554885Z"}}
# Compute scores for all folds at each epoch
score_dicts = []
for epoch in range(CFG.start_valid_epoch, CFG.epoch):
    valid_df = valid_dfs.query("epoch==@epoch").reset_index(drop=True)

    score = calc_score(valid_df)
    mean_loss = valid_df["mean_loss"].mean()
    wandb_log = {}
    wandb_log["epoch"] = epoch
    wandb_log["score_val/all"] = score
    wandb_log["loss_val/all"] = mean_loss
    # wandb.log(wandb_log)
    score_dicts.append(wandb_log)
score_df = pd.DataFrame(score_dicts)
score_df.to_csv(result_dir / "score_df.csv", index=False)
print(score_df)


best_dfs = []
for seed in CFG.fold_seeds:
    for i_fold in range(CFG.n_fold):
        best_score = 1e6
        for epoch in range(CFG.start_valid_epoch, CFG.epoch):
            valid_df = valid_dfs.query("seed==@seed and epoch==@epoch and i_fold==@i_fold").reset_index(drop=True)
            score = calc_score(valid_df)
            # print(i_fold,epoch,score)
            if score < best_score:
                best_score = score
                best_df = valid_df.copy()
        best_dfs.append(best_df)
best_dfs = pd.concat(best_dfs)
best_score = calc_score(best_dfs)
print(f"{best_score=}")
# best_dfs.to_parquet(os.path.join(result_path, "best_dfs.parquet"), index=False)
wandb_log["best_score"] = best_score
# wandb.log(wandb_log)
# wandb.finish()

# %% [code]
