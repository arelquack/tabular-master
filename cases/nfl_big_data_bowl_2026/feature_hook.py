import pandas as pd
import numpy as np
from tabular_pipeline import BaseFeatureHook

class CustomFeatures(BaseFeatureHook):
    """
    Spatio-temporal kinematic feature engineering hook for NFL trajectory prediction.
    Extracts velocity vectors, acceleration projections, and ball attraction metrics.
    """
    def transform(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None):
        dfs = [train_df] if test_df is None else [train_df, test_df]

        for df in dfs:
            # 1. Temporal Step (t in seconds, frame rate is 10 Hz -> 0.1s per frame)
            dt = df['future_frame_id'] * 0.1
            df['dt'] = dt.astype('float32')

            # 2. Polar to Cartesian Velocity & Acceleration Vectors
            dir_rad = np.radians(df['dir_0'])
            o_rad = np.radians(df['o_0'])
            
            vx_0 = df['s_0'] * np.sin(dir_rad)
            vy_0 = df['s_0'] * np.cos(dir_rad)
            ax_0 = df['a_0'] * np.sin(dir_rad)
            ay_0 = df['a_0'] * np.cos(dir_rad)

            df['vx_0'] = vx_0.astype('float32')
            df['vy_0'] = vy_0.astype('float32')
            df['ax_0'] = ax_0.astype('float32')
            df['ay_0'] = ay_0.astype('float32')

            # 3. Kinematic Position Projection (Newtonian Baseline: x_0 + v*t + 0.5*a*t^2)
            kin_x = df['x_0'] + vx_0 * dt + 0.5 * ax_0 * (dt ** 2)
            kin_y = df['y_0'] + vy_0 * dt + 0.5 * ay_0 * (dt ** 2)

            df['kinematic_x'] = kin_x.astype('float32')
            df['kinematic_y'] = kin_y.astype('float32')

            # 4. Ball Attraction Vector (Vectors to anticipated ball landing spot)
            dx_to_ball = df['ball_land_x'] - df['x_0']
            dy_to_ball = df['ball_land_y'] - df['y_0']
            dist_to_ball = np.sqrt(dx_to_ball ** 2 + dy_to_ball ** 2)

            df['dx_to_ball'] = dx_to_ball.astype('float32')
            df['dy_to_ball'] = dy_to_ball.astype('float32')
            df['dist_to_ball'] = dist_to_ball.astype('float32')

            # 5. Heading vs Motion Angle Difference
            angle_diff = np.abs(df['o_0'] - df['dir_0'])
            df['orientation_dir_diff'] = np.minimum(angle_diff, 360 - angle_diff).astype('float32')

            # 6. Direction to Ball vs Velocity Heading
            angle_to_ball_rad = np.arctan2(dx_to_ball, dy_to_ball)
            angle_to_ball_deg = np.degrees(angle_to_ball_rad) % 360
            ball_dir_diff = np.abs(angle_to_ball_deg - df['dir_0'])
            df['ball_heading_diff'] = np.minimum(ball_dir_diff, 360 - ball_dir_diff).astype('float32')

            # 7. Speed Required to Reach Ball at Event Horizon
            time_to_ball_estimate = dist_to_ball / (df['s_0'] + 1e-3)
            df['time_to_ball_est'] = time_to_ball_estimate.clip(upper=10).astype('float32')

        return train_df, test_df
