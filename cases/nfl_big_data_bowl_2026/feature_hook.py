import numpy as np
import pandas as pd
from tabular_pipeline import BaseFeatureHook

FIELD_WIDTH = 120.0
FIELD_HEIGHT = 53.33

class CustomFeatures(BaseFeatureHook):
    def transform(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None):
        for df in ([train_df, test_df] if test_df is not None else [train_df]):
            # 1. Normalisasi arah serangan (semua menyerang ke kanan)
            if 'play_direction' in df.columns:
                left = df['play_direction'] == 'left'
                for col in ['x_feature', 'ball_land_x']:
                    if col in df.columns:
                        df.loc[left, col] = FIELD_WIDTH - df.loc[left, col]
                for col in ['y_feature', 'ball_land_y']:
                    if col in df.columns:
                        df.loc[left, col] = FIELD_HEIGHT - df.loc[left, col]
                for col in ['dir', 'o']:
                    if col in df.columns:
                        df.loc[left, col] = (df.loc[left, col] + 180.0) % 360.0

            # 2. Vektor kecepatan (vx, vy)
            rad = np.radians(df.get('dir', 0.0).fillna(0))
            s = df.get('s', 0.0).fillna(0)
            df['vx'] = s * np.sin(rad)
            df['vy'] = s * np.cos(rad)

            # 3. Posisi inersia hukum gerak Newton (x_last + v * t)
            t = df.get('frame_id', 1).fillna(1) * 0.1
            df['t'] = t
            x_ref = df.get('x_feature', 0.0)
            y_ref = df.get('y_feature', 0.0)
            df['x_inertial'] = x_ref + df['vx'] * t
            df['y_inertial'] = y_ref + df['vy'] * t

            # 4. Vektor jarak dan estimasi waktu ke titik jatuh bola
            df['dist_ball_x'] = df.get('ball_land_x', 0.0) - x_ref
            df['dist_ball_y'] = df.get('ball_land_y', 0.0) - y_ref
            df['dist_ball'] = np.sqrt(df['dist_ball_x']**2 + df['dist_ball_y']**2)
            df['time_to_ball'] = df['dist_ball'] / np.maximum(s, 0.1)

        return train_df, test_df