import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values('date').reset_index(drop=True)
    
    def _calc_team_stats(self, games, team_id):
        """Helper to calculate stats from a set of games"""
        if len(games) == 0:
            return None
            
        wins = 0
        points_for = []
        points_against = []
        home_games = 0
        home_wins = 0
        away_games = 0
        away_wins = 0
        win_streak = 0
        current_streak = 0
        last_5_wins = 0
        
        for idx, (_, game) in enumerate(games.iterrows()):
            is_home = game['home_team_id'] == team_id
            won = game['home_winner'] if is_home else game['away_winner']
            
            # Overall stats
            if won:
                wins += 1
                current_streak = current_streak + 1 if current_streak >= 0 else 1
            else:
                current_streak = current_streak - 1 if current_streak <= 0 else -1
            
            # Track last 5 games
            if idx >= len(games) - 5 and won:
                last_5_wins += 1
            
            # Win streak
            win_streak = max(win_streak, current_streak)
            
            # Home/Away splits
            if is_home:
                home_games += 1
                if won:
                    home_wins += 1
                points_for.append(game['home_score'])
                points_against.append(game['away_score'])
            else:
                away_games += 1
                if won:
                    away_wins += 1
                points_for.append(game['away_score'])
                points_against.append(game['home_score'])
        
        # Calculate advanced stats
        avg_points = np.mean(points_for)
        avg_points_against = np.mean(points_against)
        
        return {
            'win_pct': wins / len(games),
            'avg_points': avg_points,
            'avg_points_against': avg_points_against,
            'point_diff': avg_points - avg_points_against,
            'std_points': np.std(points_for),  # Consistency
            'std_points_against': np.std(points_against),
            'max_points': np.max(points_for),
            'min_points': np.min(points_for),
            'home_win_pct': home_wins / home_games if home_games > 0 else 0.5,
            'away_win_pct': away_wins / away_games if away_games > 0 else 0.5,
            'win_streak': current_streak,
            'max_win_streak': win_streak,
            'last_5_win_pct': last_5_wins / min(5, len(games)),
            'offensive_efficiency': avg_points,  # Points per game
            'defensive_efficiency': avg_points_against,  # Points allowed per game
        }
    
    def add_team_features(self, window_sizes=[5, 10, 20]):
        """Add rolling statistics for all teams - NO LEAKAGE"""
        print("Calculating team features (ensuring no leakage)...")
        
        for window in window_sizes:
            print(f"  Window: {window} games")
            
            # Initialize columns for all possible features
            feature_names = [
                'win_pct', 'avg_points', 'avg_points_against', 'point_diff',
                'std_points', 'std_points_against', 'max_points', 'min_points',
                'home_win_pct', 'away_win_pct', 'win_streak', 'max_win_streak',
                'last_5_win_pct', 'offensive_efficiency', 'defensive_efficiency'
            ]
            
            for team_type in ['home', 'away']:
                for feat in feature_names:
                    self.df[f'{team_type}_{feat}_{window}'] = np.nan
            
            # Calculate for each game
            for idx in range(len(self.df)):
                game = self.df.iloc[idx]
                
                # CRITICAL: Only look at games BEFORE this one
                prev_games = self.df.iloc[:idx]
                
                # Home team stats
                home_prev = prev_games[
                    (prev_games['home_team_id'] == game['home_team_id']) | 
                    (prev_games['away_team_id'] == game['home_team_id'])
                ].tail(window)
                
                if len(home_prev) >= 3:
                    home_stats = self._calc_team_stats(home_prev, game['home_team_id'])
                    if home_stats:
                        for key, val in home_stats.items():
                            self.df.at[idx, f'home_{key}_{window}'] = val
                
                # Away team stats
                away_prev = prev_games[
                    (prev_games['home_team_id'] == game['away_team_id']) | 
                    (prev_games['away_team_id'] == game['away_team_id'])
                ].tail(window)
                
                if len(away_prev) >= 3:
                    away_stats = self._calc_team_stats(away_prev, game['away_team_id'])
                    if away_stats:
                        for key, val in away_stats.items():
                            self.df.at[idx, f'away_{key}_{window}'] = val
                
                if idx % 100 == 0:
                    print(f"    Processed {idx}/{len(self.df)} games")
        
        return self.df
    
    def add_rest_days(self):
        """Add days of rest between games"""
        print("Calculating rest days...")
        
        self.df['home_rest_days'] = np.nan
        self.df['away_rest_days'] = np.nan
        self.df['rest_advantage'] = np.nan  # Difference in rest
        
        for idx in range(len(self.df)):
            game = self.df.iloc[idx]
            prev_games = self.df.iloc[:idx]
            
            # Home team's last game
            home_prev = prev_games[
                (prev_games['home_team_id'] == game['home_team_id']) | 
                (prev_games['away_team_id'] == game['home_team_id'])
            ]
            
            home_rest = np.nan
            if len(home_prev) > 0:
                last_game_date = home_prev.iloc[-1]['date']
                home_rest = (game['date'] - last_game_date).days
                self.df.at[idx, 'home_rest_days'] = home_rest
            
            # Away team's last game
            away_prev = prev_games[
                (prev_games['home_team_id'] == game['away_team_id']) | 
                (prev_games['away_team_id'] == game['away_team_id'])
            ]
            
            away_rest = np.nan
            if len(away_prev) > 0:
                last_game_date = away_prev.iloc[-1]['date']
                away_rest = (game['date'] - last_game_date).days
                self.df.at[idx, 'away_rest_days'] = away_rest
            
            # Rest advantage
            if not pd.isna(home_rest) and not pd.isna(away_rest):
                self.df.at[idx, 'rest_advantage'] = home_rest - away_rest
        
        return self.df
    
    def add_head_to_head(self, max_h2h_games=5):
        """Add head-to-head historical performance"""
        print("Calculating head-to-head history...")
        
        # Initialize with float type
        self.df['h2h_home_wins'] = 0.0
        self.df['h2h_away_wins'] = 0.0
        self.df['h2h_home_avg_margin'] = 0.0
        
        for idx in range(len(self.df)):
            game = self.df.iloc[idx]
            prev_games = self.df.iloc[:idx]
            
            # Find previous matchups between these teams
            h2h = prev_games[
                ((prev_games['home_team_id'] == game['home_team_id']) & 
                 (prev_games['away_team_id'] == game['away_team_id'])) |
                ((prev_games['home_team_id'] == game['away_team_id']) & 
                 (prev_games['away_team_id'] == game['home_team_id']))
            ].tail(max_h2h_games)
            
            if len(h2h) > 0:
                home_wins = 0
                margins = []
                
                for _, h2h_game in h2h.iterrows():
                    # Check if current home team won in that game
                    if h2h_game['home_team_id'] == game['home_team_id']:
                        if h2h_game['home_winner']:
                            home_wins += 1
                            margins.append(float(h2h_game['home_score'] - h2h_game['away_score']))
                        else:
                            margins.append(float(h2h_game['home_score'] - h2h_game['away_score']))
                    else:
                        if h2h_game['away_winner']:
                            home_wins += 1
                            margins.append(float(h2h_game['away_score'] - h2h_game['home_score']))
                        else:
                            margins.append(float(h2h_game['away_score'] - h2h_game['home_score']))
                
                self.df.at[idx, 'h2h_home_wins'] = float(home_wins)
                self.df.at[idx, 'h2h_away_wins'] = float(len(h2h) - home_wins)
                self.df.at[idx, 'h2h_home_avg_margin'] = float(np.mean(margins)) if margins else 0.0
            
            if idx % 500 == 0:
                print(f"    Processed {idx}/{len(self.df)} games")
        
        return self.df
    
    def add_momentum_features(self):
        """Add momentum and trend features"""
        print("Calculating momentum features...")
        
        for window in [5, 10]:
            self.df[f'home_momentum_{window}'] = np.nan
            self.df[f'away_momentum_{window}'] = np.nan
            
            for idx in range(len(self.df)):
                game = self.df.iloc[idx]
                prev_games = self.df.iloc[:idx]
                
                # Home team momentum (weighted recent performance)
                home_prev = prev_games[
                    (prev_games['home_team_id'] == game['home_team_id']) | 
                    (prev_games['away_team_id'] == game['home_team_id'])
                ].tail(window)
                
                if len(home_prev) >= 3:
                    weights = np.linspace(0.5, 1.0, len(home_prev))  # Recent games weighted more
                    wins = []
                    for _, prev_game in home_prev.iterrows():
                        is_home = prev_game['home_team_id'] == game['home_team_id']
                        wins.append(1 if (prev_game['home_winner'] if is_home else prev_game['away_winner']) else 0)
                    momentum = np.average(wins, weights=weights)
                    self.df.at[idx, f'home_momentum_{window}'] = momentum
                
                # Away team momentum
                away_prev = prev_games[
                    (prev_games['home_team_id'] == game['away_team_id']) | 
                    (prev_games['away_team_id'] == game['away_team_id'])
                ].tail(window)
                
                if len(away_prev) >= 3:
                    weights = np.linspace(0.5, 1.0, len(away_prev))
                    wins = []
                    for _, prev_game in away_prev.iterrows():
                        is_home = prev_game['home_team_id'] == game['away_team_id']
                        wins.append(1 if (prev_game['home_winner'] if is_home else prev_game['away_winner']) else 0)
                    momentum = np.average(wins, weights=weights)
                    self.df.at[idx, f'away_momentum_{window}'] = momentum
                
                if idx % 500 == 0:
                    print(f"    Processed {idx}/{len(self.df)} games")
        
        return self.df
    
    def add_schedule_strength(self):
        """Add strength of schedule features"""
        print("Calculating strength of schedule...")
        
        for window in [10, 20]:
            self.df[f'home_sos_{window}'] = np.nan
            self.df[f'away_sos_{window}'] = np.nan
            
            for idx in range(len(self.df)):
                game = self.df.iloc[idx]
                prev_games = self.df.iloc[:idx]
                
                # Home team's opponents' strength
                home_prev = prev_games[
                    (prev_games['home_team_id'] == game['home_team_id']) | 
                    (prev_games['away_team_id'] == game['home_team_id'])
                ].tail(window)
                
                if len(home_prev) >= 5:
                    opponent_win_pcts = []
                    for _, prev_game in home_prev.iterrows():
                        # Get opponent
                        if prev_game['home_team_id'] == game['home_team_id']:
                            opp_id = prev_game['away_team_id']
                        else:
                            opp_id = prev_game['home_team_id']
                        
                        # Get opponent's record
                        opp_games = prev_games[
                            (prev_games['home_team_id'] == opp_id) | 
                            (prev_games['away_team_id'] == opp_id)
                        ]
                        
                        if len(opp_games) > 0:
                            opp_wins = sum([1 for _, g in opp_games.iterrows() 
                                          if (g['home_winner'] if g['home_team_id'] == opp_id else g['away_winner'])])
                            opponent_win_pcts.append(opp_wins / len(opp_games))
                    
                    if opponent_win_pcts:
                        self.df.at[idx, f'home_sos_{window}'] = np.mean(opponent_win_pcts)
                
                # Same for away team
                away_prev = prev_games[
                    (prev_games['home_team_id'] == game['away_team_id']) | 
                    (prev_games['away_team_id'] == game['away_team_id'])
                ].tail(window)
                
                if len(away_prev) >= 5:
                    opponent_win_pcts = []
                    for _, prev_game in away_prev.iterrows():
                        if prev_game['home_team_id'] == game['away_team_id']:
                            opp_id = prev_game['away_team_id']
                        else:
                            opp_id = prev_game['home_team_id']
                        
                        opp_games = prev_games[
                            (prev_games['home_team_id'] == opp_id) | 
                            (prev_games['away_team_id'] == opp_id)
                        ]
                        
                        if len(opp_games) > 0:
                            opp_wins = sum([1 for _, g in opp_games.iterrows() 
                                          if (g['home_winner'] if g['home_team_id'] == opp_id else g['away_winner'])])
                            opponent_win_pcts.append(opp_wins / len(opp_games))
                    
                    if opponent_win_pcts:
                        self.df.at[idx, f'away_sos_{window}'] = np.mean(opponent_win_pcts)
                
                if idx % 500 == 0:
                    print(f"    Processed {idx}/{len(self.df)} games")
        
        return self.df
    
    def create_matchup_features(self):
        """Create head-to-head and differential features"""
        print("Creating matchup features...")
        
        # Differential features
        for window in [5, 10, 20]:
            # Basic differentials
            self.df[f'win_pct_diff_{window}'] = (
                self.df[f'home_win_pct_{window}'] - self.df[f'away_win_pct_{window}']
            )
            self.df[f'point_diff_diff_{window}'] = (
                self.df[f'home_point_diff_{window}'] - self.df[f'away_point_diff_{window}']
            )
            
            # Advanced differentials
            self.df[f'offensive_diff_{window}'] = (
                self.df[f'home_offensive_efficiency_{window}'] - self.df[f'away_offensive_efficiency_{window}']
            )
            self.df[f'defensive_diff_{window}'] = (
                self.df[f'home_defensive_efficiency_{window}'] - self.df[f'away_defensive_efficiency_{window}']
            )
            self.df[f'consistency_diff_{window}'] = (
                self.df[f'home_std_points_{window}'] - self.df[f'away_std_points_{window}']
            )
            
        # Momentum differential
        for window in [5, 10]:
            self.df[f'momentum_diff_{window}'] = (
                self.df[f'home_momentum_{window}'] - self.df[f'away_momentum_{window}']
            )
        
        # SOS differential
        for window in [10, 20]:
            self.df[f'sos_diff_{window}'] = (
                self.df[f'home_sos_{window}'] - self.df[f'away_sos_{window}']
            )
        
        # Home/Away performance differential
        for window in [5, 10, 20]:
            self.df[f'home_away_advantage_{window}'] = (
                self.df[f'home_home_win_pct_{window}'] - self.df[f'away_away_win_pct_{window}']
            )
        
        # Home court advantage indicator
        self.df['home_advantage'] = (~self.df['neutral_site']).astype(int)
        
        return self.df
    
    def prepare_ml_dataset(self):
        """Final preparation for ML - EXCLUDE TARGET-RELATED FEATURES"""
        print("Preparing ML dataset...")
        
        # Get all valid feature columns (exclude metadata and target)
        exclude_cols = ['game_id', 'date', 'season', 'home_team', 'away_team', 
                       'home_team_id', 'away_team_id', 'home_score', 'away_score',
                       'home_winner', 'away_winner', 'status', 'home_record', 
                       'away_record', 'home_stats', 'away_stats', 'point_diff', 'home_win']
        
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        # Remove rows with too many missing features
        df_ml = self.df.dropna(subset=feature_cols, thresh=len(feature_cols)*0.7).copy()
        
        # Fill remaining NaN with median values
        for col in feature_cols:
            if df_ml[col].isna().sum() > 0:
                df_ml[col].fillna(df_ml[col].median(), inplace=True)
        
        print(f"  Samples after removing NaN: {len(df_ml)}")
        print(f"  Feature columns: {len(feature_cols)}")
        print(f"\n  Feature categories:")
        
        categories = {
            'Win percentage': [c for c in feature_cols if 'win_pct' in c],
            'Scoring': [c for c in feature_cols if 'points' in c or 'offensive' in c or 'defensive' in c],
            'Momentum': [c for c in feature_cols if 'momentum' in c or 'streak' in c],
            'Rest': [c for c in feature_cols if 'rest' in c],
            'Head-to-head': [c for c in feature_cols if 'h2h' in c],
            'Strength of schedule': [c for c in feature_cols if 'sos' in c],
            'Context': [c for c in feature_cols if c in ['home_advantage', 'neutral_site', 'conference_game']]
        }
        
        for cat, cols in categories.items():
            print(f"    {cat}: {len(cols)} features")
        
        return df_ml, feature_cols

# Usage
if __name__ == "__main__":
    import json
    
    # Load data
    df = pd.read_csv('data/training_data.csv')
    print(f"Loaded {len(df)} games\n")
    
    # Engineer features
    engineer = FeatureEngineer(df)
    
    print("="*60)
    df = engineer.add_team_features(window_sizes=[5, 10, 20])
    print("\n" + "="*60)
    df = engineer.add_rest_days()
    print("="*60)
    df = engineer.add_head_to_head()
    print("="*60)
    df = engineer.add_momentum_features()
    print("="*60)
    df = engineer.add_schedule_strength()
    print("="*60)
    df = engineer.create_matchup_features()
    print("="*60)
    
    # Prepare for ML
    df_ml, feature_cols = engineer.prepare_ml_dataset()
    
    # Save
    df_ml.to_csv('data/ml_ready_data.csv', index=False)
    
    with open('data/feature_columns.json', 'w') as f:
        json.dump(feature_cols, f)
    
    print(f"\n{'='*60}")
    print(f"✅ ML-ready dataset saved!")
    print(f"   Total samples: {len(df_ml)}")
    print(f"   Features: {len(feature_cols)}")
    print(f"{'='*60}")