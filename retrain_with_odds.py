import pandas as pd
import numpy as np
import json
from datetime import datetime
import requests
from dotenv import load_dotenv
import os
import time

load_dotenv()
ODDS_API_KEY = os.getenv('ODDS_API_KEY')

class OddsFeatureEngineer:
    def __init__(self):
        self.df = pd.read_csv('data/ml_ready_data.csv')
        self.df['date'] = pd.to_datetime(self.df['date'])
        
    def add_historical_odds_features(self):
        """
        Add simulated odds features based on team strength
        (In production, you'd use actual historical odds data)
        """
        print("Adding odds-based features...")
        
        # Simulate spread based on point differential
        self.df['simulated_spread'] = (
            self.df['home_point_diff_10'] - self.df['away_point_diff_10']
        ) * 0.8  # Scale factor
        
        # Simulate over/under based on scoring
        self.df['simulated_total'] = (
            self.df['home_avg_points_10'] + self.df['away_avg_points_10']
        )
        
        # Simulate moneyline (rough conversion from win probability)
        home_implied_prob = 0.5 + (self.df['simulated_spread'] * 0.03)
        home_implied_prob = home_implied_prob.clip(0.1, 0.9)
        
        # Convert probability to American odds
        self.df['simulated_home_ml'] = home_implied_prob.apply(
            lambda p: -100 * p / (1 - p) if p > 0.5 else 100 * (1 - p) / p
        )
        
        # Market efficiency features
        self.df['model_vs_spread_diff'] = (
            self.df['point_diff_diff_10'] - self.df['simulated_spread']
        )
        
        print(f"✅ Added odds features")
        return self.df
    
    def save_enhanced_data(self):
        """Save data with odds features"""
        self.df.to_csv('data/ml_ready_data_with_odds.csv', index=False)
        
        # Update feature columns
        feature_cols = [col for col in self.df.columns if col not in [
            'game_id', 'date', 'season', 'home_team', 'away_team',
            'home_team_id', 'away_team_id', 'home_score', 'away_score',
            'home_winner', 'away_winner', 'status', 'home_record',
            'away_record', 'home_stats', 'away_stats', 'point_diff', 'home_win'
        ]]
        
        with open('data/feature_columns_with_odds.json', 'w') as f:
            json.dump(feature_cols, f)
        
        print(f"💾 Saved enhanced dataset with {len(feature_cols)} features")

if __name__ == "__main__":
    engineer = OddsFeatureEngineer()
    engineer.add_historical_odds_features()
    engineer.save_enhanced_data()
    
    print("\n✅ Ready to retrain model with odds features!")
    print("   Run: python ensemble_optimizer.py")
    print("   (Make sure to update it to use 'data/ml_ready_data_with_odds.csv')")