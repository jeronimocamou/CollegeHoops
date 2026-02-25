import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv
import os
import glob

load_dotenv()
ODDS_API_KEY = os.getenv('ODDS_API_KEY')

class GamePredictor:
    def __init__(self, model_path=None):
        """Load the trained model and preprocessing objects"""
        
        # Find most recent model if not specified
        if model_path is None:
            model_files = glob.glob('models/optimized_ensemble_*.pkl')
            if not model_files:
                model_files = glob.glob('models/best_ensemble_model_*.pkl')
            if not model_files:
                raise FileNotFoundError("No model found! Train a model first.")
            model_path = max(model_files, key=os.path.getctime)
        
        print(f"Loading model from: {model_path}")
        
        # Extract timestamp
        basename = os.path.basename(model_path)
        parts = basename.replace('.pkl', '').split('_')
        
        if len(parts) >= 2:
            timestamp = f"{parts[-2]}_{parts[-1]}"
        else:
            raise ValueError(f"Could not extract timestamp from {model_path}")
        
        print(f"Using timestamp: {timestamp}")
        
        # Load model
        self.model = joblib.load(model_path)
        
        # Load preprocessing objects (with fallback to most recent)
        scaler_path = f'models/scaler_{timestamp}.pkl'
        imputer_path = f'models/imputer_{timestamp}.pkl'
        features_path = f'models/features_{timestamp}.json'
        
        if not os.path.exists(scaler_path):
            scaler_files = glob.glob('models/scaler_*.pkl')
            if scaler_files:
                scaler_path = max(scaler_files, key=os.path.getctime)
        
        if not os.path.exists(imputer_path):
            imputer_files = glob.glob('models/imputer_*.pkl')
            if imputer_files:
                imputer_path = max(imputer_files, key=os.path.getctime)
        
        if not os.path.exists(features_path):
            features_files = glob.glob('models/features_*.json')
            if features_files:
                features_path = max(features_files, key=os.path.getctime)
        
        self.scaler = joblib.load(scaler_path)
        self.imputer = joblib.load(imputer_path)
        
        with open(features_path, 'r') as f:
            self.feature_cols = json.load(f)
        
        # Load historical data
        self.historical_df = pd.read_csv('data/ml_ready_data.csv')
        self.historical_df['date'] = pd.to_datetime(self.historical_df['date'])
        
        self.raw_games = pd.read_csv('data/training_data.csv')
        self.raw_games['date'] = pd.to_datetime(self.raw_games['date'])
        
        print(f"✅ Model loaded successfully")
        print(f"   Features: {len(self.feature_cols)}")
        print(f"   Historical games: {len(self.raw_games)}\n")
    
    def get_espn_games(self, date=None):
        """Get upcoming games from ESPN"""
        if date is None:
            date = (datetime.now() + timedelta(days=1)).strftime("%Y%m%d")
        
        print(f"📅 Fetching games for {date}...")
        
        url = f"http://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
        params = {'dates': date}
        
        response = requests.get(url, params=params)
        data = response.json()
        
        games = []
        for event in data.get('events', []):
            try:
                competition = event['competitions'][0]
                competitors = competition['competitors']
                
                home = next((c for c in competitors if c['homeAway'] == 'home'), competitors[0])
                away = next((c for c in competitors if c['homeAway'] == 'away'), competitors[1])
                
                game = {
                    'game_id': event['id'],
                    'date': event['date'],
                    'home_team': home['team']['displayName'],
                    'home_team_id': home['team']['id'],
                    'away_team': away['team']['displayName'],
                    'away_team_id': away['team']['id'],
                    'neutral_site': competition.get('neutralSite', False),
                    'status': event['status']['type']['name']
                }
                games.append(game)
            except Exception as e:
                print(f"Error parsing game: {e}")
        
        print(f"✅ Found {len(games)} games\n")
        return pd.DataFrame(games)
    
    def get_odds(self):
        """Get current odds from The Odds API"""
        if not ODDS_API_KEY:
            print("⚠️  No Odds API key found. Predictions will not include Vegas data.\n")
            return None
        
        print("💰 Fetching betting odds...")
        
        url = f"https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds"
        params = {
            'apiKey': ODDS_API_KEY,
            'regions': 'us',
            'markets': 'h2h,spreads,totals'
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            print(f"⚠️  Odds API error: {response.status_code}\n")
            return None
        
        remaining = response.headers.get('x-requests-remaining')
        print(f"   API calls remaining: {remaining}")
        
        odds_data = response.json()
        print(f"✅ Found odds for {len(odds_data)} games\n")
        
        return odds_data
    
    def calculate_team_features(self, team_id, window=10):
        """Calculate recent features for a team"""
        team_games = self.raw_games[
            (self.raw_games['home_team_id'] == team_id) | 
            (self.raw_games['away_team_id'] == team_id)
        ].tail(window)
        
        if len(team_games) < 3:
            return None
        
        wins = 0
        points_for = []
        points_against = []
        
        for _, game in team_games.iterrows():
            is_home = game['home_team_id'] == team_id
            
            if is_home:
                wins += 1 if game['home_winner'] else 0
                points_for.append(game['home_score'])
                points_against.append(game['away_score'])
            else:
                wins += 1 if game['away_winner'] else 0
                points_for.append(game['away_score'])
                points_against.append(game['home_score'])
        
        return {
            'win_pct': wins / len(team_games),
            'avg_points': np.mean(points_for),
            'avg_points_against': np.mean(points_against),
            'point_diff': np.mean(points_for) - np.mean(points_against),
            'games_found': len(team_games)
        }
    
    def create_game_features(self, game):
        """Create feature vector for a game"""
        features = {}
        
        # Start with median values
        for col in self.feature_cols:
            if col in ['home_advantage', 'neutral_site', 'conference_game']:
                if col == 'home_advantage':
                    features[col] = 0 if game.get('neutral_site', False) else 1
                elif col == 'neutral_site':
                    features[col] = 1 if game.get('neutral_site', False) else 0
                else:
                    features[col] = 0
            else:
                features[col] = self.historical_df[col].median()
        
        # Calculate recent stats
        home_stats_all = {}
        away_stats_all = {}
        
        for window in [5, 10, 20]:
            home_stats = self.calculate_team_features(game['home_team_id'], window)
            away_stats = self.calculate_team_features(game['away_team_id'], window)
            
            # Store for debug output
            if window == 10:
                home_stats_all = home_stats
                away_stats_all = away_stats
            
            if home_stats and away_stats:
                if f'home_win_pct_{window}' in self.feature_cols:
                    features[f'home_win_pct_{window}'] = home_stats['win_pct']
                if f'away_win_pct_{window}' in self.feature_cols:
                    features[f'away_win_pct_{window}'] = away_stats['win_pct']
                if f'home_point_diff_{window}' in self.feature_cols:
                    features[f'home_point_diff_{window}'] = home_stats['point_diff']
                if f'away_point_diff_{window}' in self.feature_cols:
                    features[f'away_point_diff_{window}'] = away_stats['point_diff']
                if f'home_avg_points_{window}' in self.feature_cols:
                    features[f'home_avg_points_{window}'] = home_stats['avg_points']
                if f'away_avg_points_{window}' in self.feature_cols:
                    features[f'away_avg_points_{window}'] = away_stats['avg_points']
                if f'home_avg_points_against_{window}' in self.feature_cols:
                    features[f'home_avg_points_against_{window}'] = home_stats['avg_points_against']
                if f'away_avg_points_against_{window}' in self.feature_cols:
                    features[f'away_avg_points_against_{window}'] = away_stats['avg_points_against']
                
                # Differentials
                if f'win_pct_diff_{window}' in self.feature_cols:
                    features[f'win_pct_diff_{window}'] = home_stats['win_pct'] - away_stats['win_pct']
                if f'point_diff_diff_{window}' in self.feature_cols:
                    features[f'point_diff_diff_{window}'] = home_stats['point_diff'] - away_stats['point_diff']
                if f'offensive_diff_{window}' in self.feature_cols:
                    features[f'offensive_diff_{window}'] = home_stats['avg_points'] - away_stats['avg_points']
        
        # DEBUG OUTPUT
        print(f"\n  📊 DATA QUALITY CHECK:")
        print(f"     {game['home_team']} (HOME):")
        if home_stats_all:
            print(f"       ✅ {home_stats_all['games_found']} games found")
            print(f"       📈 {home_stats_all['avg_points']:.1f} ppg, {home_stats_all['avg_points_against']:.1f} allowed")
            print(f"       🏆 {home_stats_all['win_pct']*100:.0f}% win rate, {home_stats_all['point_diff']:+.1f} pt differential")
        else:
            print(f"       ⚠️  NO DATA - using median values from all teams")
        
        print(f"     {game['away_team']} (AWAY):")
        if away_stats_all:
            print(f"       ✅ {away_stats_all['games_found']} games found")
            print(f"       📈 {away_stats_all['avg_points']:.1f} ppg, {away_stats_all['avg_points_against']:.1f} allowed")
            print(f"       🏆 {away_stats_all['win_pct']*100:.0f}% win rate, {away_stats_all['point_diff']:+.1f} pt differential")
        else:
            print(f"       ⚠️  NO DATA - using median values from all teams")
        
        return pd.DataFrame([features])[self.feature_cols], home_stats_all, away_stats_all
    
    def add_odds_to_predictions(self, predictions, odds_data):
        """Add odds information to predictions"""
        if odds_data is None:
            return predictions
        
        for i, pred in predictions.iterrows():
            for odds_game in odds_data:
                home_match = pred['home_team'] in odds_game['home_team'] or odds_game['home_team'] in pred['home_team']
                away_match = pred['away_team'] in odds_game['away_team'] or odds_game['away_team'] in pred['away_team']
                
                if home_match and away_match:
                    if odds_game.get('bookmakers'):
                        bookmaker = odds_game['bookmakers'][0]
                        
                        # Get spread
                        spread_market = next((m for m in bookmaker['markets'] if m['key'] == 'spreads'), None)
                        if spread_market:
                            for outcome in spread_market['outcomes']:
                                if 'home' in outcome['name'].lower() or pred['home_team'].split()[-1] in outcome['name']:
                                    predictions.at[i, 'vegas_spread'] = outcome.get('point', 0)
                    
                    break
        
        return predictions
    
    def predict_games(self, date=None):
        """Predict outcomes for upcoming games"""
        games_df = self.get_espn_games(date)
        
        if len(games_df) == 0:
            print("⚠️  No games found for this date")
            return None
        
        odds_data = self.get_odds()
        
        print("🤖 Generating predictions...\n")
        print("="*90)
        
        predictions = []
        
        for idx, game in games_df.iterrows():
            print(f"\nProcessing: {game['away_team']} @ {game['home_team']}")
            
            try:
                features, home_stats, away_stats = self.create_game_features(game)
                features_imputed = self.imputer.transform(features)
                
                pred_proba = self.model.predict_proba(features_imputed)[0]
                pred_class = self.model.predict(features_imputed)[0]
                
                # Calculate predicted spread
                home_expected = home_stats['avg_points'] if home_stats else 75
                away_expected = away_stats['avg_points'] if away_stats else 75
                home_def = home_stats['avg_points_against'] if home_stats else 70
                away_def = away_stats['avg_points_against'] if away_stats else 70
                
                # Predicted scoring
                home_projected = (home_expected + away_def) / 2
                away_projected = (away_expected + home_def) / 2
                
                # Add home court advantage
                home_advantage = 3.0 if not game.get('neutral_site', False) else 0
                home_projected += home_advantage
                
                # Predicted spread (NEGATIVE means home team favored)
                predicted_spread = -(home_projected - away_projected)
                
                # Mark data quality
                has_good_data = (home_stats is not None and away_stats is not None)
                
                prediction = {
                    'game_id': game['game_id'],
                    'date': game['date'],
                    'away_team': game['away_team'],
                    'home_team': game['home_team'],
                    'neutral_site': game['neutral_site'],
                    'predicted_winner': 'HOME' if pred_class == 1 else 'AWAY',
                    'confidence': 'HIGH' if abs(pred_proba[1] - 0.5) > 0.2 else 'MEDIUM' if abs(pred_proba[1] - 0.5) > 0.1 else 'LOW',
                    'predicted_spread': predicted_spread,
                    'data_quality': 'GOOD' if has_good_data else 'POOR'
                }
                
                predictions.append(prediction)
                
            except Exception as e:
                print(f"❌ Error predicting {game['home_team']} vs {game['away_team']}: {e}")
        
        print("\n" + "="*90 + "\n")
        
        predictions_df = pd.DataFrame(predictions)
        predictions_df = self.add_odds_to_predictions(predictions_df, odds_data)
        
        return predictions_df
    
    def display_predictions(self, predictions):
        """Display predictions in clean, simple format"""
        if predictions is None or len(predictions) == 0:
            return
        
        # Filter out predictions with extreme disagreement (likely bad data)
        predictions_filtered = []
        skipped = []
        
        for _, pred in predictions.iterrows():
            if 'vegas_spread' in pred and not pd.isna(pred['vegas_spread']):
                spread_diff = abs(pred['predicted_spread'] - pred['vegas_spread'])
                if spread_diff > 15:  # If we disagree by more than 15 points
                    skipped.append({
                        'game': f"{pred['away_team']} @ {pred['home_team']}",
                        'reason': f"Off by {spread_diff:.1f} pts - likely insufficient data",
                        'data_quality': pred.get('data_quality', 'UNKNOWN')
                    })
                    continue
            predictions_filtered.append(pred)
        
        if len(skipped) > 0:
            print("="*90)
            print("⚠️  SKIPPED PREDICTIONS (Unreliable Data)")
            print("="*90)
            for skip in skipped:
                print(f"  {skip['game']}")
                print(f"    Reason: {skip['reason']}")
                print(f"    Data Quality: {skip['data_quality']}")
            print()
        
        if len(predictions_filtered) == 0:
            print("No reliable predictions available.\n")
            return
        
        predictions = pd.DataFrame(predictions_filtered)
        
        print("="*90)
        print(f"{'COLLEGE BASKETBALL PREDICTIONS':^90}")
        print("="*90 + "\n")
        
        for idx, pred in enumerate(predictions.iterrows(), 1):
            _, pred = pred
            
            print(f"{'─'*90}")
            print(f"GAME {idx}: {pred['away_team']} @ {pred['home_team']}")
            
            # Show data quality warning
            if pred.get('data_quality') == 'POOR':
                print(f"         ⚠️  WARNING: Limited historical data - use caution")
            
            print(f"{'─'*90}")
            
            # Prediction
            winner = pred['home_team'] if pred['predicted_winner'] == 'HOME' else pred['away_team']
            print(f"\n  PICK:                {winner}")
            print(f"  CONFIDENCE:          {pred['confidence']}")
            
            # Spreads
            has_vegas = 'vegas_spread' in pred and not pd.isna(pred['vegas_spread'])
            
            print(f"\n  PREDICTED SPREAD:    {pred['home_team'][:30]} {pred['predicted_spread']:+.1f}")
            
            if has_vegas:
                print(f"  VEGAS SPREAD:        {pred['home_team'][:30]} {pred['vegas_spread']:+.1f}")
                
                # Value analysis
                spread_diff = abs(pred['predicted_spread'] - pred['vegas_spread'])
                
                print(f"\n  VALUE ANALYSIS:")
                
                if spread_diff > 4:
                    if pred['predicted_spread'] < pred['vegas_spread']:
                        print(f"    🔥 STRONG VALUE on {pred['home_team'][:30]}")
                        print(f"       Our model has them {spread_diff:.1f} points stronger than Vegas")
                    else:
                        print(f"    🔥 STRONG VALUE on {pred['away_team'][:30]}")
                        print(f"       Our model has them {spread_diff:.1f} points stronger than Vegas")
                
                elif spread_diff > 2:
                    if pred['predicted_spread'] < pred['vegas_spread']:
                        print(f"    📊 Moderate value on {pred['home_team'][:30]} ({spread_diff:.1f} pt edge)")
                    else:
                        print(f"    📊 Moderate value on {pred['away_team'][:30]} ({spread_diff:.1f} pt edge)")
                
                else:
                    print(f"    ✅ Model agrees with Vegas (within {spread_diff:.1f} pts)")
            else:
                print(f"  VEGAS SPREAD:        Not available")
            
            print()
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"predictions/predictions_{timestamp}.csv"
        os.makedirs('predictions', exist_ok=True)
        predictions.to_csv(output_file, index=False)
        
        print("="*90)
        print(f"SUMMARY: {len(predictions)} reliable predictions")
        if len(skipped) > 0:
            print(f"         {len(skipped)} games skipped due to insufficient data")
        print(f"Saved to: {output_file}")
        print("="*90 + "\n")

# Usage
if __name__ == "__main__":
    predictor = GamePredictor()
    predictions = predictor.predict_games()
    predictor.display_predictions(predictions)

