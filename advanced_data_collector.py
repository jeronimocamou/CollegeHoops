import requests
from dotenv import load_dotenv
import os
import json
from datetime import datetime, timedelta
import time
import pandas as pd

load_dotenv()
ODDS_API_KEY = os.getenv('ODDS_API_KEY')

class AdvancedDataCollector:
    def __init__(self):
        self.espn_base = "http://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
        self.odds_base = "https://api.the-odds-api.com/v4"
    
    def get_full_season_games(self, year=2025, max_games=10000):
        """Get entire season of games"""
        all_games = []
        
        # College basketball season runs Nov-March
        # For 2024-25 season: Nov 2024 - March 2025
        start_date = datetime(year - 1, 11, 1)  # November of previous year
        end_date = datetime(year, 3, 31)  # End of March
        
        current_date = start_date
        
        print(f"Collecting {year} season data...")
        print(f"Date range: {start_date.date()} to {end_date.date()}")
        
        while current_date <= end_date and len(all_games) < max_games:
            date_str = current_date.strftime("%Y%m%d")
            
            try:
                response = requests.get(
                    f"{self.espn_base}/scoreboard",
                    params={'dates': date_str, 'limit': 100}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    events = data.get('events', [])
                    all_games.extend(events)
                    
                    if events:
                        print(f"  {date_str}: {len(events)} games (Total: {len(all_games)})")
                
                time.sleep(0.3)  # Be nice to API
                
            except Exception as e:
                print(f"  Error on {date_str}: {e}")
            
            current_date += timedelta(days=1)
        
        print(f"✅ Collected {len(all_games)} games")
        return all_games
    
    def get_multiple_seasons(self, years=[2023, 2024, 2025]):
        """Get multiple seasons of data"""
        all_seasons = {}
        
        for year in years:
            print(f"\n{'='*60}")
            games = self.get_full_season_games(year)
            all_seasons[year] = games
            
            # Save each season separately
            filename = f"data/season_{year}_raw.json"
            os.makedirs('data', exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(games, f, indent=2)
            print(f"💾 Saved to {filename}")
        
        return all_seasons
    
    def parse_game_with_features(self, game):
        """Extract comprehensive features from a game"""
        try:
            competition = game['competitions'][0]
            competitors = competition['competitors']
            
            home = next((c for c in competitors if c['homeAway'] == 'home'), competitors[0])
            away = next((c for c in competitors if c['homeAway'] == 'away'), competitors[1])
            
            # Extract all useful features
            features = {
                # Basic info
                'game_id': game['id'],
                'date': game['date'],
                'season': game.get('season', {}).get('year'),
                
                # Teams
                'home_team': home['team']['displayName'],
                'home_team_id': home['team']['id'],
                'away_team': away['team']['displayName'],
                'away_team_id': away['team']['id'],
                
                # Scores
                'home_score': home.get('score'),
                'away_score': away.get('score'),
                
                # Outcome
                'home_winner': home.get('winner', False),
                'away_winner': away.get('winner', False),
                
                # Game context
                'status': game['status']['type']['name'],
                'neutral_site': competition.get('neutralSite', False),
                'conference_game': competition.get('conferenceCompetition', False),
                
                # Records at time of game
                'home_record': home.get('records', [{}])[0].get('summary', 'N/A') if home.get('records') else 'N/A',
                'away_record': away.get('records', [{}])[0].get('summary', 'N/A') if away.get('records') else 'N/A',
                
                # Stats (if available)
                'home_stats': home.get('statistics', []),
                'away_stats': away.get('statistics', []),
            }
            
            return features
        except Exception as e:
            print(f"Error parsing game: {e}")
            return None
    
    def create_training_dataset(self, seasons_data):
        """Convert raw games into ML-ready dataset"""
        all_games = []
        
        for season, games in seasons_data.items():
            print(f"Processing {season} season: {len(games)} games")
            for game in games:
                parsed = self.parse_game_with_features(game)
                if parsed and parsed['home_score'] is not None:
                    all_games.append(parsed)
        
        df = pd.DataFrame(all_games)
        
        # Filter out games without scores
        df = df[df['home_score'].notna() & df['away_score'].notna()]
        
        # Convert scores to numeric
        df['home_score'] = pd.to_numeric(df['home_score'])
        df['away_score'] = pd.to_numeric(df['away_score'])
        
        # Create target variable (1 if home wins, 0 if away wins)
        df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
        
        # Create point differential
        df['point_diff'] = df['home_score'] - df['away_score']
        
        print(f"\n✅ Created dataset with {len(df)} complete games")
        print(f"   Home wins: {df['home_win'].sum()}")
        print(f"   Away wins: {len(df) - df['home_win'].sum()}")
        
        return df

# Usage
if __name__ == "__main__":
    collector = AdvancedDataCollector()
    
    # Collect multiple seasons (this will take a while!)
    seasons = collector.get_multiple_seasons(years=[2023, 2024, 2025])
    
    # Create training dataset
    df = collector.create_training_dataset(seasons)
    
    # Save
    df.to_csv('data/training_data.csv', index=False)
    print(f"\n💾 Saved training data to data/training_data.csv")