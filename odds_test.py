import requests
from dotenv import load_dotenv
import os
import json

load_dotenv()
ODDS_API_KEY = os.getenv("ODDS_API_KEY")

base_url = "https://api.the-odds-api.com/v4"

def get_sports():
    """Get all available sports"""
    url = f"{base_url}/sports"
    params = {'apiKey': ODDS_API_KEY}
    response = requests.get(url, params=params)
    return response.json()

def get_odds(sport='basketball_ncaab', regions='us', markets='h2h'):
    """
    Get odds for college basketball
    markets: 'h2h' (moneyline), 'spreads', 'totals'
    """
    url = f"{base_url}/sports/{sport}/odds"
    params = {
        'apiKey': ODDS_API_KEY,
        'regions': regions,
        'markets': markets
    }
    response = requests.get(url, params=params)
    
    # Check remaining requests
    remaining = response.headers.get('x-requests-remaining')
    used = response.headers.get('x-requests-used')
    print(f"📊 API Usage - Used: {used}, Remaining: {remaining}")
    
    return response.json()

# Test it
print("Testing The Odds API...")
print("="*60)

# Test 1: Get available sports - DEBUG VERSION
sports = get_sports()

# First, let's see what we actually got
print(f"\n🔍 DEBUG - Sports response type: {type(sports)}")
print(f"🔍 DEBUG - Sports response content:")
print(json.dumps(sports, indent=2)[:500])  # Print first 500 chars

# Check if it's an error message
if isinstance(sports, dict) and 'message' in sports:
    print(f"\n❌ ERROR: {sports['message']}")
    print("Check your ODDS_API_KEY in .env file")
    exit()

# If it's a list, continue
if isinstance(sports, list):
    print(f"\n✅ Available sports: {len(sports)}")
    
    # Find college basketball
    ncaab = [s for s in sports if isinstance(s, dict) and 'ncaab' in s.get('key', '')]
    if ncaab:
        print(f"   College Basketball found: {ncaab[0]['title']}")
        print(f"   Key: {ncaab[0]['key']}")
        
        # Test 2: Get current odds
        print(f"\n🎲 Fetching odds...")
        odds_data = get_odds(sport='basketball_ncaab', markets='h2h,spreads,totals')
        
        # Debug odds response too
        print(f"\n🔍 DEBUG - Odds response type: {type(odds_data)}")
        
        if isinstance(odds_data, dict) and 'message' in odds_data:
            print(f"\n❌ ERROR: {odds_data['message']}")
        elif isinstance(odds_data, list):
            print(f"✅ Games with odds: {len(odds_data)}")
            
            if odds_data:
                # Show sample
                game = odds_data[0]
                print(f"\nSample game:")
                print(f"   {game['away_team']} @ {game['home_team']}")
                print(f"   Starts: {game['commence_time']}")
                
                if game.get('bookmakers'):
                    bookmaker = game['bookmakers'][0]
                    print(f"   Bookmaker: {bookmaker['title']}")
                    
                    for market in bookmaker['markets']:
                        print(f"   Market: {market['key']}")
                        for outcome in market['outcomes'][:2]:  # Show first 2
                            print(f"      {outcome['name']}: {outcome.get('price', outcome.get('point', 'N/A'))}")
            
            # Save sample
            with open('odds_sample.json', 'w') as f:
                json.dump(odds_data, f, indent=2)
                print(f"\n💾 Saved odds_sample.json")
    else:
        print("\n❌ College Basketball (ncaab) not found in available sports")
        print("Available sports:")
        for sport in sports:
            if isinstance(sport, dict):
                print(f"   - {sport.get('title', 'Unknown')}: {sport.get('key', 'Unknown')}")
else:
    print(f"\n❌ Unexpected response format")
    print("Response:", sports)

print("\n" + "="*60)