import requests
import json
from datetime import datetime, timedelta

base_url = "http://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"

def get_scoreboard(date=None):
    """Get games for a specific date (YYYYMMDD format)"""
    url = f"{base_url}/scoreboard"
    if date:
        url += f"?dates={date}"
    response = requests.get(url)
    return response.json()

def get_teams():
    """Get all college basketball teams"""
    url = f"{base_url}/teams"
    params = {'limit': 1000}  # Get more teams
    response = requests.get(url, params=params)
    return response.json()

def get_team_info(team_id):
    """Get detailed info for a specific team"""
    url = f"{base_url}/teams/{team_id}"
    response = requests.get(url)
    return response.json()

def get_rankings():
    """Get current AP rankings"""
    url = f"{base_url}/rankings"
    response = requests.get(url)
    return response.json()

# Test everything
print("Testing ESPN API...")
print("="*60)

# Test 1: Today's games
today = datetime.now().strftime("%Y%m%d")
print(f"\n📅 Today's date: {today}")
scoreboard = get_scoreboard(today)
events = scoreboard.get('events', [])
print(f"✅ Games today: {len(events)}")

if events:
    print("\nSample game data:")
    game = events[0]
    home = game['competitions'][0]['competitors'][0]['team']['displayName']
    away = game['competitions'][0]['competitors'][1]['team']['displayName']
    print(f"   {away} @ {home}")
    print(f"   Status: {game['status']['type']['description']}")

# Test 2: Get historical games
yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
past_games = get_scoreboard(yesterday)
print(f"\n✅ Games yesterday: {len(past_games.get('events', []))}")

# Test 3: Teams
print(f"\n🏀 Fetching teams...")
teams_data = get_teams()
# ESPN returns nested structure
all_teams = []
if 'sports' in teams_data:
    for sport in teams_data['sports']:
        for league in sport.get('leagues', []):
            all_teams.extend(league.get('teams', []))

print(f"✅ Total teams: {len(all_teams)}")

if all_teams:
    sample_team = all_teams[0]['team']
    print(f"   Sample team: {sample_team['displayName']} (ID: {sample_team['id']})")

# Test 4: Rankings
rankings = get_rankings()
print(f"\n🏆 Rankings available: {len(rankings.get('rankings', []))}")

# Test 5: Get detailed team info
if all_teams:
    team_id = all_teams[0]['team']['id']
    team_detail = get_team_info(team_id)
    print(f"\n📊 Team detail for {all_teams[0]['team']['displayName']}:")
    if 'team' in team_detail:
        print(f"   Record: {team_detail['team'].get('record', {}).get('items', [{}])[0].get('summary', 'N/A')}")

# Save sample data
print(f"\n💾 Saving sample data...")
with open('espn_scoreboard_sample.json', 'w') as f:
    json.dump(scoreboard, f, indent=2)
with open('espn_teams_sample.json', 'w') as f:
    json.dump(teams_data, f, indent=2)

print("\n" + "="*60)
print("✅ ESPN API is working!")
print("\nNext: Set up The Odds API")