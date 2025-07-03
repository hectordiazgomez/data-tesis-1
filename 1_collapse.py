import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import defaultdict

def load_commits_data(file_path):
    """Load commits data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def get_week_start_date(date_str):
    """Get the start date of the week using July 17, 2024 as reference point."""
    date = datetime.strptime(date_str, '%Y-%m-%d')
    
    # July 17, 2024 is our reference start date for week grouping
    reference_date = datetime(2024, 7, 17)
    
    # Calculate how many days have passed since July 17
    days_diff = (date - reference_date).days
    
    # Group into 7-day weeks starting from July 17
    week_number = days_diff // 7
    week_start = reference_date + timedelta(days=week_number * 7)
    
    return week_start.strftime('%Y-%m-%d')

def collapse_to_weekly(commits_data):
    """Collapse daily commits to weekly totals."""
    weekly_commits = defaultdict(int)
    
    for username, user_data in commits_data.items():
        daily_commits = user_data.get('daily_commits', {})
        
        for date_str, commit_count in daily_commits.items():
            if commit_count > 0:  # Only process dates with commits
                week_start = get_week_start_date(date_str)
                weekly_commits[week_start] += commit_count
    
    return dict(weekly_commits)

def filter_complete_weeks(weekly_data):
    """Remove first and last weeks to ensure complete 7-day periods."""
    if not weekly_data:
        return weekly_data
    
    # Sort weeks by date
    sorted_weeks = sorted(weekly_data.items())
    
    # Remove first and last week
    if len(sorted_weeks) <= 2:
        print("Warning: Not enough weeks to remove first and last. Keeping all data.")
        return weekly_data
    
    filtered_weeks = sorted_weeks[1:-1]  # Remove first and last
    
    first_removed = sorted_weeks[0]
    last_removed = sorted_weeks[-1]
    
    print(f"Removed first week: {first_removed[0]} ({first_removed[1]:,} commits)")
    print(f"Removed last week: {last_removed[0]} ({last_removed[1]:,} commits)")
    print(f"Kept {len(filtered_weeks)} complete weeks out of {len(sorted_weeks)} total weeks")
    
    return dict(filtered_weeks)

def plot_weekly_commits(weekly_data):
    """Create a simple line plot of weekly commits."""
    # Sort dates for proper plotting
    sorted_weeks = sorted(weekly_data.items())
    dates = [item[0] for item in sorted_weeks]
    commits = [item[1] for item in sorted_weeks]
    
    # Convert date strings to datetime objects for better plotting
    date_objects = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
    
    plt.figure(figsize=(15, 8))
    plt.plot(date_objects, commits, marker='o', linewidth=2, markersize=4)
    plt.title('Weekly Commits - Philippines GitHub Users (2024)\n(Complete weeks only)', fontsize=16, fontweight='bold')
    plt.xlabel('Week Starting Date', fontsize=12)
    plt.ylabel('Total Commits', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Format y-axis to show comma-separated numbers
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('philippines/weekly_commits_2024_filtered.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'philippines/weekly_commits_2024_filtered.png'")
    
    # Also save as PDF for high quality
    plt.savefig('philippines/weekly_commits_2024_filtered.pdf', bbox_inches='tight')
    print("Plot saved as 'philippines/weekly_commits_2024_filtered.pdf'")
    
    # Print some basic statistics
    print(f"\n--- Weekly Commits Summary (Complete Weeks Only) ---")
    print(f"Complete weeks analyzed: {len(weekly_data)}")
    print(f"Date range: {dates[0]} to {dates[-1]}")
    print(f"Total commits: {sum(commits):,}")
    print(f"Average commits per week: {sum(commits)/len(commits):,.0f}")
    print(f"Peak week: {dates[commits.index(max(commits))]} with {max(commits):,} commits")
    print(f"Lowest week: {dates[commits.index(min(commits))]} with {min(commits):,} commits")

def main():
    # Load the commits data
    print("Loading commits data...")
    commits_data = load_commits_data('philippines/commits.json')
    
    print(f"Loaded data for {len(commits_data)} users")
    
    # Collapse to weekly data
    print("Collapsing daily data to weekly...")
    weekly_data = collapse_to_weekly(commits_data)
    
    print(f"Generated weekly data for {len(weekly_data)} weeks")
    
    # Filter out incomplete first and last weeks
    print("Filtering out incomplete first and last weeks...")
    filtered_weekly_data = filter_complete_weeks(weekly_data)
    
    # Create the plot
    print("Creating weekly commits plot...")
    plot_weekly_commits(filtered_weekly_data)

if __name__ == "__main__":
    main()