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
    """Get the start date of the week using May 1, 2024 as reference point."""
    date = datetime.strptime(date_str, '%Y-%m-%d')
    
    # May 1, 2024 is our reference start date for week grouping
    reference_date = datetime(2024, 5, 1)
    
    # Calculate how many days have passed since May 1
    days_diff = (date - reference_date).days
    
    # Group into 7-day weeks starting from May 1
    week_number = days_diff // 7
    week_start = reference_date + timedelta(days=week_number * 7)
    
    return week_start.strftime('%Y-%m-%d')

def collapse_to_weekly(commits_data):
    """Collapse daily commits to weekly totals."""
    weekly_commits = defaultdict(int)

    for username, user_data in commits_data.items():
        daily_commits = user_data.get('daily_commits', {})
        
        for date_str, commit_count in daily_commits.items():
            if commit_count > 0:
                week_start = get_week_start_date(date_str)
                weekly_commits[week_start] += commit_count
    
    return dict(weekly_commits)

def filter_complete_weeks(weekly_data):
    """Filter to only include complete weeks within May-August."""
    if not weekly_data:
        return weekly_data
    
    # May-August boundaries
    may_start = datetime(2024, 5, 1)
    aug_end = datetime(2024, 9, 4)
    
    filtered_weeks = {}
    for week, commits in weekly_data.items():
        week_date = datetime.strptime(week, '%Y-%m-%d')
        # Include week if it starts within our period
        if may_start <= week_date <= aug_end:
            filtered_weeks[week] = commits
    
    print(f"Kept {len(filtered_weeks)} weeks in May-August period")
    
    return filtered_weeks

def analyze_outliers(weekly_data):
    """Analyze weekly commits for outliers based on average."""
    if not weekly_data:
        raise ValueError("No weekly data available for analysis")
    
    commits_list = list(weekly_data.values())
    if len(commits_list) == 0:
        raise ValueError("No commits data available for analysis")
    
    average_commits = sum(commits_list) / len(commits_list)
    
    # Define thresholds
    threshold_50_above = average_commits * 1.5  # 50% above average
    threshold_50_below = average_commits * 0.5  # 50% below average
    threshold_100_above = average_commits * 2.0  # 100% above average
    
    # Categorize weeks
    normal_weeks = []
    high_50_weeks = []
    low_50_weeks = []
    high_100_weeks = []
    
    for week, commits in weekly_data.items():
        if commits >= threshold_100_above:
            high_100_weeks.append((week, commits))
        elif commits >= threshold_50_above:
            high_50_weeks.append((week, commits))
        elif commits <= threshold_50_below:
            low_50_weeks.append((week, commits))
        else:
            normal_weeks.append((week, commits))
    
    return {
        'average': average_commits,
        'thresholds': {
            '50_above': threshold_50_above,
            '50_below': threshold_50_below,
            '100_above': threshold_100_above
        },
        'categories': {
            'normal': normal_weeks,
            'high_50': high_50_weeks,
            'low_50': low_50_weeks,
            'high_100': high_100_weeks
        }
    }

def print_outliers_summary(analysis):
    """Print detailed summary of outliers analysis."""
    avg = analysis['average']
    thresholds = analysis['thresholds']
    categories = analysis['categories']
    
    print("=" * 60)
    print("WEEKLY COMMITS OUTLIERS ANALYSIS - PHILIPPINES (MAY-AUG 2024)")
    print("=" * 60)
    
    print(f"\n📊 BASIC STATISTICS:")
    print(f"Average weekly commits: {avg:,.0f}")
    print(f"50% above threshold: {thresholds['50_above']:,.0f}")
    print(f"50% below threshold: {thresholds['50_below']:,.0f}")
    print(f"100% above threshold: {thresholds['100_above']:,.0f}")
    
    total_weeks = sum(len(cat) for cat in categories.values())
    
    print(f"\n📈 CATEGORIES BREAKDOWN:")
    print(f"Normal weeks (±50%): {len(categories['normal'])} ({len(categories['normal'])/total_weeks*100:.1f}%)")
    print(f"High activity (+50% to +100%): {len(categories['high_50'])} ({len(categories['high_50'])/total_weeks*100:.1f}%)")
    print(f"Very high activity (+100% or more): {len(categories['high_100'])} ({len(categories['high_100'])/total_weeks*100:.1f}%)")
    print(f"Low activity (-50% or less): {len(categories['low_50'])} ({len(categories['low_50'])/total_weeks*100:.1f}%)")
    
    # Show specific weeks
    if categories['high_100']:
        print(f"\n🔥 VERY HIGH ACTIVITY WEEKS (+100% above average):")
        for week, commits in sorted(categories['high_100'], key=lambda x: x[1], reverse=True):
            percentage = (commits / avg - 1) * 100
            print(f"   {week}: {commits:,} commits (+{percentage:.0f}%)")
    
    if categories['high_50']:
        print(f"\n📈 HIGH ACTIVITY WEEKS (+50% to +100% above average):")
        for week, commits in sorted(categories['high_50'], key=lambda x: x[1], reverse=True):
            percentage = (commits / avg - 1) * 100
            print(f"   {week}: {commits:,} commits (+{percentage:.0f}%)")
    
    if categories['low_50']:
        print(f"\n📉 LOW ACTIVITY WEEKS (-50% or less below average):")
        for week, commits in sorted(categories['low_50'], key=lambda x: x[1]):
            percentage = (1 - commits / avg) * 100
            print(f"   {week}: {commits:,} commits (-{percentage:.0f}%)")

def plot_outliers(weekly_data, analysis):
    """Create a plot highlighting outliers."""
    # Sort dates for proper plotting
    sorted_weeks = sorted(weekly_data.items())
    dates = [item[0] for item in sorted_weeks]
    commits = [item[1] for item in sorted_weeks]
    
    # Convert date strings to datetime objects
    date_objects = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
    
    avg = analysis['average']
    thresholds = analysis['thresholds']
    
    plt.figure(figsize=(16, 10))
    
    # Plot the main line
    plt.plot(date_objects, commits, color='steelblue', linewidth=2, alpha=0.7, label='Weekly Commits')
    
    # Add horizontal lines for thresholds
    plt.axhline(y=avg, color='green', linestyle='-', linewidth=2, label=f'Average ({avg:,.0f})')
    plt.axhline(y=thresholds['50_above'], color='orange', linestyle='--', linewidth=1.5, label=f'+50% ({thresholds["50_above"]:,.0f})')
    plt.axhline(y=thresholds['50_below'], color='red', linestyle='--', linewidth=1.5, label=f'-50% ({thresholds["50_below"]:,.0f})')
    plt.axhline(y=thresholds['100_above'], color='darkred', linestyle=':', linewidth=2, label=f'+100% ({thresholds["100_above"]:,.0f})')
    
    # Highlight outlier points
    categories = analysis['categories']
    
    for week, commit_count in categories['high_100']:
        week_date = datetime.strptime(week, '%Y-%m-%d')
        plt.scatter(week_date, commit_count, color='darkred', s=100, zorder=5, marker='^')
    
    for week, commit_count in categories['high_50']:
        week_date = datetime.strptime(week, '%Y-%m-%d')
        plt.scatter(week_date, commit_count, color='orange', s=80, zorder=5, marker='o')
    
    for week, commit_count in categories['low_50']:
        week_date = datetime.strptime(week, '%Y-%m-%d')
        plt.scatter(week_date, commit_count, color='red', s=80, zorder=5, marker='v')
    
    # Add treatment period shading
    treatment_start = datetime(2024, 7, 17)
    treatment_end = datetime(2024, 7, 24)
    plt.axvspan(treatment_start, treatment_end, alpha=0.3, color='gray', label='Apagón Bangladesh')
    
    plt.title('Weekly Commits with Outliers Analysis - Philippines (May-Aug 2024)', fontsize=16, fontweight='bold')
    plt.xlabel('Week Starting Date', fontsize=12)
    plt.ylabel('Total Commits', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.legend(loc='upper right')
    
    # Format y-axis
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('philippines/weekly_commits_outliers_may_aug_2024.png', dpi=300, bbox_inches='tight')
    plt.savefig('philippines/weekly_commits_outliers_may_aug_2024.pdf', bbox_inches='tight')
    print("\nOutliers plot saved as 'philippines/weekly_commits_outliers_may_aug_2024.png' and .pdf")

def save_outliers_data(analysis, weekly_data):
    """Save outliers analysis to JSON file."""
    # Prepare data for JSON serialization
    output_data = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'period': 'May-August 2024',
        'note': 'Analysis for May-August period only',
        'statistics': {
            'average_weekly_commits': round(analysis['average'], 2),
            'total_weeks_analyzed': sum(len(cat) for cat in analysis['categories'].values())
        },
        'thresholds': {
            'average': round(analysis['average'], 2),
            '50_percent_above': round(analysis['thresholds']['50_above'], 2),
            '50_percent_below': round(analysis['thresholds']['50_below'], 2),
            '100_percent_above': round(analysis['thresholds']['100_above'], 2)
        },
        'categories': {
            'normal_weeks': len(analysis['categories']['normal']),
            'high_activity_weeks_50_100': len(analysis['categories']['high_50']),
            'very_high_activity_weeks_100_plus': len(analysis['categories']['high_100']),
            'low_activity_weeks': len(analysis['categories']['low_50'])
        },
        'outlier_weeks': {
            'very_high_activity': [{'week': week, 'commits': commits, 'percentage_above_avg': round((commits/analysis['average'] - 1) * 100, 1)} 
                                 for week, commits in analysis['categories']['high_100']],
            'high_activity': [{'week': week, 'commits': commits, 'percentage_above_avg': round((commits/analysis['average'] - 1) * 100, 1)} 
                            for week, commits in analysis['categories']['high_50']],
            'low_activity': [{'week': week, 'commits': commits, 'percentage_below_avg': round((1 - commits/analysis['average']) * 100, 1)} 
                           for week, commits in analysis['categories']['low_50']]
        }
    }
    
    with open('philippines/outliers_analysis_may_aug_2024.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("Outliers analysis saved as 'philippines/outliers_analysis_may_aug_2024.json'")

def main():
    print("Loading commits data...")
    commits_data = load_commits_data('philippines/commits.json')
    
    print(f"Loaded data for {len(commits_data)} users")
    
    print("Collapsing to weekly data (May-August only)...")
    weekly_data = collapse_to_weekly(commits_data)
    
    print(f"Generated weekly data for {len(weekly_data)} weeks")
    
    # Filter to complete weeks in May-August
    print("Filtering to May-August period...")
    filtered_weekly_data = filter_complete_weeks(weekly_data)
    
    if not filtered_weekly_data:
        print("❌ No weekly data available after filtering!")
        return
    
    print("Analyzing outliers...")
    try:
        analysis = analyze_outliers(filtered_weekly_data)
    except ValueError as e:
        print(f"❌ Error in outliers analysis: {e}")
        return
    
    # Print detailed summary
    print_outliers_summary(analysis)
    
    # Create and save plots
    print("\nCreating outliers visualization...")
    plot_outliers(filtered_weekly_data, analysis)
    
    # Save analysis data
    save_outliers_data(analysis, filtered_weekly_data)
    
    print("\n✅ Outliers analysis complete!")

if __name__ == "__main__":
    main()