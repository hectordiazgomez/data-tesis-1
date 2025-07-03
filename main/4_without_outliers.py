import json
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
    """Collapse daily commits to weekly totals for May-August period."""
    weekly_commits = defaultdict(int)
    
    # Define May-August boundaries
    may_start = datetime(2024, 5, 1)
    aug_end = datetime(2024, 9, 4)
    
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

def remove_weekly_outliers(weekly_data, threshold_percentage=50):
    """Remove weeks that are outliers (above threshold% of average)."""
    if not weekly_data:
        return {}, []
    
    commits_list = list(weekly_data.values())
    if len(commits_list) == 0:
        return {}, []
    
    average_commits = sum(commits_list) / len(commits_list)
    threshold = average_commits * (1 + threshold_percentage / 100)
    
    filtered_data = {}
    removed_weeks = []
    
    for week, commits in weekly_data.items():
        if commits <= threshold:
            filtered_data[week] = commits
        else:
            removed_weeks.append((week, commits))
    
    print(f"Weekly outliers analysis:")
    print(f"Average weekly commits: {average_commits:,.0f}")
    print(f"Threshold ({threshold_percentage}% above avg): {threshold:,.0f}")
    print(f"Removed {len(removed_weeks)} outlier weeks")
    print(f"Kept {len(filtered_data)} normal weeks")
    
    if removed_weeks:
        print("Removed weeks:")
        for week, commits in sorted(removed_weeks, key=lambda x: x[1], reverse=True)[:5]:
            percentage = (commits / average_commits - 1) * 100
            print(f"   {week}: {commits:,} commits (+{percentage:.0f}%)")
    
    return filtered_data, removed_weeks

def identify_power_users_by_month(commits_data, monthly_threshold=200):
    """Identify users who had more than threshold commits in ANY single month during May-Aug."""
    power_users = set()
    user_monthly_stats = {}
    
    # Define May-August boundaries
    may_start = datetime(2024, 5, 1)
    aug_end = datetime(2024, 9, 4)
    
    for username, user_data in commits_data.items():
        daily_commits = user_data.get('daily_commits', {})
        monthly_commits = defaultdict(int)
        
        # Group commits by month (May-Aug only)
        for date_str, commit_count in daily_commits.items():
            if commit_count > 0:
                try:
                    month_key = datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y-%m')
                    monthly_commits[month_key] += commit_count
                except ValueError:
                    # Skip malformed dates
                    continue
        
        # Check if any month exceeds threshold
        max_monthly = max(monthly_commits.values()) if monthly_commits else 0
        user_monthly_stats[username] = {
            'max_monthly_commits': max_monthly,
            'monthly_breakdown': dict(monthly_commits),
            'total_commits': sum(monthly_commits.values())
        }
        
        if max_monthly > monthly_threshold:
            power_users.add(username)
    
    print(f"\nMonthly power users analysis (May-Aug):")
    print(f"Users with >{monthly_threshold} commits in any single month: {len(power_users)}")
    print(f"Regular users to include: {len(commits_data) - len(power_users)}")
    
    # Show top power users
    if power_users:
        top_power_users = sorted(
            [(user, stats['max_monthly_commits']) for user, stats in user_monthly_stats.items() if user in power_users],
            key=lambda x: x[1], reverse=True
        )[:5]
        print("Top power users by max monthly commits:")
        for user, max_commits in top_power_users:
            print(f"   {user}: {max_commits:,} commits (max month)")
    
    return power_users, user_monthly_stats

def collapse_to_weekly_without_power_users(commits_data, power_users):
    """Collapse to weekly data excluding power users (May-Aug only)."""
    weekly_commits = defaultdict(int)
    
    # Define May-August boundaries
    may_start = datetime(2024, 5, 1)
    aug_end = datetime(2024, 9, 4)
    
    for username, user_data in commits_data.items():
        if username in power_users:
            continue  # Skip power users
            
        daily_commits = user_data.get('daily_commits', {})
        
        for date_str, commit_count in daily_commits.items():
            if commit_count > 0:
                week_start = get_week_start_date(date_str)
                weekly_commits[week_start] += commit_count
    
    return dict(weekly_commits)

def plot_weekly_without_outliers(filtered_weekly_data, removed_weeks):
    """Plot weekly commits without outlier weeks."""
    if not filtered_weekly_data:
        print("No data available for plotting after filtering outliers")
        return
    
    # Sort dates for proper plotting
    sorted_weeks = sorted(filtered_weekly_data.items())
    dates = [item[0] for item in sorted_weeks]
    commits = [item[1] for item in sorted_weeks]
    
    # Convert date strings to datetime objects
    date_objects = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
    
    plt.figure(figsize=(15, 8))
    plt.plot(date_objects, commits, marker='o', linewidth=2, markersize=4, color='steelblue')
    
    # Add treatment period shading
    treatment_start = datetime(2024, 7, 17)
    treatment_end = datetime(2024, 7, 24)
    plt.axvspan(treatment_start, treatment_end, alpha=0.3, color='gray', label='Apagón Bangladesh')
    
    plt.title('Weekly Commits - bangladesh (May-Aug 2024)\nOutlier Weeks >50% Removed', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Week Starting Date', fontsize=12)
    plt.ylabel('Total Commits', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Add average line
    avg_commits = sum(commits) / len(commits)
    plt.axhline(y=avg_commits, color='red', linestyle='--', linewidth=2, 
                label=f'Average: {avg_commits:,.0f}')
    
    # Format y-axis
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('bangladesh/weekly_commits_no_outliers_may_aug_2024.png', dpi=300, bbox_inches='tight')
    plt.savefig('bangladesh/weekly_commits_no_outliers_may_aug_2024.pdf', bbox_inches='tight')
    
    print(f"\nWeekly plot (no outliers) saved as 'bangladesh/weekly_commits_no_outliers_may_aug_2024.png/.pdf'")
    
    # Print some stats
    print(f"Plot statistics:")
    print(f"Weeks included: {len(commits)}")
    print(f"Average weekly commits (after removing outliers): {avg_commits:,.0f}")
    print(f"Peak week: {max(commits):,} commits")
    print(f"Lowest week: {min(commits):,} commits")

def plot_weekly_without_power_users(weekly_data_no_power, power_users_count):
    """Plot weekly commits without monthly power users."""
    if not weekly_data_no_power:
        print("No data available for plotting after removing power users")
        return
    
    # Filter complete weeks for consistency
    filtered_data = filter_complete_weeks(weekly_data_no_power)
    
    if not filtered_data:
        print("No complete weeks available after removing power users")
        return
    
    # Sort dates for proper plotting
    sorted_weeks = sorted(filtered_data.items())
    dates = [item[0] for item in sorted_weeks]
    commits = [item[1] for item in sorted_weeks]
    
    # Convert date strings to datetime objects
    date_objects = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
    
    plt.figure(figsize=(15, 8))
    plt.plot(date_objects, commits, marker='o', linewidth=2, markersize=4, color='forestgreen')
    
    # Add treatment period shading
    treatment_start = datetime(2024, 7, 17)
    treatment_end = datetime(2024, 7, 24)
    plt.axvspan(treatment_start, treatment_end, alpha=0.3, color='gray', label='Apagón Bangladesh')
    
    plt.title(f'Weekly Commits - bangladesh (May-Aug 2024)\nUsers with >200 monthly commits removed ({power_users_count} power users excluded)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Week Starting Date', fontsize=12)
    plt.ylabel('Total Commits', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Add average line
    avg_commits = sum(commits) / len(commits)
    plt.axhline(y=avg_commits, color='red', linestyle='--', linewidth=2, 
                label=f'Average: {avg_commits:,.0f}')
    
    # Format y-axis
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('bangladesh/weekly_commits_no_power_users_may_aug_2024.png', dpi=300, bbox_inches='tight')
    plt.savefig('bangladesh/weekly_commits_no_power_users_may_aug_2024.pdf', bbox_inches='tight')
    
    print(f"\nWeekly plot (no power users) saved as 'bangladesh/weekly_commits_no_power_users_may_aug_2024.png/.pdf'")
    
    # Print some stats
    print(f"Plot statistics:")
    print(f"Weeks included: {len(commits)}")
    print(f"Average weekly commits (without power users): {avg_commits:,.0f}")
    print(f"Peak week: {max(commits):,} commits")
    print(f"Lowest week: {min(commits):,} commits")

def save_analysis_summary(original_weekly_data, filtered_weekly_data, removed_weeks, weekly_data_no_power, power_users):
    """Save analysis summary to JSON."""
    
    # Calculate averages safely
    original_avg = sum(original_weekly_data.values()) / len(original_weekly_data) if original_weekly_data else 0
    filtered_avg = sum(filtered_weekly_data.values()) / len(filtered_weekly_data) if filtered_weekly_data else 0
    no_power_avg = sum(weekly_data_no_power.values()) / len(weekly_data_no_power) if weekly_data_no_power else 0
    
    output_data = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'period': 'May-August 2024',
        'note': 'Analysis for May-August period only',
        'outlier_weeks_analysis': {
            'total_weeks_original': len(original_weekly_data),
            'outlier_weeks_removed': len(removed_weeks),
            'normal_weeks_kept': len(filtered_weekly_data),
            'original_average_commits': round(original_avg, 2),
            'average_commits_after_removal': round(filtered_avg, 2),
            'percentage_change': round(((filtered_avg/original_avg-1)*100), 1) if original_avg > 0 else 0,
            'removed_weeks_details': [{'week': week, 'commits': commits, 'percentage_above_avg': round((commits/original_avg-1)*100, 1)} 
                                    for week, commits in sorted(removed_weeks, key=lambda x: x[1], reverse=True)[:10]]
        },
        'power_users_analysis': {
            'total_power_users_removed': len(power_users),
            'total_weeks_without_power_users': len(weekly_data_no_power),
            'original_average_commits': round(original_avg, 2),
            'average_commits_without_power_users': round(no_power_avg, 2),
            'percentage_change': round(((no_power_avg/original_avg-1)*100), 1) if original_avg > 0 else 0
        }
    }
    
    with open('bangladesh/without_outliers_analysis_may_aug_2024.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("\nAnalysis summary saved as 'bangladesh/without_outliers_analysis_may_aug_2024.json'")

def main():
    try:
        print("Loading commits data...")
        commits_data = load_commits_data('bangladesh/commits.json')
        
        if not commits_data:
            print("❌ No commits data found!")
            return
        
        print(f"Loaded data for {len(commits_data)} users")
        
        print("Processing weekly data (May-August only)...")
        weekly_data = collapse_to_weekly(commits_data)
        
        if not weekly_data:
            print("❌ No weekly data generated!")
            return
        
        print(f"Generated weekly data for {len(weekly_data)} weeks")
        
        # Filter to May-August weeks
        print("Filtering to May-August period...")
        complete_weekly_data = filter_complete_weeks(weekly_data)
        
        if not complete_weekly_data:
            print("❌ No weeks available in May-August period!")
            return
        
        # GRAPH 1: Remove outlier weeks (>50% above average)
        print("\n" + "="*60)
        print("GRAPH 1: REMOVING OUTLIER WEEKS (>50% ABOVE AVERAGE)")
        print("="*60)
        
        filtered_weekly_data, removed_weeks = remove_weekly_outliers(complete_weekly_data, threshold_percentage=50)
        
        if filtered_weekly_data:
            plot_weekly_without_outliers(filtered_weekly_data, removed_weeks)
        else:
            print("❌ No weeks remain after removing outliers!")
        
        # GRAPH 2: Remove users with >200 commits in any single month
        print("\n" + "="*60)
        print("GRAPH 2: REMOVING USERS WITH >200 COMMITS IN ANY MONTH (MAY-AUG)")
        print("="*60)
        
        power_users, user_monthly_stats = identify_power_users_by_month(commits_data, monthly_threshold=200)
        weekly_data_no_power = collapse_to_weekly_without_power_users(commits_data, power_users)
        
        if weekly_data_no_power:
            plot_weekly_without_power_users(weekly_data_no_power, len(power_users))
        else:
            print("❌ No data remains after removing power users!")
        
        # Save analysis summary
        save_analysis_summary(complete_weekly_data, filtered_weekly_data, removed_weeks, weekly_data_no_power, power_users)
        
        # Summary comparison
        print("\n" + "="*60)
        print("SUMMARY COMPARISON (MAY-AUGUST)")
        print("="*60)
        
        original_avg = sum(complete_weekly_data.values()) / len(complete_weekly_data)
        filtered_avg = sum(filtered_weekly_data.values()) / len(filtered_weekly_data) if filtered_weekly_data else 0
        no_power_avg = sum(weekly_data_no_power.values()) / len(weekly_data_no_power) if weekly_data_no_power else 0
        
        print(f"Original average weekly commits (May-Aug): {original_avg:,.0f}")
        if filtered_avg > 0:
            print(f"After removing outlier weeks: {filtered_avg:,.0f} ({((filtered_avg/original_avg-1)*100):+.1f}%)")
        if no_power_avg > 0:
            print(f"After removing power users: {no_power_avg:,.0f} ({((no_power_avg/original_avg-1)*100):+.1f}%)")
        
        print("\n✅ Analysis without outliers complete!")
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON format: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()