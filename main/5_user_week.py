import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

def load_commits_data(file_path):
    """Load commits data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def get_week_start_date(date_str):
    """Get the start date of the week using July 17, 2024 as reference point."""
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        
        # July 17, 2024 is our reference start date for week grouping
        reference_date = datetime(2024, 7, 17)  # ← Cambiar aquí
        
        # Calculate how many days have passed since July 17
        days_diff = (date - reference_date).days
        
        # Group into 7-day weeks starting from July 17
        week_number = days_diff // 7
        week_start = reference_date + timedelta(days=week_number * 7)
        
        return week_start.strftime('%Y-%m-%d')
    except ValueError:
        return None

def is_within_may_august(date_str):
    """Check if date is within May-August 2024 period."""
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        may_start = datetime(2024, 5, 1)
        aug_end = datetime(2024,  9, 4)
        return may_start <= date <= aug_end
    except ValueError:
        return False

def filter_complete_weeks(all_weeks):
    """Filter to only include complete weeks within May-August period."""
    if not all_weeks:
        return all_weeks
    
    # May-August boundaries
    may_start = datetime(2024, 5, 1)
    aug_end = datetime(2024,  9, 4)
    
    filtered_weeks = []
    for week in all_weeks:
        try:
            week_date = datetime.strptime(week, '%Y-%m-%d')
            # Include week if it starts within our period
            if may_start <= week_date <= aug_end:
                filtered_weeks.append(week)
        except ValueError:
            continue
    
    sorted_weeks = sorted(filtered_weeks)
    
    print(f"Filtered to {len(sorted_weeks)} complete weeks in May-August 2024")
    if sorted_weeks:
        print(f"Week range: {sorted_weeks[0]} to {sorted_weeks[-1]}")
    
    return sorted_weeks

def create_user_week_matrix(commits_data):
    """Create user-week matrix for May-August period only."""
    user_week_data = defaultdict(lambda: defaultdict(int))
    all_weeks = set()
    
    print("Creating user-week matrix for May-August 2024...")
    
    # Define May-August boundaries
    may_start = datetime(2024, 5, 1)
    aug_end = datetime(2024, 9, 4)
    
    processed_users = 0
    for username, user_data in commits_data.items():
        processed_users += 1
        if processed_users % 10000 == 0:  # Progress indicator
            print(f"Processed {processed_users}/{len(commits_data)} users...")
        
        daily_commits = user_data.get('daily_commits', {})
        
        for date_str, commit_count in daily_commits.items():
            if commit_count > 0:  # Only process days with commits
                week_start = get_week_start_date(date_str)
                if week_start:  # Only if date parsing succeeded
                    user_week_data[username][week_start] += commit_count
                    all_weeks.add(week_start)
    
    # Filter to complete weeks within May-August
    complete_weeks = filter_complete_weeks(all_weeks)
    
    print(f"Created matrix: {len(user_week_data)} users × {len(complete_weeks)} complete weeks (May-Aug)")
    return user_week_data, complete_weeks

def calculate_weekly_averages(user_week_data, sorted_weeks):
    """Calculate average commits per user for each week."""
    if not user_week_data or not sorted_weeks:
        return {}
    
    weekly_averages = {}
    total_users = len(user_week_data)
    
    for week in sorted_weeks:
        # Get all user commits for this week (including 0s for users who didn't commit)
        week_commits = [user_week_data[user][week] for user in user_week_data.keys()]
        
        # Calculate average (including users with 0 commits)
        avg_commits = sum(week_commits) / total_users if total_users > 0 else 0
        weekly_averages[week] = avg_commits
    
    return weekly_averages

def remove_user_week_outliers(user_week_data, sorted_weeks, threshold_percentage=50):
    """Remove user-week outliers and recalculate weekly averages."""
    if not user_week_data or not sorted_weeks:
        return {}, []
    
    cleaned_data = defaultdict(lambda: defaultdict(int))
    removed_outliers = []
    total_users = len(user_week_data)
    
    print(f"Analyzing {total_users} users across {len(sorted_weeks)} weeks (May-Aug)...")
    
    # First pass: identify outliers
    for week in sorted_weeks:
        week_commits = [user_week_data[user][week] for user in user_week_data.keys()]
        week_average = sum(week_commits) / total_users if total_users > 0 else 0
        threshold = week_average * (1 + threshold_percentage / 100)
        
        for user in user_week_data.keys():
            commits = user_week_data[user][week]
            
            if commits <= threshold:
                cleaned_data[user][week] = commits
            else:
                removed_outliers.append({
                    'user': user,
                    'week': week,
                    'commits': commits,
                    'week_average': week_average,
                    'threshold': threshold
                })
    
    # Calculate new weekly averages without outliers
    cleaned_averages = {}
    for week in sorted_weeks:
        # Count all users (including those with 0 commits after outlier removal)
        week_commits = [cleaned_data[user][week] for user in user_week_data.keys()]
        cleaned_averages[week] = sum(week_commits) / total_users if total_users > 0 else 0
    
    print(f"User-week outliers analysis (May-Aug):")
    print(f"Removed {len(removed_outliers)} user-week outliers (>{threshold_percentage}% above weekly average)")
    print(f"Total observations: {total_users * len(sorted_weeks)}")
    print(f"Outlier rate: {len(removed_outliers)/(total_users * len(sorted_weeks))*100:.2f}%")
    
    return cleaned_averages, removed_outliers

def identify_monthly_power_users(commits_data, monthly_threshold=200):
    """Identify users who had >threshold commits in any single month during May-Aug."""
    power_users = set()
    power_user_stats = {}
    
    print(f"Analyzing monthly activity for {len(commits_data)} users (May-Aug)...")
    
    # Define May-August boundaries
    may_start = datetime(2024, 5, 1)
    aug_end = datetime(2024, 9, 4)
    
    for username, user_data in commits_data.items():
        daily_commits = user_data.get('daily_commits', {})
        monthly_commits = defaultdict(int)
        
        for date_str, commit_count in daily_commits.items():
            if commit_count > 0:
                try:
                    date = datetime.strptime(date_str, '%Y-%m-%d')
                    # Only consider May-August period
                    if may_start <= date <= aug_end:
                        month_key = date.strftime('%Y-%m')
                        monthly_commits[month_key] += commit_count
                except ValueError:
                    # Skip malformed dates
                    continue
        
        max_monthly = max(monthly_commits.values()) if monthly_commits else 0
        if max_monthly > monthly_threshold:
            power_users.add(username)
            power_user_stats[username] = {
                'max_monthly_commits': max_monthly,
                'total_commits': sum(monthly_commits.values())
            }
    
    print(f"Identified {len(power_users)} power users with >{monthly_threshold} commits in any month (May-Aug)")
    
    if power_user_stats:
        top_power_users = sorted(power_user_stats.items(), key=lambda x: x[1]['max_monthly_commits'], reverse=True)[:5]
        print("Top 5 power users by max monthly commits (May-Aug):")
        for username, stats in top_power_users:
            print(f"   {username}: {stats['max_monthly_commits']:,} commits (max month)")
    
    return power_users

def calculate_averages_without_power_users(user_week_data, sorted_weeks, power_users):
    """Calculate weekly averages excluding power users."""
    if not user_week_data or not sorted_weeks:
        return {}
    
    averages_no_power = {}
    
    # Get list of non-power users
    regular_users = [user for user in user_week_data.keys() if user not in power_users]
    total_regular_users = len(regular_users)
    
    if total_regular_users == 0:
        print("Warning: No regular users remain after removing power users!")
        return {week: 0 for week in sorted_weeks}
    
    for week in sorted_weeks:
        # Get commits only from non-power users
        week_commits = [user_week_data[user][week] for user in regular_users]
        avg_commits = sum(week_commits) / total_regular_users
        averages_no_power[week] = avg_commits
    
    print(f"Calculated averages for {total_regular_users} regular users (excluded {len(power_users)} power users)")
    return averages_no_power

def plot_user_week_analysis(weekly_averages, cleaned_averages, averages_no_power, sorted_weeks):
    """Create three plots for user-week analysis (May-August)."""
    
    if not weekly_averages or not sorted_weeks:
        print("No data available for plotting")
        return 0, 0, 0
    
    # Convert dates for plotting
    date_objects = [datetime.strptime(date, '%Y-%m-%d') for date in sorted_weeks]
    
    # Original averages
    original_values = [weekly_averages[week] for week in sorted_weeks]
    
    # Cleaned averages (without user-week outliers)
    cleaned_values = [cleaned_averages.get(week, 0) for week in sorted_weeks]
    
    # Without power users
    no_power_values = [averages_no_power.get(week, 0) for week in sorted_weeks]
    
    # Create three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 18))
    
    # Plot 1: Original averages
    ax1.plot(date_objects, original_values, marker='o', linewidth=2, markersize=4, color='steelblue')
    ax1.set_title('Average Commits per User per Week - Original Data (May-Aug 2024)\n(Complete weeks only)', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Average Commits per User', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add treatment period shading
    treatment_start = datetime(2024, 7, 17)
    treatment_end = datetime(2024, 7, 24)
    ax1.axvspan(treatment_start, treatment_end, alpha=0.3, color='gray', label='Apagón Bangladesh')
    
    # Add average line
    overall_avg = np.mean(original_values)
    ax1.axhline(y=overall_avg, color='red', linestyle='--', linewidth=2, 
                label=f'Overall Average: {overall_avg:.3f}')
    ax1.legend()
    
    # Plot 2: Without user-week outliers
    ax2.plot(date_objects, cleaned_values, marker='o', linewidth=2, markersize=4, color='forestgreen')
    ax2.set_title('Average Commits per User per Week - User-Week Outliers Removed (>50%) (May-Aug 2024)\n(Complete weeks only)', 
                  fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Commits per User', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add treatment period shading
    ax2.axvspan(treatment_start, treatment_end, alpha=0.3, color='gray', label='Apagón Bangladesh')
    
    # Add average line
    cleaned_avg = np.mean(cleaned_values)
    ax2.axhline(y=cleaned_avg, color='red', linestyle='--', linewidth=2, 
                label=f'Overall Average: {cleaned_avg:.3f}')
    ax2.legend()
    
    # Plot 3: Without power users
    ax3.plot(date_objects, no_power_values, marker='o', linewidth=2, markersize=4, color='darkorange')
    ax3.set_title('Average Commits per User per Week - Power Users Removed (>200 monthly) (May-Aug 2024)\n(Complete weeks only)', 
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('Week Starting Date', fontsize=12)
    ax3.set_ylabel('Average Commits per User', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # Add treatment period shading
    ax3.axvspan(treatment_start, treatment_end, alpha=0.3, color='gray', label='Apagón Bangladesh')
    
    # Add average line
    no_power_avg = np.mean(no_power_values)
    ax3.axhline(y=no_power_avg, color='red', linestyle='--', linewidth=2, 
                label=f'Overall Average: {no_power_avg:.3f}')
    ax3.legend()
    
    plt.tight_layout()
    
    # Save plots
    plt.savefig('bangladesh/user_week_analysis_may_aug_2024.png', dpi=300, bbox_inches='tight')
    plt.savefig('bangladesh/user_week_analysis_may_aug_2024.pdf', bbox_inches='tight')
    
    print("\nUser-week analysis plots saved as 'bangladesh/user_week_analysis_may_aug_2024.png/.pdf'")
    
    return overall_avg, cleaned_avg, no_power_avg

def save_user_week_analysis(weekly_averages, cleaned_averages, averages_no_power, 
                           removed_outliers, power_users, sorted_weeks):
    """Save user-week analysis results to JSON."""
    
    output_data = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'period': 'May-August 2024',
        'methodology': 'user-week_based_analysis',
        'note': 'Analysis for May-August 2024 period only',
        'summary_statistics': {
            'total_complete_weeks_analyzed': len(sorted_weeks),
            'original_average_commits_per_user_per_week': round(np.mean(list(weekly_averages.values())), 4) if weekly_averages else 0,
            'cleaned_average_commits_per_user_per_week': round(np.mean(list(cleaned_averages.values())), 4) if cleaned_averages else 0,
            'no_power_users_average_commits_per_user_per_week': round(np.mean(list(averages_no_power.values())), 4) if averages_no_power else 0,
            'user_week_outliers_removed': len(removed_outliers),
            'power_users_removed': len(power_users)
        },
        'weekly_data': {
            'original_averages': {week: round(avg, 4) for week, avg in weekly_averages.items()},
            'cleaned_averages': {week: round(avg, 4) for week, avg in cleaned_averages.items()},
            'no_power_users_averages': {week: round(avg, 4) for week, avg in averages_no_power.items()}
        },
        'top_outliers_removed': [
            {
                'user': outlier['user'],
                'week': outlier['week'],
                'commits': outlier['commits'],
                'week_average': round(outlier['week_average'], 3),
                'threshold': round(outlier['threshold'], 3),
                'percentage_above_avg': round((outlier['commits']/outlier['week_average'] - 1) * 100, 1) if outlier['week_average'] > 0 else 0
            }
            for outlier in sorted(removed_outliers, key=lambda x: x['commits'], reverse=True)[:20]
        ]
    }
    
    with open('bangladesh/user_week_analysis_may_aug_2024.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("User-week analysis results saved as 'bangladesh/user_week_analysis_may_aug_2024.json'")

def print_analysis_summary(weekly_averages, cleaned_averages, averages_no_power, 
                          removed_outliers, power_users, sorted_weeks):
    """Print comprehensive analysis summary."""
    
    print("=" * 80)
    print("USER-WEEK ANALYSIS - bangladesh GITHUB USERS (MAY-AUG 2024)")
    print("(Complete weeks only - May-August 2024 period)")
    print("=" * 80)
    
    if not weekly_averages:
        print("No data available for analysis")
        return
    
    original_avg = np.mean(list(weekly_averages.values()))
    cleaned_avg = np.mean(list(cleaned_averages.values())) if cleaned_averages else 0
    no_power_avg = np.mean(list(averages_no_power.values())) if averages_no_power else 0
    
    print(f"\nAnalysis Period: May-August 2024")
    print(f"Analysis Methodology: User-Week Based")
    print(f"Unit of Analysis: Individual user's commits in a specific week")
    print(f"Total complete weeks analyzed: {len(sorted_weeks)}")
    
    print(f"\nAVERAGE COMMITS PER USER PER WEEK (MAY-AUG):")
    print(f"Original data: {original_avg:.4f} commits/user/week")
    if cleaned_avg > 0:
        print(f"After removing user-week outliers: {cleaned_avg:.4f} commits/user/week ({((cleaned_avg/original_avg-1)*100):+.1f}%)")
    if no_power_avg > 0:
        print(f"After removing power users: {no_power_avg:.4f} commits/user/week ({((no_power_avg/original_avg-1)*100):+.1f}%)")
    
    print(f"\nOUTLIER ANALYSIS (MAY-AUG):")
    print(f"User-week outliers removed: {len(removed_outliers)}")
    print(f"Power users identified and removed: {len(power_users)}")
    
    if removed_outliers:
        print(f"\nTop 5 user-week outliers removed:")
        top_outliers = sorted(removed_outliers, key=lambda x: x['commits'], reverse=True)[:5]
        for outlier in top_outliers:
            percentage = (outlier['commits']/outlier['week_average'] - 1) * 100 if outlier['week_average'] > 0 else 0
            print(f"   {outlier['user']} (week {outlier['week']}): {outlier['commits']} commits "
                  f"(+{percentage:.0f}% above avg)")
    
    print(f"\nWEEKLY TRENDS (MAY-AUG):")
    max_week = max(weekly_averages, key=weekly_averages.get)
    min_week = min(weekly_averages, key=weekly_averages.get)
    print(f"Peak week: {max_week} with {weekly_averages[max_week]:.4f} avg commits/user")
    print(f"Lowest week: {min_week} with {weekly_averages[min_week]:.4f} avg commits/user")

def main():
    try:
        print("Loading commits data...")
        commits_data = load_commits_data('bangladesh/commits.json')
        
        if not commits_data:
            print("❌ No commits data found!")
            return
        
        print(f"Loaded data for {len(commits_data)} users")
        
        print("Creating user-week analysis for May-August 2024...")
        user_week_data, sorted_weeks = create_user_week_matrix(commits_data)
        
        if not user_week_data or not sorted_weeks:
            print("❌ No user-week data generated!")
            return
        
        # Calculate original weekly averages
        print("\nCalculating original weekly averages (May-Aug)...")
        weekly_averages = calculate_weekly_averages(user_week_data, sorted_weeks)
        
        if not weekly_averages:
            print("❌ No weekly averages calculated!")
            return
        
        # Remove user-week outliers
        print("\nRemoving user-week outliers (May-Aug)...")
        cleaned_averages, removed_outliers = remove_user_week_outliers(user_week_data, sorted_weeks, threshold_percentage=50)
        
        # Identify and remove power users
        print("\nIdentifying power users (May-Aug)...")
        power_users = identify_monthly_power_users(commits_data, monthly_threshold=200)
        
        print("\nCalculating averages without power users (May-Aug)...")
        averages_no_power = calculate_averages_without_power_users(user_week_data, sorted_weeks, power_users)
        
        # Create plots
        print("\nCreating user-week analysis plots (May-Aug)...")
        overall_avg, cleaned_avg, no_power_avg = plot_user_week_analysis(
            weekly_averages, cleaned_averages, averages_no_power, sorted_weeks
        )
        
        # Print summary
        print_analysis_summary(weekly_averages, cleaned_averages, averages_no_power, 
                              removed_outliers, power_users, sorted_weeks)
        
        # Save results
        save_user_week_analysis(weekly_averages, cleaned_averages, averages_no_power, 
                               removed_outliers, power_users, sorted_weeks)
        
        print("\n✅ User-week analysis (May-Aug) complete!")
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON format: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()