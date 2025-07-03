import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict, Counter

def load_commits_data(file_path):
    """Load commits data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def get_week_start_date(date_str):
    """Get the start date of the week using May 1, 2024 as reference point."""
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        reference_date = datetime(2024, 5, 1)  # Changed to May 1
        days_diff = (date - reference_date).days
        week_number = days_diff // 7
        week_start = reference_date + timedelta(days=week_number * 7)
        return week_start.strftime('%Y-%m-%d')
    except ValueError:
        return None

def filter_complete_weeks_may_aug(all_weeks):
    """Filter weeks to May-August period only."""
    may_start = datetime(2024, 5, 1)
    aug_end = datetime(2024, 9, 4)
    
    filtered_weeks = []
    for week in sorted(all_weeks):
        week_date = datetime.strptime(week, '%Y-%m-%d')
        if may_start <= week_date <= aug_end:
            filtered_weeks.append(week)
    
    return filtered_weeks

def filter_consistent_users(commits_data, pre_treatment_weeks):
    """Filter users who have at least 1 commit in ALL pre-treatment weeks."""
    consistent_users = []
    
    for username, user_data in commits_data.items():
        daily_commits = user_data.get('daily_commits', {})
        user_weekly_commits = defaultdict(int)
        
        # Aggregate weekly commits
        for date_str, commit_count in daily_commits.items():
            if commit_count > 0:
                week_start = get_week_start_date(date_str)
                if week_start and week_start in pre_treatment_weeks:
                    user_weekly_commits[week_start] += commit_count
        
        # Check if user has commits in ALL pre-treatment weeks
        if len(user_weekly_commits) == len(pre_treatment_weeks) and all(v > 0 for v in user_weekly_commits.values()):
            consistent_users.append(username)
    
    return consistent_users

def create_user_week_matrix(commits_data, consistent_users_only=True):
    """Create user-week matrix with all user-week observations."""
    user_week_data = []
    all_weeks = set()
    
    print("Creating user-week observations...")
    
    # Define treatment start
    treatment_start = datetime(2024, 7, 17)
    
    for username, user_data in commits_data.items():
        daily_commits = user_data.get('daily_commits', {})
        user_weekly_commits = defaultdict(int)
        
        for date_str, commit_count in daily_commits.items():
            if commit_count > 0:
                week_start = get_week_start_date(date_str)
                if week_start:
                    user_weekly_commits[week_start] += commit_count
                    all_weeks.add(week_start)
        
        # Add user-week observations
        for week in user_weekly_commits:
            user_week_data.append({
                'user': username,
                'week': week,
                'commits': user_weekly_commits[week]
            })
    
    # Filter to May-August weeks
    complete_weeks = filter_complete_weeks_may_aug(all_weeks)
    
    # Define pre-treatment weeks
    pre_treatment_weeks = [w for w in complete_weeks if datetime.strptime(w, '%Y-%m-%d') < treatment_start]
    
    if consistent_users_only:
        # Filter to only consistent users
        consistent_users = filter_consistent_users(commits_data, pre_treatment_weeks)
        filtered_data = [obs for obs in user_week_data if obs['user'] in consistent_users and obs['week'] in complete_weeks]
        print(f"Filtered to {len(consistent_users)} consistent users (had commits in all {len(pre_treatment_weeks)} pre-treatment weeks)")
    else:
        filtered_data = [obs for obs in user_week_data if obs['week'] in complete_weeks]
    
    print(f"Created {len(filtered_data)} user-week observations across {len(complete_weeks)} weeks (May-Aug)")
    return filtered_data, complete_weeks, len(consistent_users) if consistent_users_only else len(commits_data)

def classify_users_by_activity(commits_data):
    """Classify users into activity categories."""
    inactive, casual, power_users, regular = [], [], [], []
    
    for username, user_data in commits_data.items():
        total_commits = user_data.get('total_commits', 0)
        avg_commits_per_month = total_commits / 12
        
        if total_commits == 0:
            inactive.append(username)
        elif avg_commits_per_month <= 3:
            casual.append(username)
        elif avg_commits_per_month >= 200:
            power_users.append(username)
        else:
            regular.append(username)
    
    return {
        'inactive': inactive,
        'casual': casual, 
        'power_users': power_users,
        'regular': regular
    }

def remove_top_5_percent_outliers(user_week_data):
    """Remove user-week observations in the top 5% of weekly commits."""
    if not user_week_data:
        return [], []
    
    commits_values = [obs['commits'] for obs in user_week_data]
    p95_threshold = np.percentile(commits_values, 95)
    
    # Separate outliers and clean data
    outliers = [obs for obs in user_week_data if obs['commits'] > p95_threshold]
    clean_data = [obs for obs in user_week_data if obs['commits'] <= p95_threshold]
    
    print(f"P95 threshold: {p95_threshold:.1f} commits")
    print(f"Removed {len(outliers)} outlier observations (top 5%)")
    print(f"Kept {len(clean_data)} normal observations")
    
    return clean_data, outliers

def analyze_user_week_statistics(user_week_data, data_label=""):
    """Calculate comprehensive user-week statistics."""
    if not user_week_data:
        return {}
    
    commits_list = [obs['commits'] for obs in user_week_data]
    
    stats = {
        'total_observations': len(user_week_data),
        'unique_users': len(set(obs['user'] for obs in user_week_data)),
        'unique_weeks': len(set(obs['week'] for obs in user_week_data)),
        'total_commits': sum(commits_list),
        'mean_commits': np.mean(commits_list),
        'std_commits': np.std(commits_list),
        'median_commits': np.median(commits_list),
        'percentiles': {
            'p10': np.percentile(commits_list, 10),
            'p25': np.percentile(commits_list, 25),
            'p50': np.percentile(commits_list, 50),
            'p75': np.percentile(commits_list, 75),
            'p90': np.percentile(commits_list, 90),
            'p95': np.percentile(commits_list, 95),
            'p99': np.percentile(commits_list, 99)
        },
        'zero_commit_weeks': sum(1 for c in commits_list if c == 0),
        'active_weeks': sum(1 for c in commits_list if c > 0)
    }
    
    stats['zero_commit_percentage'] = (stats['zero_commit_weeks'] / stats['total_observations']) * 100
    stats['active_percentage'] = (stats['active_weeks'] / stats['total_observations']) * 100
    
    return stats

def analyze_temporal_patterns_user_week(user_week_data):
    """Analyze temporal patterns at user-week level."""
    if not user_week_data:
        return {}, {}
    
    # Group by quarter and month
    quarterly_activity = defaultdict(list)
    monthly_activity = defaultdict(list)
    
    for obs in user_week_data:
        if obs['commits'] > 0:  # Only count active weeks
            try:
                date = datetime.strptime(obs['week'], '%Y-%m-%d')
                month = date.month
                quarter = (month - 1) // 3 + 1
                
                quarterly_activity[quarter].append(obs['commits'])
                monthly_activity[month].append(obs['commits'])
            except ValueError:
                continue
    
    quarterly_stats = {}
    for quarter, commits_list in quarterly_activity.items():
        if commits_list:
            quarterly_stats[quarter] = {
                'active_weeks': len(commits_list),
                'total_commits': sum(commits_list),
                'mean_commits': np.mean(commits_list),
                'median_commits': np.median(commits_list)
            }
    
    monthly_stats = {}
    for month, commits_list in monthly_activity.items():
        if commits_list:
            monthly_stats[month] = {
                'active_weeks': len(commits_list),
                'total_commits': sum(commits_list),
                'mean_commits': np.mean(commits_list),
                'median_commits': np.median(commits_list)
            }
    
    return quarterly_stats, monthly_stats

def analyze_users_by_activity_frequency(user_week_data, complete_weeks):
    """Analyze users by how frequently they are active."""
    if not user_week_data:
        return {}
    
    # Count active weeks per user
    user_activity = defaultdict(int)
    total_weeks = len(complete_weeks)
    
    for obs in user_week_data:
        if obs['commits'] > 0:
            user_activity[obs['user']] += 1
    
    # Classify users by activity frequency
    never_active = 0
    rarely_active = 0  # 1-5 weeks
    sometimes_active = 0  # 6-10 weeks
    often_active = 0  # 11-15 weeks
    very_active = 0  # 16+ weeks
    
    for user, active_weeks in user_activity.items():
        if active_weeks == 0:
            never_active += 1
        elif active_weeks <= 5:
            rarely_active += 1
        elif active_weeks <= 10:
            sometimes_active += 1
        elif active_weeks <= 15:
            often_active += 1
        else:
            very_active += 1
    
    # Count users with no observations (completely inactive)
    all_users_in_data = set(obs['user'] for obs in user_week_data)
    
    return {
        'total_users_with_data': len(all_users_in_data),
        'never_active': never_active,
        'rarely_active': rarely_active,
        'sometimes_active': sometimes_active,
        'often_active': often_active,
        'very_active': very_active,
        'total_weeks_possible': total_weeks
    }

def compare_with_without_outliers(original_data, clean_data, outliers):
    """Compare statistics before and after removing outliers."""
    original_stats = analyze_user_week_statistics(original_data, "Original")
    clean_stats = analyze_user_week_statistics(clean_data, "Clean")
    
    print(f"\n🔍 COMPARISON: ORIGINAL vs CLEAN DATA")
    print(f"{'Metric':<25} {'Original':<15} {'Clean':<15} {'Change':<15}")
    print(f"{'-'*70}")
    
    metrics = [
        ('Total Observations', 'total_observations'),
        ('Mean Commits', 'mean_commits'),
        ('Median Commits', 'median_commits'),
        ('Std Commits', 'std_commits'),
        ('P95 Commits', lambda x: x['percentiles']['p95']),
        ('P99 Commits', lambda x: x['percentiles']['p99']),
        ('Active %', 'active_percentage')
    ]
    
    for label, key in metrics:
        if callable(key):
            orig_val = key(original_stats)
            clean_val = key(clean_stats)
        else:
            orig_val = original_stats[key]
            clean_val = clean_stats[key]
        
        if orig_val != 0:
            change_pct = ((clean_val - orig_val) / orig_val) * 100
            print(f"{label:<25} {orig_val:<15.1f} {clean_val:<15.1f} {change_pct:<15.1f}%")
        else:
            print(f"{label:<25} {orig_val:<15.1f} {clean_val:<15.1f} {'N/A':<15}")
    
    print(f"\nOutliers removed: {len(outliers)} observations")
    if outliers:
        outlier_commits = [obs['commits'] for obs in outliers]
        print(f"Outlier range: {min(outlier_commits)} - {max(outlier_commits)} commits")
        print(f"Top 5 outliers: {sorted(outlier_commits, reverse=True)[:5]}")

def analyze_country_user_week(commits_data, country_name):
    """Comprehensive user-week analysis for a single country."""
    print(f"\n{'='*80}")
    print(f"USER-WEEK ANALYSIS - {country_name.upper()} (MAY-AUGUST 2024)")
    print(f"{'='*80}")
    
    # Create user-week matrix with consistent users only
    user_week_data, complete_weeks, total_consistent_users = create_user_week_matrix(commits_data, consistent_users_only=True)
    
    if not user_week_data:
        print("❌ No user-week data generated!")
        return None
    
    # Remove top 5% outliers
    clean_data, outliers = remove_top_5_percent_outliers(user_week_data)
    
    # Basic statistics
    print(f"\n📊 BASIC USER-WEEK STATISTICS:")
    print(f"Total users in dataset: {len(commits_data):,}")
    print(f"Consistent users (commits every pre-treatment week): {total_consistent_users:,}")
    print(f"Weeks analyzed (May-Aug): {len(complete_weeks)}")
    print(f"Total user-week observations: {len(user_week_data):,}")
    print(f"After removing top 5% outliers: {len(clean_data):,}")
    
    # Analyze clean data statistics
    clean_stats = analyze_user_week_statistics(clean_data, "Clean")
    print(f"\n📈 USER-WEEK STATISTICS (TOP 5% OUTLIERS REMOVED):")
    print(f"Mean commits per user-week: {clean_stats['mean_commits']:.3f}")
    print(f"Median commits per user-week: {clean_stats['median_commits']:.3f}")
    print(f"Standard deviation: {clean_stats['std_commits']:.3f}")
    print(f"Zero-commit weeks: {clean_stats['zero_commit_percentage']:.1f}%")
    print(f"Active weeks: {clean_stats['active_percentage']:.1f}%")
    
    print(f"\nPercentile distribution:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = clean_stats['percentiles'][f'p{p}']
        print(f"  P{p}: {val:.1f} commits")
    
    # Activity frequency analysis
    activity_freq = analyze_users_by_activity_frequency(clean_data, complete_weeks)
    print(f"\n📅 USER ACTIVITY FREQUENCY (WEEKS ACTIVE):")
    total_analyzed = activity_freq['total_users_with_data']
    print(f"Rarely active (1-5 weeks): {activity_freq['rarely_active']:,} ({activity_freq['rarely_active']/total_analyzed*100:.1f}%)")
    print(f"Sometimes active (6-10 weeks): {activity_freq['sometimes_active']:,} ({activity_freq['sometimes_active']/total_analyzed*100:.1f}%)")
    print(f"Often active (11-15 weeks): {activity_freq['often_active']:,} ({activity_freq['often_active']/total_analyzed*100:.1f}%)")
    print(f"Very active (16+ weeks): {activity_freq['very_active']:,} ({activity_freq['very_active']/total_analyzed*100:.1f}%)")
    
    # Temporal patterns
    quarterly_stats, monthly_stats = analyze_temporal_patterns_user_week(clean_data)
    print(f"\n🕒 TEMPORAL PATTERNS (ACTIVE USER-WEEKS ONLY):")
    print(f"Monthly activity:")
    if monthly_stats:
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month in [5, 6, 7, 8]:  # May to August
            if month in monthly_stats:
                stats = monthly_stats[month]
                print(f"  {month_names[month-1]}: {stats['active_weeks']:,} active user-weeks, {stats['mean_commits']:.1f} avg commits")
    
    # Compare original vs clean
    compare_with_without_outliers(user_week_data, clean_data, outliers)
    
    return {
        'country': country_name,
        'user_week_data': user_week_data,
        'clean_data': clean_data,
        'outliers': outliers,
        'complete_weeks': complete_weeks,
        'clean_stats': clean_stats,
        'total_consistent_users': total_consistent_users,
        'activity_freq': activity_freq,
        'temporal': {'quarterly': quarterly_stats, 'monthly': monthly_stats}
    }

def compare_countries_user_week(bangladesh_results, philippines_results):
    """Generate comparative analysis between countries."""
    print(f"\n{'='*80}")
    print(f"COMPARATIVE USER-WEEK ANALYSIS - BANGLADESH vs PHILIPPINES (MAY-AUG 2024)")
    print(f"{'='*80}")
    
    bg_stats = bangladesh_results['clean_stats']
    ph_stats = philippines_results['clean_stats']
    
    print(f"\n📊 BASIC USER-WEEK COMPARISON:")
    print(f"{'Metric':<35} {'Bangladesh':<15} {'Philippines':<15} {'Ratio':<10}")
    print(f"{'-'*75}")
    
    # Add consistent users comparison
    print(f"{'Consistent Users':<35} {bangladesh_results['total_consistent_users']:<15,} {philippines_results['total_consistent_users']:<15,} {bangladesh_results['total_consistent_users']/philippines_results['total_consistent_users'] if philippines_results['total_consistent_users'] != 0 else float('inf'):<10.2f}")
    
    comparisons = [
        ('User-Week Observations', 'total_observations'),
        ('Unique Users', 'unique_users'),
        ('Weeks Analyzed', 'unique_weeks'),
        ('Mean Commits/User-Week', 'mean_commits'),
        ('Median Commits/User-Week', 'median_commits'),
        ('Std Commits/User-Week', 'std_commits'),
        ('Active Weeks %', 'active_percentage')
    ]
    
    for label, key in comparisons:
        bg_val = bg_stats[key]
        ph_val = ph_stats[key]
        ratio = bg_val / ph_val if ph_val != 0 else float('inf')
        print(f"{label:<35} {bg_val:<15.3f} {ph_val:<15.3f} {ratio:<10.2f}")
    
    print(f"\n📈 PERCENTILE COMPARISON:")
    print(f"{'Percentile':<15} {'Bangladesh':<15} {'Philippines':<15} {'Ratio':<10}")
    print(f"{'-'*55}")
    
    for p in [25, 50, 75, 90, 95]:
        bg_val = bg_stats['percentiles'][f'p{p}']
        ph_val = ph_stats['percentiles'][f'p{p}']
        ratio = bg_val / ph_val if ph_val != 0 else float('inf')
        print(f"P{p:<14} {bg_val:<15.1f} {ph_val:<15.1f} {ratio:<10.2f}")

def save_user_week_summary(bangladesh_results, philippines_results):
    """Save user-week analysis summary to JSON."""
    summary_data = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'methodology': 'user_week_level_analysis_may_august_consistent_users_p95_outlier_removal',
        'period': 'May-August 2024',
        'criteria': 'Users with at least 1 commit in ALL pre-treatment weeks',
        'bangladesh': {
            'clean_stats': bangladesh_results['clean_stats'],
            'outliers_removed': len(bangladesh_results['outliers']),
            'total_consistent_users': bangladesh_results['total_consistent_users'],
            'activity_frequency': bangladesh_results['activity_freq'],
            'temporal_patterns': bangladesh_results['temporal']
        },
        'philippines': {
            'clean_stats': philippines_results['clean_stats'],
            'outliers_removed': len(philippines_results['outliers']),
            'total_consistent_users': philippines_results['total_consistent_users'],
            'activity_frequency': philippines_results['activity_freq'],
            'temporal_patterns': philippines_results['temporal']
        }
    }
    
    with open('user_week_summary_may_aug_consistent.json', 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n💾 User-week summary data saved as 'user_week_summary_may_aug_consistent.json'")

def main():
    try:
        print("Loading Bangladesh data...")
        bangladesh_data = load_commits_data('main/bangladesh/commits.json')
        bangladesh_results = analyze_country_user_week(bangladesh_data, 'Bangladesh')
        
        print("\nLoading Philippines data...")
        philippines_data = load_commits_data('philippines/commits.json')
        philippines_results = analyze_country_user_week(philippines_data, 'Philippines')
        
        if bangladesh_results and philippines_results:
            # Comparative analysis
            compare_countries_user_week(bangladesh_results, philippines_results)
            
            # Save summary data
            save_user_week_summary(bangladesh_results, philippines_results)
        
        print(f"\n✅ User-week analysis complete!")
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Make sure you have the commits.json files in bangladesh/ and philippines/ directories")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()