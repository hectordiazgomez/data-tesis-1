import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import defaultdict, Counter

def load_commits_data(file_path):
    """Load commits data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def filter_may_august_commits(daily_commits):
    """Filter daily commits to only May-August period."""
    may_start = datetime(2024, 5, 1)
    aug_end = datetime(2024, 9, 3)
    
    filtered_commits = {}
    for date_str, commit_count in daily_commits.items():
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            if may_start <= date <= aug_end:
                filtered_commits[date_str] = commit_count
        except ValueError:
            continue
    
    return filtered_commits

def calculate_total_commits_may_aug(daily_commits):
    """Calculate total commits for May-August period."""
    filtered = filter_may_august_commits(daily_commits)
    return sum(filtered.values())

def classify_users_by_activity(commits_data):
    """Classify users into activity categories based on May-Aug activity."""
    inactive = []  # 0 commits
    casual = []    # 1-3 commits/month average
    power_users = []  # 200+ commits/month average
    regular = []   # Everyone else
    
    for username, user_data in commits_data.items():
        daily_commits = user_data.get('daily_commits', {})
        total_commits = calculate_total_commits_may_aug(daily_commits)
        avg_commits_per_month = total_commits / 4  # 4 months: May-Aug
        
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

def calculate_consistency_score(daily_commits):
    """Calculate consistency score for May-Aug period only."""
    filtered_commits = filter_may_august_commits(daily_commits)
    
    if not filtered_commits:
        return 0
    
    commits_list = list(filtered_commits.values())
    if not commits_list or sum(commits_list) == 0:
        return 0
    
    # Only consider days with commits for consistency calculation
    active_days = [c for c in commits_list if c > 0]
    if len(active_days) < 2:
        return 0  # Need at least 2 active days to measure consistency
    
    # Calculate coefficient of variation (lower = more consistent)
    mean_commits = np.mean(active_days)
    if mean_commits == 0:
        return 0
    
    std_commits = np.std(active_days)
    cv = std_commits / mean_commits
    
    # Convert to consistency score (inverse of CV, normalized)
    consistency = 1 / (1 + cv)
    return consistency

def calculate_streak_analysis(daily_commits):
    """Calculate longest commit streak and longest break for May-Aug."""
    filtered_commits = filter_may_august_commits(daily_commits)
    
    if not filtered_commits:
        return {'longest_streak': 0, 'longest_break': 0}
    
    # Create date range for May-Aug
    start_date = datetime(2024, 5, 1)
    end_date = datetime(2024, 9, 3)
    
    current_date = start_date
    current_streak = 0
    longest_streak = 0
    current_break = 0
    longest_break = 0
    
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        commits = filtered_commits.get(date_str, 0)
        
        if commits > 0:
            current_streak += 1
            longest_streak = max(longest_streak, current_streak)
            current_break = 0
        else:
            current_break += 1
            longest_break = max(longest_break, current_break)
            current_streak = 0
        
        current_date += timedelta(days=1)
    
    return {
        'longest_streak': longest_streak,
        'longest_break': longest_break
    }

def analyze_seasonal_patterns(daily_commits):
    """Analyze commits by month for May-Aug."""
    monthly_commits = {5: 0, 6: 0, 7: 0, 8: 0}  # May-Aug
    
    filtered_commits = filter_may_august_commits(daily_commits)
    
    for date_str, commits in filtered_commits.items():
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            month = date.month
            if month in monthly_commits:
                monthly_commits[month] += commits
        except ValueError:
            continue
    
    # Only return most active month if user has commits
    total_commits = sum(monthly_commits.values())
    if total_commits == 0:
        return {
            'monthly_commits': monthly_commits,
            'most_active_month': None
        }
    
    # Find most active month
    most_active_month = max(monthly_commits, key=monthly_commits.get)
    
    return {
        'monthly_commits': monthly_commits,
        'most_active_month': most_active_month
    }

def analyze_weekday_patterns(daily_commits):
    """Analyze weekend vs weekday activity for May-Aug."""
    weekday_commits = 0
    weekend_commits = 0
    
    filtered_commits = filter_may_august_commits(daily_commits)
    
    for date_str, commits in filtered_commits.items():
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            if date.weekday() < 5:  # Monday = 0, Sunday = 6
                weekday_commits += commits
            else:
                weekend_commits += commits
        except ValueError:
            continue
    
    total = weekday_commits + weekend_commits
    if total == 0:
        return {
            'weekday_commits': 0,
            'weekend_commits': 0,
            'weekday_ratio': 0,
            'weekend_ratio': 0
        }
    
    return {
        'weekday_commits': weekday_commits,
        'weekend_commits': weekend_commits,
        'weekday_ratio': weekday_commits / total,
        'weekend_ratio': weekend_commits / total
    }

def calculate_distribution_stats(values):
    """Calculate percentiles and standard deviation stats."""
    if not values:
        return {
            'mean': 0, 'std': 0,
            'percentiles': {f'P{p}': 0 for p in [25, 50, 75, 90, 95, 99]},
            'std_boundaries': {'1_sigma': (0, 0), '2_sigma': (0, 0), '3_sigma': (0, 0)},
            'outlier_bounds': {'lower_bound': 0, 'upper_bound': 0}
        }
    
    values_array = np.array(values)
    mean_val = np.mean(values_array)
    std_val = np.std(values_array)
    
    # Percentiles
    percentiles = {
        'P25': np.percentile(values_array, 25),
        'P50': np.percentile(values_array, 50),  # median
        'P75': np.percentile(values_array, 75),
        'P90': np.percentile(values_array, 90),
        'P95': np.percentile(values_array, 95),
        'P99': np.percentile(values_array, 99)
    }
    
    # Standard deviation boundaries
    std_boundaries = {
        '1_sigma': (mean_val - std_val, mean_val + std_val),
        '2_sigma': (mean_val - 2*std_val, mean_val + 2*std_val),
        '3_sigma': (mean_val - 3*std_val, mean_val + 3*std_val)
    }
    
    # IQR outliers
    q1, q3 = percentiles['P25'], percentiles['P75']
    iqr = q3 - q1
    outlier_bounds = {
        'lower_bound': q1 - 1.5 * iqr,
        'upper_bound': q3 + 1.5 * iqr
    }
    
    return {
        'mean': mean_val,
        'std': std_val,
        'percentiles': percentiles,
        'std_boundaries': std_boundaries,
        'outlier_bounds': outlier_bounds
    }

def find_outliers_by_method(values, usernames, stats):
    """Find outliers using different methods."""
    outliers = {
        'iqr_outliers': [],
        'above_2_sigma': [],
        'above_3_sigma': [],
        'top_1_percent': [],
        'bottom_1_percent': []
    }
    
    if not values or not usernames or stats['mean'] == 0:
        return outliers
    
    for username, value in zip(usernames, values):
        # IQR outliers
        if value < stats['outlier_bounds']['lower_bound'] or value > stats['outlier_bounds']['upper_bound']:
            outliers['iqr_outliers'].append((username, value))
        
        # Standard deviation outliers
        if abs(value - stats['mean']) > 2 * stats['std']:
            outliers['above_2_sigma'].append((username, value))
        
        if abs(value - stats['mean']) > 3 * stats['std']:
            outliers['above_3_sigma'].append((username, value))
        
        # Percentile outliers
        if value >= stats['percentiles']['P99']:
            outliers['top_1_percent'].append((username, value))
        
        # Bottom 1% based on actual low values, not percentage of P25
        if value <= stats['percentiles']['P25'] * 0.25:  # Bottom quartile of bottom quartile
            outliers['bottom_1_percent'].append((username, value))
    
    return outliers

def analyze_individual_users(commits_data):
    """Comprehensive analysis of individual user patterns for May-Aug."""
    user_metrics = {}
    
    print(f"Analyzing individual user patterns for {len(commits_data)} users (May-Aug period)...")
    
    for i, (username, user_data) in enumerate(commits_data.items(), 1):
        if i % 10000 == 0:  # Progress indicator for large datasets
            print(f"Processed {i}/{len(commits_data)} users...")
        
        daily_commits = user_data.get('daily_commits', {})
        total_commits = calculate_total_commits_may_aug(daily_commits)
        
        # Basic metrics (for 4 months)
        avg_commits_per_day = total_commits / 123 if total_commits > 0 else 0  # 123 days in May-Aug
        avg_commits_per_week = total_commits / 17.5 if total_commits > 0 else 0  # ~17.5 weeks
        avg_commits_per_month = total_commits / 4 if total_commits > 0 else 0
        
        # Advanced analysis
        consistency = calculate_consistency_score(daily_commits)
        streaks = calculate_streak_analysis(daily_commits)
        seasonal = analyze_seasonal_patterns(daily_commits)
        weekday_pattern = analyze_weekday_patterns(daily_commits)
        
        user_metrics[username] = {
            'total_commits': total_commits,
            'avg_commits_per_day': avg_commits_per_day,
            'avg_commits_per_week': avg_commits_per_week,
            'avg_commits_per_month': avg_commits_per_month,
            'consistency_score': consistency,
            'longest_streak': streaks['longest_streak'],
            'longest_break': streaks['longest_break'],
            'most_active_month': seasonal['most_active_month'],
            'weekday_ratio': weekday_pattern['weekday_ratio'],
            'weekend_ratio': weekday_pattern['weekend_ratio']
        }
    
    return user_metrics

def create_distribution_plots(user_metrics):
    """Create distribution plots for key metrics."""
    if not user_metrics:
        print("No user metrics available for plotting")
        return
    
    metrics_to_plot = [
        ('total_commits', 'Total Commits (May-Aug)'),
        ('avg_commits_per_month', 'Average Commits per Month (May-Aug)'),
        ('consistency_score', 'Consistency Score (May-Aug)'),
        ('weekday_ratio', 'Weekday Commit Ratio (May-Aug)')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (metric, title) in enumerate(metrics_to_plot):
        values = [user_metrics[user][metric] for user in user_metrics]
        
        # Filter out zero values for better visualization
        non_zero_values = [v for v in values if v > 0] if metric != 'consistency_score' else values
        
        if not non_zero_values:
            axes[i].text(0.5, 0.5, 'No data available', transform=axes[i].transAxes, ha='center')
            axes[i].set_title(title, fontweight='bold')
            continue
        
        axes[i].hist(non_zero_values, bins=min(50, len(set(non_zero_values))), alpha=0.7, color='steelblue', edgecolor='black')
        axes[i].set_title(title, fontweight='bold')
        axes[i].set_xlabel(metric.replace('_', ' ').title())
        axes[i].set_ylabel('Number of Users')
        axes[i].grid(True, alpha=0.3)
        
        # Add mean line
        mean_val = np.mean(non_zero_values)
        axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('bangladesh/individual_user_distributions_may_aug_2024.png', dpi=300, bbox_inches='tight')
    plt.savefig('bangladesh/individual_user_distributions_may_aug_2024.pdf', bbox_inches='tight')
    print("Distribution plots saved as 'bangladesh/individual_user_distributions_may_aug_2024.png/.pdf'")

def print_comprehensive_analysis(categories, user_metrics, commits_data):
    """Print comprehensive analysis results."""
    print("=" * 80)
    print("INDIVIDUAL USER ANALYSIS - bangladesh GITHUB USERS (MAY-AUG 2024)")
    print("=" * 80)
    
    total_users = len(commits_data)
    
    if total_users == 0:
        print("No user data available for analysis")
        return
    
    # Activity Classification
    print(f"\n📊 USER ACTIVITY CLASSIFICATION (MAY-AUG):")
    print(f"Total Users Analyzed: {total_users:,}")
    print(f"Inactive (0 commits): {len(categories['inactive'])} ({len(categories['inactive'])/total_users*100:.1f}%)")
    print(f"Casual (1-3 commits/month): {len(categories['casual'])} ({len(categories['casual'])/total_users*100:.1f}%)")
    print(f"Power Users (200+ commits/month): {len(categories['power_users'])} ({len(categories['power_users'])/total_users*100:.1f}%)")
    print(f"Regular Users: {len(categories['regular'])} ({len(categories['regular'])/total_users*100:.1f}%)")
    
    # Distribution Analysis for key metrics
    key_metrics = ['total_commits', 'avg_commits_per_month', 'consistency_score', 'weekday_ratio']
    
    for metric in key_metrics:
        values = [user_metrics[user][metric] for user in user_metrics]
        usernames = list(user_metrics.keys())
        stats = calculate_distribution_stats(values)
        outliers = find_outliers_by_method(values, usernames, stats)
        
        print(f"\n📈 {metric.replace('_', ' ').upper()} ANALYSIS:")
        print(f"Mean: {stats['mean']:.2f}")
        print(f"Median (P50): {stats['percentiles']['P50']:.2f}")
        print(f"Standard Deviation: {stats['std']:.2f}")
        print(f"P25: {stats['percentiles']['P25']:.2f} | P75: {stats['percentiles']['P75']:.2f} | P95: {stats['percentiles']['P95']:.2f}")
        
        print(f"IQR Outliers: {len(outliers['iqr_outliers'])} users")
        print(f"Beyond 2σ: {len(outliers['above_2_sigma'])} users")
        print(f"Top 1%: {len(outliers['top_1_percent'])} users")
        
        if outliers['top_1_percent']:
            print("Top performers:")
            for username, value in sorted(outliers['top_1_percent'], key=lambda x: x[1], reverse=True)[:5]:
                print(f"   {username}: {value:.2f}")
    
    # Temporal Patterns Summary
    print(f"\n🕒 TEMPORAL PATTERNS SUMMARY (MAY-AUG):")
    
    # Consistency scores
    consistency_scores = [user_metrics[user]['consistency_score'] for user in user_metrics]
    high_consistency = sum(1 for score in consistency_scores if score > 0.7)
    print(f"Highly consistent users (>0.7): {high_consistency} ({high_consistency/total_users*100:.1f}%)")
    
    # Streak analysis
    long_streaks = sum(1 for user in user_metrics if user_metrics[user]['longest_streak'] > 30)
    print(f"Users with 30+ day streaks: {long_streaks} ({long_streaks/total_users*100:.1f}%)")
    
    # Weekend vs Weekday
    weekend_heavy = sum(1 for user in user_metrics if user_metrics[user]['weekend_ratio'] > 0.5)
    print(f"Weekend-heavy users (>50% weekend commits): {weekend_heavy} ({weekend_heavy/total_users*100:.1f}%)")
    
    # Monthly patterns (only count users with commits)
    active_users_months = [
        user_metrics[user]['most_active_month'] 
        for user in user_metrics 
        if user_metrics[user]['total_commits'] > 0 and user_metrics[user]['most_active_month'] is not None
    ]
    
    if active_users_months:
        month_names = {5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug'}
        month_preferences = Counter(active_users_months)
        for month in [5, 6, 7, 8]:
            if month in month_preferences:
                print(f"{month_names[month]}: {month_preferences[month]} active users")

def save_analysis_results(categories, user_metrics):
    """Save analysis results to JSON."""
    # Calculate metrics analysis for saving
    key_metrics = ['total_commits', 'avg_commits_per_month', 'consistency_score', 'weekday_ratio']
    metrics_analysis = {}
    
    for metric in key_metrics:
        values = [user_metrics[user][metric] for user in user_metrics]
        usernames = list(user_metrics.keys())
        stats = calculate_distribution_stats(values)
        outliers = find_outliers_by_method(values, usernames, stats)
        
        metrics_analysis[metric] = {
            'stats': stats,
            'outliers': outliers
        }
    
    output_data = {
        'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'period': 'May-August 2024',
        'total_users_analyzed': len(user_metrics),
        'user_classification': {
            'inactive_users': len(categories['inactive']),
            'casual_users': len(categories['casual']),
            'power_users': len(categories['power_users']),
            'regular_users': len(categories['regular'])
        },
        'distribution_statistics': {}
    }
    
    for metric, analysis in metrics_analysis.items():
        output_data['distribution_statistics'][metric] = {
            'mean': round(analysis['stats']['mean'], 3),
            'median': round(analysis['stats']['percentiles']['P50'], 3),
            'std': round(analysis['stats']['std'], 3),
            'percentiles': {k: round(v, 3) for k, v in analysis['stats']['percentiles'].items()},
            'outlier_counts': {k: len(v) for k, v in analysis['outliers'].items()}
        }
    
    with open('bangladesh/individual_analysis_may_aug_2024.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("\nAnalysis results saved as 'bangladesh/individual_analysis_may_aug_2024.json'")

def main():
    try:
        print("Loading commits data...")
        commits_data = load_commits_data('bangladesh/commits.json')
        
        if not commits_data:
            print("❌ No commits data found!")
            return
        
        print("Classifying users by activity level (May-Aug)...")
        categories = classify_users_by_activity(commits_data)
        
        print("Analyzing individual user patterns (May-Aug)...")
        user_metrics = analyze_individual_users(commits_data)
        
        if not user_metrics:
            print("❌ No user metrics generated!")
            return
        
        # Print comprehensive analysis
        print_comprehensive_analysis(categories, user_metrics, commits_data)
        
        # Create and save plots
        print("\nCreating distribution plots...")
        create_distribution_plots(user_metrics)
        
        # Save analysis results
        save_analysis_results(categories, user_metrics)
        
        print("\n✅ Individual user analysis complete!")
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON format: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()