import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from collections import defaultdict
import seaborn as sns

# Set style for academic papers
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11

def load_commits_data(file_path):
    """Load commits data from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def get_week_start_date(date_str):
    """Get the start date of the week using May 1, 2024 as reference point."""
    try:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        reference_date = datetime(2024, 5, 1)
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
        
        for date_str, commit_count in daily_commits.items():
            if commit_count > 0:
                week_start = get_week_start_date(date_str)
                if week_start and week_start in pre_treatment_weeks:
                    user_weekly_commits[week_start] += commit_count
        
        if len(user_weekly_commits) == len(pre_treatment_weeks) and all(v > 0 for v in user_weekly_commits.values()):
            consistent_users.append(username)
    
    return consistent_users

def collapse_to_weekly(commits_data, consistent_users_only=True):
    """Collapse daily commits to weekly totals."""
    weekly_commits = defaultdict(int)
    all_weeks = set()
    
    # Define treatment start
    treatment_start = datetime(2024, 7, 17)
    
    # Get all weeks first
    for username, user_data in commits_data.items():
        daily_commits = user_data.get('daily_commits', {})
        for date_str, commit_count in daily_commits.items():
            if commit_count > 0:
                week_start = get_week_start_date(date_str)
                if week_start:
                    all_weeks.add(week_start)
    
    # Filter to May-August
    complete_weeks = filter_complete_weeks_may_aug(all_weeks)
    pre_treatment_weeks = [w for w in complete_weeks if datetime.strptime(w, '%Y-%m-%d') < treatment_start]
    
    # Get consistent users if requested
    if consistent_users_only:
        consistent_users = filter_consistent_users(commits_data, pre_treatment_weeks)
        users_to_process = {u: commits_data[u] for u in consistent_users}
        print(f"Processing {len(consistent_users)} consistent users")
    else:
        users_to_process = commits_data
    
    # Aggregate commits
    for username, user_data in users_to_process.items():
        daily_commits = user_data.get('daily_commits', {})
        for date_str, commit_count in daily_commits.items():
            if commit_count > 0:
                week_start = get_week_start_date(date_str)
                if week_start and week_start in complete_weeks:
                    weekly_commits[week_start] += commit_count
    
    filtered_weekly = {week: weekly_commits[week] for week in complete_weeks}
    return filtered_weekly, len(users_to_process)

def create_user_week_data_for_averages(commits_data, consistent_users_only=True):
    """Create user-week data for calculating averages."""
    user_week_data = []
    all_weeks = set()
    
    # Define treatment start
    treatment_start = datetime(2024, 7, 17)
    
    # Get all weeks first
    for username, user_data in commits_data.items():
        daily_commits = user_data.get('daily_commits', {})
        for date_str, commit_count in daily_commits.items():
            if commit_count > 0:
                week_start = get_week_start_date(date_str)
                if week_start:
                    all_weeks.add(week_start)
    
    # Filter to May-August
    complete_weeks = filter_complete_weeks_may_aug(all_weeks)
    pre_treatment_weeks = [w for w in complete_weeks if datetime.strptime(w, '%Y-%m-%d') < treatment_start]
    
    # Get consistent users if requested
    if consistent_users_only:
        consistent_users = filter_consistent_users(commits_data, pre_treatment_weeks)
        users_to_process = {u: commits_data[u] for u in consistent_users}
    else:
        users_to_process = commits_data
    
    for username, user_data in users_to_process.items():
        daily_commits = user_data.get('daily_commits', {})
        user_weekly_commits = defaultdict(int)
        
        for date_str, commit_count in daily_commits.items():
            if commit_count > 0:
                week_start = get_week_start_date(date_str)
                if week_start and week_start in complete_weeks:
                    user_weekly_commits[week_start] += commit_count
        
        for week in complete_weeks:
            user_week_data.append({
                'user': username,
                'week': week,
                'commits': user_weekly_commits.get(week, 0)
            })
    
    return user_week_data, complete_weeks, len(users_to_process)

def calculate_weekly_averages_per_user(user_week_data, complete_weeks, total_users):
    """Calculate average commits per user per week."""
    weekly_averages = {}
    
    for week in complete_weeks:
        week_commits = [obs['commits'] for obs in user_week_data if obs['week'] == week]
        avg_commits = sum(week_commits) / total_users if total_users > 0 else 0
        weekly_averages[week] = avg_commits
    
    return weekly_averages

def remove_top_5_percent_outliers(user_week_data):
    """Remove user-week observations in the top 5% of weekly commits."""
    if not user_week_data:
        return [], []
    
    commits_values = [obs['commits'] for obs in user_week_data]
    p95_threshold = np.percentile(commits_values, 95)
    
    outliers = [obs for obs in user_week_data if obs['commits'] > p95_threshold]
    clean_data = [obs for obs in user_week_data if obs['commits'] <= p95_threshold]
    
    return clean_data, outliers, p95_threshold

# GRÁFICA 1: Serie Temporal Básica - Commits Totales por Semana
def plot_basic_weekly_series(bangladesh_weekly, philippines_weekly, bg_users, ph_users, save_prefix="01_basic"):
    """Create basic weekly time series plot."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Convert to datetime and sort
    bg_dates = [datetime.strptime(date, '%Y-%m-%d') for date in sorted(bangladesh_weekly.keys())]
    bg_commits = [bangladesh_weekly[date.strftime('%Y-%m-%d')] for date in bg_dates]
    
    ph_dates = [datetime.strptime(date, '%Y-%m-%d') for date in sorted(philippines_weekly.keys())]
    ph_commits = [philippines_weekly[date.strftime('%Y-%m-%d')] for date in ph_dates]
    
    # Plot lines
    ax.plot(bg_dates, bg_commits, 'b-', linewidth=2.5, label=f'Bangladesh (n={bg_users})', marker='o', markersize=4)
    ax.plot(ph_dates, ph_commits, 'r-', linewidth=2.5, label=f'Filipinas (n={ph_users})', marker='s', markersize=4)
    
    # Add treatment period shading
    treatment_start = datetime(2024, 7, 17)
    treatment_end = datetime(2024, 7, 24)
    ax.axvspan(treatment_start, treatment_end, alpha=0.3, color='gray', label='Apagón Bangladesh (17-24 Jul)')
    
    # Formatting
    ax.set_title('Commits Semanales Totales - Bangladesh vs Filipinas (Mayo-Setiembre 2024)', fontweight='bold', fontsize=14)
    ax.set_xlabel('Fecha (Inicio de Semana)', fontsize=12)
    ax.set_ylabel('Total de Commits por Semana', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
    
    # Format y-axis with comma separators
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save
    plt.savefig(f'graphs/{save_prefix}_weekly_commits_total_may_aug.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'graphs/{save_prefix}_weekly_commits_total_may_aug.pdf', bbox_inches='tight')
    plt.show()
    
    print(f"Gráfica guardada: {save_prefix}_weekly_commits_total_may_aug.png/.pdf")

# GRÁFICA 2: Serie Temporal - Promedio de Commits por Usuario por Semana
def plot_weekly_averages_per_user(bangladesh_data, philippines_data, save_prefix="02_averages"):
    """Create weekly averages per user time series plot."""
    # Get user-week data
    bg_user_week, bg_weeks, bg_users = create_user_week_data_for_averages(bangladesh_data, consistent_users_only=True)
    ph_user_week, ph_weeks, ph_users = create_user_week_data_for_averages(philippines_data, consistent_users_only=True)
    
    # Remove outliers
    bg_clean, _, bg_threshold = remove_top_5_percent_outliers(bg_user_week)
    ph_clean, _, ph_threshold = remove_top_5_percent_outliers(ph_user_week)
    
    # Calculate averages
    bg_averages = calculate_weekly_averages_per_user(bg_clean, bg_weeks, bg_users)
    ph_averages = calculate_weekly_averages_per_user(ph_clean, ph_weeks, ph_users)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Convert to datetime and sort
    bg_dates = [datetime.strptime(date, '%Y-%m-%d') for date in sorted(bg_averages.keys())]
    bg_avg_commits = [bg_averages[date.strftime('%Y-%m-%d')] for date in bg_dates]
    
    ph_dates = [datetime.strptime(date, '%Y-%m-%d') for date in sorted(ph_averages.keys())]
    ph_avg_commits = [ph_averages[date.strftime('%Y-%m-%d')] for date in ph_dates]
    
    # Plot lines
    ax.plot(bg_dates, bg_avg_commits, 'b-', linewidth=2.5, label=f'Bangladesh (n={bg_users})', marker='o', markersize=4)
    ax.plot(ph_dates, ph_avg_commits, 'r-', linewidth=2.5, label=f'Filipinas (n={ph_users})', marker='s', markersize=4)
    
    # Add treatment period shading
    treatment_start = datetime(2024, 7, 17)
    treatment_end = datetime(2024, 7, 24)
    ax.axvspan(treatment_start, treatment_end, alpha=0.3, color='gray', label='Apagón Bangladesh (17-24 Jul)')
    
    # Formatting
    ax.set_title('Promedio de Commits por Usuario por Semana (Mayo-Setiembre 2024)\nOutliers P95+ Removidos', fontweight='bold', fontsize=14)
    ax.set_xlabel('Fecha (Inicio de Semana)', fontsize=12)
    ax.set_ylabel('Promedio de Commits por Usuario por Semana', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save
    plt.savefig(f'graphs/{save_prefix}_weekly_averages_per_user_may_aug.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'graphs/{save_prefix}_weekly_averages_per_user_may_aug.pdf', bbox_inches='tight')
    plt.show()
    
    print(f"Gráfica guardada: {save_prefix}_weekly_averages_per_user_may_aug.png/.pdf")

# GRÁFICA 3: Comparación Pre-Post Tratamiento
def plot_pre_post_comparison(bangladesh_data, philippines_data, save_prefix="03_pre_post"):
    """Create pre-post treatment comparison."""
    # Get user-week data
    bg_user_week, bg_weeks, bg_users = create_user_week_data_for_averages(bangladesh_data, consistent_users_only=True)
    ph_user_week, ph_weeks, ph_users = create_user_week_data_for_averages(philippines_data, consistent_users_only=True)
    
    # Remove outliers
    bg_clean, _, _ = remove_top_5_percent_outliers(bg_user_week)
    ph_clean, _, _ = remove_top_5_percent_outliers(ph_user_week)
    
    # Define periods
    treatment_week = "2024-07-17"
    
    def categorize_period(week_str):
        week_date = datetime.strptime(week_str, '%Y-%m-%d')
        treatment_date = datetime.strptime(treatment_week, '%Y-%m-%d')
        
        if week_date < treatment_date:
            return 'Pre-Tratamiento'
        elif week_date == treatment_date:
            return 'Tratamiento'
        else:
            return 'Post-Tratamiento'
    
    # Calculate averages by period
    def calc_period_averages(clean_data, total_users):
        period_data = defaultdict(list)
        for obs in clean_data:
            period = categorize_period(obs['week'])
            period_data[period].append(obs['commits'])
        
        period_averages = {}
        for period, commits_list in period_data.items():
            period_averages[period] = np.mean(commits_list) if commits_list else 0
        
        return period_averages
    
    bg_period_avg = calc_period_averages(bg_clean, bg_users)
    ph_period_avg = calc_period_averages(ph_clean, ph_users)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    periods = ['Pre-Tratamiento', 'Tratamiento', 'Post-Tratamiento']
    bg_values = [bg_period_avg.get(p, 0) for p in periods]
    ph_values = [ph_period_avg.get(p, 0) for p in periods]
    
    x = np.arange(len(periods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, bg_values, width, label=f'Bangladesh (n={bg_users})', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, ph_values, width, label=f'Filipinas (n={ph_users})', color='coral', alpha=0.8)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
    
    ax.set_title('Promedio de Commits por Usuario-Semana: Pre vs Durante vs Post Tratamiento\nOutliers P95+ Removidos', fontweight='bold', fontsize=14)
    ax.set_xlabel('Período', fontsize=12)
    ax.set_ylabel('Promedio de Commits por Usuario-Semana', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(periods)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    plt.savefig(f'graphs/{save_prefix}_pre_post_comparison_may_aug.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'graphs/{save_prefix}_pre_post_comparison_may_aug.pdf', bbox_inches='tight')
    plt.show()
    
    print(f"Gráfica guardada: {save_prefix}_pre_post_comparison_may_aug.png/.pdf")

# GRÁFICA 4: Correlación de Tendencias Pre-Tratamiento
def plot_pre_treatment_correlation(bangladesh_data, philippines_data, save_prefix="04_correlation"):
    """Plot pre-treatment correlation between countries."""
    # Get user-week data
    bg_user_week, bg_weeks, bg_users = create_user_week_data_for_averages(bangladesh_data, consistent_users_only=True)
    ph_user_week, ph_weeks, ph_users = create_user_week_data_for_averages(philippines_data, consistent_users_only=True)
    
    # Remove outliers
    bg_clean, _, _ = remove_top_5_percent_outliers(bg_user_week)
    ph_clean, _, _ = remove_top_5_percent_outliers(ph_user_week)
    
    # Calculate weekly averages
    bg_averages = calculate_weekly_averages_per_user(bg_clean, bg_weeks, bg_users)
    ph_averages = calculate_weekly_averages_per_user(ph_clean, ph_weeks, ph_users)
    
    # Filter to pre-treatment period only
    treatment_date = datetime(2024, 7, 17)
    
    pre_treatment_weeks = [week for week in bg_weeks 
                          if datetime.strptime(week, '%Y-%m-%d') < treatment_date]
    
    bg_pre = [bg_averages[week] for week in pre_treatment_weeks]
    ph_pre = [ph_averages[week] for week in pre_treatment_weeks]
    
    # Create correlation plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot
    ax1.scatter(bg_pre, ph_pre, alpha=0.7, s=60, color='steelblue')
    
    # Add trend line
    z = np.polyfit(bg_pre, ph_pre, 1)
    p = np.poly1d(z)
    ax1.plot(bg_pre, p(bg_pre), "r--", alpha=0.8, linewidth=2)
    
    # Calculate correlation
    correlation = np.corrcoef(bg_pre, ph_pre)[0, 1]
    
    ax1.set_xlabel('Bangladesh - Promedio Usuario-Semana', fontsize=12)
    ax1.set_ylabel('Filipinas - Promedio Usuario-Semana', fontsize=12)
    ax1.set_title(f'Correlación Pre-Tratamiento (Mayo-Julio 16)\nr = {correlation:.3f}', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Time series comparison
    pre_dates = [datetime.strptime(week, '%Y-%m-%d') for week in pre_treatment_weeks]
    
    ax2.plot(pre_dates, bg_pre, 'b-', linewidth=2, marker='o', markersize=4, label=f'Bangladesh (n={bg_users})')
    ax2.plot(pre_dates, ph_pre, 'r-', linewidth=2, marker='s', markersize=4, label=f'Filipinas (n={ph_users})')
    
    ax2.set_title('Tendencias Paralelas Pre-Tratamiento', fontweight='bold')
    ax2.set_xlabel('Fecha', fontsize=12)
    ax2.set_ylabel('Promedio Usuario-Semana', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(f'graphs/{save_prefix}_pre_treatment_correlation_may_aug.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'graphs/{save_prefix}_pre_treatment_correlation_may_aug.pdf', bbox_inches='tight')
    plt.show()
    
    print(f"Gráfica guardada: {save_prefix}_pre_treatment_correlation_may_aug.png/.pdf")
    print(f"Correlación pre-tratamiento: {correlation:.3f}")

def create_all_graphs():
    """Create all graphs for May-August analysis."""
    
    print("🎨 CREANDO GRÁFICAS PARA ANÁLISIS MAYO-SETIEMBRE (USUARIOS CONSISTENTES)")
    print("="*70)
    
    # Create output directory
    import os
    os.makedirs('graphs', exist_ok=True)
    
    # Load data
    print("Cargando datos...")
    bangladesh_data = load_commits_data('main/bangladesh/commits.json')
    philippines_data = load_commits_data('philippines/commits.json')
    
    # Get weekly aggregates with consistent users
    bangladesh_weekly, bg_users = collapse_to_weekly(bangladesh_data, consistent_users_only=True)
    philippines_weekly, ph_users = collapse_to_weekly(philippines_data, consistent_users_only=True)
    
    print(f"\nUsuarios consistentes encontrados:")
    print(f"Bangladesh: {bg_users} usuarios")
    print(f"Filipinas: {ph_users} usuarios\n")
    
    # Create all graphs
    print("1. Creando gráfica básica de series temporales...")
    plot_basic_weekly_series(bangladesh_weekly, philippines_weekly, bg_users, ph_users)
    
    print("\n2. Creando gráfica de promedios por usuario...")
    plot_weekly_averages_per_user(bangladesh_data, philippines_data)
    
    print("\n3. Creando comparación pre-post tratamiento...")
    plot_pre_post_comparison(bangladesh_data, philippines_data)
    
    print("\n4. Creando análisis de correlación pre-tratamiento...")
    plot_pre_treatment_correlation(bangladesh_data, philippines_data)
    
    print("\n✅ ¡Todas las gráficas creadas exitosamente!")
    print("📁 Revisa la carpeta 'graphs/' para todos los archivos .png y .pdf")

def main():
    """Main function to create all graphs."""
    try:
        create_all_graphs()
        
    except FileNotFoundError as e:
        print(f"❌ Error: No se encontraron los archivos de datos.")
        print(f"💡 Asegúrate de tener 'main/bangladesh/commits.json' y 'philippines/commits.json'")
        
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

if __name__ == "__main__":
    main()