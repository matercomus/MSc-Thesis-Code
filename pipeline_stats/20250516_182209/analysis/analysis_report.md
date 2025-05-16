# Pipeline Analysis Report

**Run ID:** `20250516_182209`

**Compared to Run ID:** `20250516_181913`

## Summary Table
| Step                   | Before   | After   | Filtered   | % Filtered   | Cumulative % Retained   | Reused   | Reused From                                                     |
|------------------------|----------|---------|------------|--------------|-------------------------|----------|-----------------------------------------------------------------|
| cleaned_points         |          |         |            |              |                         | Yes      | data/cleaned_points_in_beijing_20250516_175741.parquet          |
| cleaned_with_period_id |          |         |            |              |                         | Yes      | data/cleaned_with_period_id_in_beijing_20250516_175741.parquet  |
| periods_with_sld_ratio |          |         |            |              |                         | Yes      | data/periods_with_sld_ratio_20250516_175741.parquet             |
| network_outlier_flag   |          |         |            |              |                         | Yes      | data/periods_with_network_ratio_flagged_20250516_175741.parquet |

## Indicator Overlaps
| Overlap                                               | Count                                                  |
|-------------------------------------------------------|--------------------------------------------------------|
| is_traj_outlier                                       | {'n_flagged': 4, 'flagged_pct': 0.032110459982339246}  |
| is_sld_outlier                                        | {'n_flagged': 17, 'flagged_pct': 0.1364694549249418}   |
| is_traj_outlier & is_sld_outlier                      | {'n_flagged': 0, 'flagged_pct': 0.0}                   |
| is_network_outlier                                    | {'n_flagged': 1603, 'flagged_pct': 12.868266837922453} |
| is_traj_outlier & is_network_outlier                  | {'n_flagged': 0, 'flagged_pct': 0.0}                   |
| is_sld_outlier & is_network_outlier                   | {'n_flagged': 13, 'flagged_pct': 0.10435899494260255}  |
| is_traj_outlier & is_sld_outlier & is_network_outlier | {'n_flagged': 0, 'flagged_pct': 0.0}                   |

### Route Deviation Ratio Histogram

![](route_deviation_ratio_hist.png)

### Network Shortest Distance Histogram

![](network_shortest_distance_hist.png)

# Comparison to Previous Run

## Indicator Overlap Comparison
| Overlap                                               | 20250516_182209                                        | 20250516_181913                                        | Diff   |
|-------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------|--------|
| is_traj_outlier & is_network_outlier                  | {'n_flagged': 0, 'flagged_pct': 0.0}                   | {'n_flagged': 0, 'flagged_pct': 0.0}                   | N/A    |
| is_traj_outlier                                       | {'n_flagged': 4, 'flagged_pct': 0.032110459982339246}  | {'n_flagged': 4, 'flagged_pct': 0.032110459982339246}  | N/A    |
| is_sld_outlier & is_network_outlier                   | {'n_flagged': 13, 'flagged_pct': 0.10435899494260255}  | {'n_flagged': 13, 'flagged_pct': 0.10435899494260255}  | N/A    |
| is_traj_outlier & is_sld_outlier & is_network_outlier | {'n_flagged': 0, 'flagged_pct': 0.0}                   | {'n_flagged': 0, 'flagged_pct': 0.0}                   | N/A    |
| is_traj_outlier & is_sld_outlier                      | {'n_flagged': 0, 'flagged_pct': 0.0}                   | {'n_flagged': 0, 'flagged_pct': 0.0}                   | N/A    |
| is_sld_outlier                                        | {'n_flagged': 17, 'flagged_pct': 0.1364694549249418}   | {'n_flagged': 17, 'flagged_pct': 0.1364694549249418}   | N/A    |
| is_network_outlier                                    | {'n_flagged': 1603, 'flagged_pct': 12.868266837922453} | {'n_flagged': 1603, 'flagged_pct': 12.868266837922453} | N/A    |

