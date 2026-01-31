"""Backward compatibility wrapper for export_dashboard_data."""

from rugby_ranking.tools.export_dashboard_data import export_dashboard_data

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../Rugby-Data")
    output_dir = Path("dashboard/data")
    
    export_dashboard_data(
        data_dir=data_dir,
        output_dir=output_dir,
        checkpoint_name="international-mini5",
        recent_seasons_only=3,
    )
