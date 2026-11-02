import os
import pandas as pd
from src.config.database import SessionLocal
from src.analysis.market_overview import get_market_overview
from src.analysis.data_quality import analyze_data_quality
from src.analysis.liquidity_analysis import analyze_liquidity
from src.analysis.volatility_analysis import analyze_volatility

def generate_market_reports():
    db = SessionLocal()
    try:
        print("\n" + "="*50)
        print(" BVMT AI TRADER - MARKET & QUALITY REPORT")
        print("="*50)

        # 1. Market Overview
        overview = get_market_overview(db)
        print(f"\n[1] MARKET OVERVIEW")
        print(f" - Total Registered Securities : {overview['total_registered_securities']}")
        print(f" - Securities with Data        : {overview['securities_with_data']}")
        print(f" - Total Historical Records    : {overview['total_historical_records']}")
        print(f" - First Trading Date          : {overview['first_trading_date']}")
        print(f" - Last Trading Date           : {overview['last_trading_date']}")

        # 2. Data Quality
        quality = analyze_data_quality(db)
        print(f"\n[2] DATA QUALITY REPORT")
        print(f" - Total Records Analyzed      : {quality['total_records']}")
        print(f" - Total Issues Detected       : {quality['total_issues']}")

        # 3. Liquidity Rankings
        liquidity = analyze_liquidity(db)
        print(f"\n[3] LIQUIDITY RANKING (Top Active Stocks)")
        for item in liquidity[:5]:
            print(f" - {item['symbol']}: Avg Vol={item['average_daily_volume']}, Avg Val={item['average_daily_value']}")

        # 4. Volatility Rankings
        volatility = analyze_volatility(db)
        print(f"\n[4] VOLATILITY & RISK ANALYSIS")
        for item in volatility[:5]:
            print(f" - {item['symbol']}: Ann. Vol={item['annualized_volatility']*100:.2f}%, Max DD={item['max_drawdown']*100:.2f}%")

        # Export to CSV reports directory
        report_dir = "data/processed/reports"
        os.makedirs(report_dir, exist_ok=True)

        if overview['stock_stats']:
            pd.DataFrame(overview['stock_stats']).to_csv(f"{report_dir}/market_overview.csv", index=False)
        if quality['issues']:
            pd.DataFrame(quality['issues']).to_csv(f"{report_dir}/data_quality_issues.csv", index=False)
        if liquidity:
            pd.DataFrame(liquidity).to_csv(f"{report_dir}/liquidity_ranking.csv", index=False)
        if volatility:
            pd.DataFrame(volatility).to_csv(f"{report_dir}/volatility_ranking.csv", index=False)

        print(f"\n[SUCCESS] All reports exported successfully to '{report_dir}/'")
        print("="*50 + "\n")

    finally:
        db.close()

if __name__ == "__main__":
    generate_market_reports()