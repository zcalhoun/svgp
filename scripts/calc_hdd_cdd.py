"""
This script simply loads the data and calculates the HDD and CDD for each month
and year, and saves the results into a single CSV file.

"""

import os
import argparse
import pandas as pd


def main(args):
    """
    Iterate through and load the data, calculate the HDD and CDD, and save the results.
    """
    all_dfs = []
    for year in range(2019, 2025):
        print(f"On year {year}")
        for month in range(1, 13):
            print(f"On month {month}")
            df = pd.read_csv(os.path.join(args.ref_data, f"{year}-{month}.csv"))
            df["hdd"] = (args.threshold - df["pred"]).clip(lower=0)
            df["cdd"] = (df["pred"] - args.threshold).clip(lower=0)
            df["hdd_upper95"] = (args.threshold - df["upper95"]).clip(lower=0)
            df["cdd_upper95"] = (df["upper95"] - args.threshold).clip(lower=0)
            df["hdd_lower95"] = (args.threshold - df["lower95"]).clip(lower=0)
            df["cdd_lower95"] = (df["lower95"] - args.threshold).clip(lower=0)
            sum_df = df.groupby(["lat", "lon"], as_index=False)[
                [
                    "hdd",
                    "cdd",
                    "hdd_upper95",
                    "cdd_upper95",
                    "hdd_lower95",
                    "cdd_lower95",
                ]
            ].sum()
            sum_df["hdd"] /= 24
            sum_df["cdd"] /= 24
            sum_df["hdd_upper95"] /= 24
            sum_df["cdd_upper95"] /= 24
            sum_df["hdd_lower95"] /= 24
            sum_df["cdd_lower95"] /= 24
            sum_df["year"] = year
            sum_df["month"] = month
            all_dfs.append(sum_df)

    all_df = pd.concat(all_dfs, ignore_index=True)
    output_file = os.path.join(args.output, "hdd_cdd.csv")
    all_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate HDD and CDD from predictions."
    )
    parser.add_argument(
        "--ref_data",
        type=str,
        required=True,
        help="Path to the reference data directory.",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to the output directory."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=18.3,
        help="Temperature threshold for HDD/CDD calculation (default: 18.3 Celsius).",
    )

    main(parser.parse_args())
