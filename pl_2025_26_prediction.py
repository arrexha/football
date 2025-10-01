

import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def parse_match_results(df: pd.DataFrame) -> pd.DataFrame:

    goals = df["FT"].str.split("-", expand=True)
    df = df.copy()
    df["home_goals"] = goals[0].astype(int)
    df["away_goals"] = goals[1].astype(int)
    return df


def summarise_season(matches: pd.DataFrame) -> pd.DataFrame:

    teams: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "points": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "goals_for": 0,
        "goals_against": 0,
    })
    # iterate through each match and update team statistics
    for _, row in matches.iterrows():
        home, away = row["Team 1"], row["Team 2"]
        hg, ag = row["home_goals"], row["away_goals"]
        # update goals
        teams[home]["goals_for"] += hg
        teams[home]["goals_against"] += ag
        teams[away]["goals_for"] += ag
        teams[away]["goals_against"] += hg
        # determine match outcome
        if hg > ag:
            # home win
            teams[home]["points"] += 3
            teams[home]["wins"] += 1
            teams[away]["losses"] += 1
        elif hg < ag:
            # away win
            teams[away]["points"] += 3
            teams[away]["wins"] += 1
            teams[home]["losses"] += 1
        else:
            # draw
            teams[home]["points"] += 1
            teams[away]["points"] += 1
            teams[home]["draws"] += 1
            teams[away]["draws"] += 1
    # build DataFrame
    data = []
    for team, stats in teams.items():
        goal_diff = stats["goals_for"] - stats["goals_against"]
        data.append(
            {
                "team": team,
                "points": stats["points"],
                "wins": stats["wins"],
                "draws": stats["draws"],
                "losses": stats["losses"],
                "goals_for": stats["goals_for"],
                "goals_against": stats["goals_against"],
                "goal_diff": goal_diff,
            }
        )
    summary = pd.DataFrame(data)
    # sort by points, goal diff, goals for
    summary = summary.sort_values(
        ["points", "goal_diff", "goals_for"], ascending=[False, False, False]
    ).reset_index(drop=True)
    summary["position"] = summary.index + 1
    return summary


def prepare_training_data(season_files: List[str]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:

    season_summaries: Dict[str, pd.DataFrame] = {}
    # compute summary stats for each season
    for file_path in season_files:
        raw = pd.read_csv(file_path)
        parsed = parse_match_results(raw)
        summary = summarise_season(parsed)
        season_summaries[file_path] = summary
    # Build training dataset: use season n's stats to predict season n+1's position
    feature_rows = []
    target_rows = []
    files_sorted = season_files
    for i in range(len(files_sorted) - 1):
        prev_summary = season_summaries[files_sorted[i]].copy().set_index("team")
        curr_summary = season_summaries[files_sorted[i + 1]].copy().set_index("team")
        # compute default features based on bottom three teams from previous season
        bottom_three = prev_summary.sort_values(
            ["points", "goal_diff", "goals_for"], ascending=[True, True, True]
        ).head(3)
        default_features = bottom_three.mean().to_dict()
        # for each team in current season, collect features
        for team, row in curr_summary.iterrows():
            if team in prev_summary.index:
                feats = prev_summary.loc[team][
                    ["points", "wins", "draws", "losses", "goals_for", "goals_against", "goal_diff"]
                ].to_dict()
            else:
                # promoted team – assign default bottom three stats
                feats = {k: default_features[k] for k in [
                    "points", "wins", "draws", "losses", "goals_for", "goals_against", "goal_diff"
                ]}
            feature_rows.append(feats)
            target_rows.append(row["position"])
    X_train = pd.DataFrame(feature_rows)
    y_train = pd.Series(target_rows)
    # features for the most recent season for which we will predict the next season
    last_summary = season_summaries[files_sorted[-1]].copy().set_index("team")
    # compute default features for new promoted teams in the upcoming season
    # this uses bottom three of last_summary
    bottom_three_last = last_summary.sort_values(
        ["points", "goal_diff", "goals_for"], ascending=[True, True, True]
    ).head(3)
    default_features_last = bottom_three_last.mean().to_dict()
    latest_features_rows = []
    latest_teams = last_summary.index.tolist()
    # incorporate promoted teams for 2025/26 (Leeds United, Burnley, Sunderland)
    promoted = ["Leeds United", "Burnley", "Sunderland"]
    # if a promoted team already exists in last_summary (e.g. Burnley was relegated earlier), use its stats
    for team in latest_teams:
        feats = last_summary.loc[team][
            ["points", "wins", "draws", "losses", "goals_for", "goals_against", "goal_diff"]
        ].to_dict()
        latest_features_rows.append((team, feats))
    for team in promoted:
        if team not in latest_teams:
            feats = {k: default_features_last[k] for k in [
                "points", "wins", "draws", "losses", "goals_for", "goals_against", "goal_diff"
            ]}
            latest_features_rows.append((team, feats))
    latest_features_df = pd.DataFrame([feats for _, feats in latest_features_rows],
                                      index=[t for t, _ in latest_features_rows])
    return X_train, y_train, latest_features_df


def build_and_train_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
   
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            class_weight="balanced"
        ))
    ])
    model.fit(X, y)
    return model


def predict_league_table(model: Pipeline, features: pd.DataFrame) -> pd.DataFrame:

    probas = model.predict_proba(features)
    classes = model.named_steps["rf"].classes_
    exp_positions = probas.dot(classes)
    prediction_df = pd.DataFrame({
        "team": features.index,
        "expected_position": exp_positions
    })
    # sort teams by lowest expected position (i.e. best finish)
    prediction_df = prediction_df.sort_values("expected_position").reset_index(drop=True)
    # assign integer ranks 1..n based on sorted order
    prediction_df["predicted_rank"] = prediction_df.index + 1
    return prediction_df[["predicted_rank", "team", "expected_position"]]


def main():
    # define the season files in chronological order
    season_files = [
        os.path.join(os.path.dirname(__file__), "eng1_2018-19.csv"),
        os.path.join(os.path.dirname(__file__), "eng1_2019-20.csv"),
        os.path.join(os.path.dirname(__file__), "eng1_2020-21.csv"),
        os.path.join(os.path.dirname(__file__), "eng1_2021-22.csv"),
        os.path.join(os.path.dirname(__file__), "eng1_2022-23.csv"),
        os.path.join(os.path.dirname(__file__), "eng1_2023-24.csv"),
    ]
    # prepare training data
    X_train, y_train, latest_features = prepare_training_data(season_files)
    # train model
    model = build_and_train_model(X_train, y_train)
    # predict ranking for 2025/26
    predictions = predict_league_table(model, latest_features)
    # keep only the top 20 teams based on expected position.  In reality
    # the Premier League contains exactly 20 clubs.  Since we may
    # include extra promoted teams due to unavailable data for the
    # intermediate 2024/25 season, truncate to 20.
    predictions = predictions.iloc[:20].copy()
    print("Predicted Premier League 2025/26 table (1 = champion):")
    for _, row in predictions.iterrows():
        print(
            f"{int(row['predicted_rank'])}. {row['team']} "
            f"(expected pos {row['expected_position']:.2f})"
        )


if __name__ == "__main__":
    main()
