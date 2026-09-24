from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ALLOWED_FILES = {
    "Accelerometer.csv",
    "Gyroscope.csv",
    "Gravity.csv",
    "Metadata.csv",
    "Annotation.csv",
}

SENSOR_FILES = {
    "Accelerometer.csv": "accel",
    "Gyroscope.csv": "gyro",
    "Gravity.csv": "gravity",
}


def _read_csv_file(path):
    """
    Reads a CSV file from disk into a DataFrame.

    :param path: Path object pointing to the file.
    :return: DataFrame or None if missing, empty or unreadable.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return None


def data_ingestion(input_path, output_path=None, save_csv=True):
    """
    Concatenates matching CSV file types across all session directories into single DataFrames.

    :param input_path: Path to the root input directory.
    :param output_path: Optional path for saving merged files.
    :param save_csv: Boolean indicating whether to write CSVs to disk.
    :return: Dict of DataFrames per sensor/file type.
    """
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path.parent / "OutputData"

    collected = {}

    for session_dir in sorted(p for p in input_path.iterdir() if p.is_dir()):
        for csv_path in session_dir.glob("*.csv"):
            if csv_path.name not in ALLOWED_FILES:
                continue

            df = _read_csv_file(csv_path)
            if df is not None:
                # Optimized removal and insertion of recording_id
                df = df.drop(columns=["recording_id"], errors="ignore")
                df.insert(0, "recording_id", session_dir.name)

                collected.setdefault(csv_path.stem, []).append(df)

    if not collected:
        print(f"No valid session data found in {input_path}")
        return {}

    dataframes = {
        key: pd.concat(dfs, ignore_index=True).drop_duplicates() 
        for key, dfs in collected.items()
    }

    if save_csv:
        output_path.mkdir(parents=True, exist_ok=True)
        for key, df in dataframes.items():
            df.to_csv(output_path / f"{key}.csv", index=False)
            print(f"Saved: {key}.csv ({len(df)} rows)")

    return dataframes


def build_session_dataset(input_path, output_path=None, save_csv=True):
    """
    Merges sensor axes (Accelerometer, Gyroscope, Gravity), metadata, and annotations.

    :param input_path: Path to the root input directory.
    :param output_path: Optional path for saving the final dataset.
    :param save_csv: Boolean indicating whether to write CSV to disk.
    :return: Merged dataset DataFrame.
    """
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path.parent / "OutputData"

    sessions = []
    session_dirs = sorted(p for p in input_path.iterdir() if p.is_dir())

    for session_num, session_dir in enumerate(session_dirs, start=1):
        sensor_dfs = {}

        # 1. Load and process sensor files
        for fname, prefix in SENSOR_FILES.items():
            df = _read_csv_file(session_dir / fname)
            if df is not None:
                df.columns = df.columns.str.strip()
                df = df.drop(columns=["recording_id", "time"], errors="ignore")

                # Rename sensor channels
                rename_map = {
                    c: f"{prefix}_{c}" for c in df.columns if c != "seconds_elapsed"
                }
                df = df.rename(columns=rename_map)

                # Deduplicate and sort timestamps
                df = df.drop_duplicates(subset=["seconds_elapsed"]).sort_values("seconds_elapsed")
                sensor_dfs[prefix] = df

        # Accelerometer is mandatory as base reference
        if "accel" not in sensor_dfs:
            continue

        # 2. Sequential merge using Accelerometer as main timeline
        merged = sensor_dfs["accel"]

        for prefix in ["gyro", "gravity"]:
            if prefix in sensor_dfs:
                merged = pd.merge_asof(
                    merged,
                    sensor_dfs[prefix],
                    on="seconds_elapsed",
                    direction="nearest",
                )

        # 3. Metadata and Annotation extraction
        meta_df = _read_csv_file(session_dir / "Metadata.csv")
        type_device = meta_df.iloc[0].get("device name") if meta_df is not None else None
        id_device = meta_df.iloc[0].get("device id") if meta_df is not None else None

        annot_file = session_dir / "Annotation.csv"
        has_annotation = annot_file.is_file() and annot_file.stat().st_size > 0

        # 4. Attach session attributes
        merged["type_device"] = type_device
        merged["id_device"] = id_device
        merged["annotation"] = has_annotation
        merged.insert(0, "session", session_num)

        sessions.append(merged)

    if not sessions:
        return pd.DataFrame()

    # Concatenate and assign unique sequential ID fast
    dataset = pd.concat(sessions, ignore_index=True)
    dataset = dataset.reset_index().rename(columns={"index": "id"})

    if save_csv:
        output_path.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(output_path / "sessions_dataset.csv", index=False)
        print(
            f"Saved: sessions_dataset.csv ({len(dataset)} rows, {len(sessions)} sessions merged)"
        )

    return dataset


def check_annotation_percentage(df, plot=True):
    """
    Checks annotation coverage across all recorded sessions.

    :param df: pandas.DataFrame (must contain 'session', 'annotation', 'seconds_elapsed')
    :param plot: Boolean flag to visualize results with matplotlib.
    :return: Summary metrics dict.
    """
    if df.empty:
        return {}

    # Vectorized fast aggregation
    stats = df.groupby("session").agg(
        max_seconds=("seconds_elapsed", "max"),
        has_annotation=("annotation", "any"),
    )
    stats["minutes"] = stats["max_seconds"] / 60.0

    annotated = stats[stats["has_annotation"]]
    not_annotated = stats[~stats["has_annotation"]]

    ann_min = annotated["minutes"].sum()
    not_ann_min = not_annotated["minutes"].sum()
    total_min = ann_min + not_ann_min

    result = {
        "annotated_sessions": len(annotated),
        "not_annotated_sessions": len(not_annotated),
        "annotated_minutes": round(ann_min, 2),
        "not_annotated_minutes": round(not_ann_min, 2),
        "total_minutes": round(total_min, 2),
        "annotated_pct": round(ann_min / total_min * 100, 2) if total_min > 0 else 0.0,
    }

    print(
        f"Annotated: {result['annotated_sessions']} sess. ({result['annotated_minutes']} min) | "
        f"Not annotated: {result['not_annotated_sessions']} sess. ({result['not_annotated_minutes']} min) | "
        f"Coverage: {result['annotated_pct']}% of total ({result['total_minutes']} min)"
    )

    if plot and total_min > 0:
        labels = ["Annotated", "Not Annotated"]
        sizes = [result["annotated_minutes"], result["not_annotated_minutes"]]
        colors = ["#075e2b", "#972518"]
        explode = (0.05, 0)

        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
            wedgeprops={"edgecolor": "black"},
        )
        plt.title(
            f"Annotated vs Not Annotated Recording Time\nTotal: {result['total_minutes']} min"
        )
        plt.tight_layout()
        plt.show()

    return result


def train_test_split_by_annotation(df):
    """
    Splits the dataset into train and test sets based on annotation presence.

    - Train: All sessions without annotations (annotation == False)
    - Test : All sessions with annotations (annotation == True)
    """
    if df.empty:
        print("Warning: Input DataFrame is empty.")
        return pd.DataFrame(), pd.DataFrame()

    train_df = df[~df["annotation"]]
    test_df = df[df["annotation"]]

    train_sessions = train_df["session"].nunique()
    test_sessions = test_df["session"].nunique()

    print(
        f"Split Summary:\n"
        f" - Train set (Not Annotated): {train_sessions} sessions, {len(train_df)} rows\n"
        f" - Test set  (Annotated)    : {test_sessions} sessions, {len(test_df)} rows"
    )

    return train_df, test_df