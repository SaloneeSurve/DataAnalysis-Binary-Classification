from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def _read_csv_file(path):
    """
    Reads a CSV file from disk into a DataFrame.

    param path: 

    :return: None if the file is missing, empty or unreadable.
    """
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(path)
        return None if df.empty else df
    except pd.errors.EmptyDataError:
        return None

def data_ingestion(input_path, output_path=None, save_csv=True):
    """
    Concatenates matching CSV file types across all session directories into single DataFrames

    param input_path:       
    param output_path:
    param save_csv:         

    :return: Merged DataFrame 
    """
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path / "output"

    collected = {}

    for session_dir in filter(Path.is_dir, sorted(input_path.iterdir())):
        for csv_path in session_dir.glob("*.csv"):
            df = _read_csv_file(csv_path)
            if df is not None:
                df.insert(0, "recording_id", session_dir.name)
                collected.setdefault(csv_path.stem, []).append(df)

    if not collected:
        print(f"No valid session data found in {input_path}")
        return {}

    dataframes = {
        key: pd.concat(dfs, ignore_index=True) for key, dfs in collected.items()
    }

    if save_csv:
        output_path.mkdir(parents=True, exist_ok=True)
        for key, df in dataframes.items():
            df.to_csv(output_path / f"{key}.csv", index=False)
            print(f"Saved: {key}.csv ({len(df)} rows)")

    return dataframes

def build_session_dataset(input_path, output_path=None, save_csv=True):
    """
    Merges sensor axes (Accelerometer, Gyroscope, Gravity), metadata, and annotations. (All data coming from Sensor Log)

    param input_path:       
    param output_path:
    param save_csv: 

    :return: DataFrame, All sensor rec plus other needed infos
    """
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path / "output"

    sensor_files = {
        "Accelerometer.csv": "accel",
        "Gyroscope.csv": "gyro",
        "Gravity.csv": "gravity",
    }
    merge_keys = ["seconds_elapsed"]
    sessions = []

    session_dirs = [p for p in sorted(input_path.iterdir()) if p.is_dir()]

    for session_num, session_dir in enumerate(session_dirs, start=1):
        # 1. Read and rename sensors
        dfs_to_merge = []
        for fname, prefix in sensor_files.items():
            df = _read_csv_file(session_dir / fname)
            if df is not None:
                cols_to_drop = [c for c in ["recording_id", "time"] if c in df.columns]
                if cols_to_drop:
                    df = df.drop(columns=cols_to_drop)

                df = df.rename(
                    columns={
                        c: f"{prefix}_{c}" for c in df.columns if c not in merge_keys
                    }
                )
                dfs_to_merge.append(df)

        if not dfs_to_merge:
            continue

        # 2. Merge Accelerometer, Gyro e Gravity
        merged = dfs_to_merge[0]
        for df in dfs_to_merge[1:]:
            merged = pd.merge(merged, df, on=merge_keys, how="inner")

        # 3. Extract Metadata and Annotations
        meta_df = _read_csv_file(session_dir / "Metadata.csv")
        type_device = (
            meta_df.iloc[0].get("device name") if meta_df is not None else None
        )
        id_device = (
            meta_df.iloc[0].get("device id") if meta_df is not None else None
        )

        annot_file = session_dir / "Annotation.csv"
        has_annotation = annot_file.exists() and annot_file.stat().st_size > 0

        # 4. Add new col
        merged["type_device"] = type_device
        merged["id_device"] = id_device
        merged["annotation"] = has_annotation
        merged.insert(0, "session", session_num)

        sessions.append(merged)

    if not sessions:
        return pd.DataFrame()

    # Concat and assign global ID
    dataset = pd.concat(sessions, ignore_index=True)
    dataset.insert(0, "id", dataset.index)

    if save_csv:
        output_path.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(output_path / "sessions_dataset.csv", index=False)
        print(
            f"Saved: sessions_dataset.csv ({len(dataset)} rows, {len(sessions)} Saved)"
        )

    return dataset

def check_annotation_percentage(df, plot=True):
    """
    Checks how many sessions have annotations and how many don't

    :param df : pandas.DataFrame, (must contain the columns 'session', 'annotation', 'seconds_elapsed')

    :return:    dict{
                        'annotated_sessions': int,
                        'not_annotated_sessions': int,
                        'annotated_minutes': float,
                        'not_annotated_minutes': float,
                        'total_minutes': float,
                        'annotated_pct': float,   # % of minutes that are annotated
                    }
    """
    if df.empty:
        return {}

    # Calculate duration (minutes) and annotation status per session
    stats = (
        df.groupby("session")
        .agg(
            minutes=("seconds_elapsed", lambda x: x.max() / 60),
            has_annotation=("annotation", "any"),
        )
        .reset_index()
    )

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
        "annotated_pct": round(ann_min / total_min * 100, 2)
        if total_min > 0
        else 0.0,
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
        explode = (0.05, 0)  # Highlight the annotated slice

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



#   DONT USE THSI ONE
#   DONT USE THSI ONE
#   DONT USE THSI ONE
def train_test_split_by_annotation(df):
    """Splits the dataset into train and test sets based on annotation presence.

    - Train: All sessions without annotations (annotation == False)
    - Test : All sessions with annotations (annotation == True)
    """
    if df.empty:
        print("Warning: Input DataFrame is empty.")
        return pd.DataFrame(), pd.DataFrame()

    train_df = df[~df["annotation"]].copy()
    test_df = df[df["annotation"]].copy()

    train_sessions = train_df["session"].nunique()
    test_sessions = test_df["session"].nunique()

    print(
        f"Split Summary:\n"
        f" - Train set (Not Annotated): {train_sessions} sessions, {len(train_df)} rows\n"
        f" - Test set  (Annotated)    : {test_sessions} sessions, {len(test_df)} rows"
    )

    return train_df, test_df