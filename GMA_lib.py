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
# ==================================================================
# PREPROCESSING FROM HERE ON OUT(in line w the code above): trimming, missing values, band-pass filter, windowing, EDA
# -- operating on the DataFrame returned by build_session_dataset(), whose
# columns are: id, session, seconds_elapsed, accel_x/y/z, gyro_x/y/z,
# gravity_x/y/z, type_device, id_device, annotation.
#
# Every step below groups by "session" so trimming/filtering/interpolation/
# windowing never blends two different recordings together.
# ==================================================================

from scipy.signal import butter, filtfilt

FS = 100.0  # nominal sampling rate

ACCEL_COLS = ["accel_x", "accel_y", "accel_z"]
GYRO_COLS = ["gyro_x", "gyro_y", "gyro_z"]
GRAVITY_COLS = ["gravity_x", "gravity_y", "gravity_z"]
ALL_SENSOR_COLS = ACCEL_COLS + GYRO_COLS + GRAVITY_COLS


def trim_dataframe(df: pd.DataFrame, trim_start_sec: float = 0, trim_end_sec: float = 0,
                    fs: float = FS, session_col: str = "session") -> pd.DataFrame:
    """
    in: dataframe (multi-session), seconds to trim from the front and back
        of EACH session, sampling rate
    out: trimmed dataframe (all sessions concatenated back together)
    """
    start_samples = int(trim_start_sec * fs)
    end_samples = int(trim_end_sec * fs)

    trimmed_sessions = []
    for session_id, group in df.groupby(session_col, sort=False):
        group = group.reset_index(drop=True)
        if start_samples + end_samples >= len(group):
            print(f"  WARNING: session {session_id} too short to trim ({len(group)} rows) — skipping.")
            continue
        trimmed = group.iloc[start_samples:-end_samples] if end_samples > 0 else group.iloc[start_samples:]
        trimmed_sessions.append(trimmed)

    result = pd.concat(trimmed_sessions, ignore_index=True)
    print(f"Rows before trimming: {len(df)} -> after: {len(result)}")
    return result


def handle_missing_values(df: pd.DataFrame, columns: list = ALL_SENSOR_COLS,
                           session_col: str = "session", strategy: str = "interpolate") -> pd.DataFrame:
    """
    in: dataframe, columns to check/fill, strategy ("interpolate"/"drop"/"ffill")
    out: dataframe with missing values handled (per session, so interpolation
         never pulls in a sample from a different recording)
    """
    total_missing_before = df[columns].isna().sum().sum()

    filled_sessions = []
    for session_id, group in df.groupby(session_col, sort=False):
        group = group.copy()
        if strategy == "interpolate":
            group[columns] = group[columns].interpolate(method="linear", limit_direction="both")
        elif strategy == "drop":
            group = group.dropna(subset=columns)
        elif strategy == "ffill":
            group[columns] = group[columns].ffill().bfill()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        filled_sessions.append(group)

    result = pd.concat(filled_sessions, ignore_index=True)
    total_missing_after = result[columns].isna().sum().sum()
    print(f"Missing values: {total_missing_before} -> {total_missing_after} (strategy={strategy})")
    return result


def apply_bandpass_filter(df: pd.DataFrame, columns: list = ALL_SENSOR_COLS,
                           low_cutoff: float = 5.0, high_cutoff: float = 30.0,
                           fs: float = FS, order: int = 4,
                           session_col: str = "session") -> pd.DataFrame:
    """
    in: dataframe, columns to filter (defaults to accel+gyro+gravity,
        matching G.ipynb), band cutoffs, sampling rate
    out: dataframe with added *_filt columns, filtered per session
    """
    nyquist = fs / 2
    low_norm = low_cutoff / nyquist
    high_norm = min(high_cutoff, nyquist * 0.95) / nyquist
    b, a = butter(order, [low_norm, high_norm], btype="band", analog=False)
    padlen = 3 * (max(len(b), len(a)) - 1)

    filtered_sessions = []
    for session_id, group in df.groupby(session_col, sort=False):
        group = group.copy()
        if len(group) <= padlen:
            print(f"  WARNING: session {session_id} too short to filter ({len(group)} rows) — keeping raw.")
            for col in columns:
                group[f"{col}_filt"] = group[col]
        else:
            for col in columns:
                group[f"{col}_filt"] = filtfilt(b, a, group[col].values)
        filtered_sessions.append(group)

    return pd.concat(filtered_sessions, ignore_index=True)


def windowing(df: pd.DataFrame, window_length: float, overlap_percentage: float,
              fs: float = FS, session_col: str = "session"):
    """
    in: dataframe, window_length (seconds), overlap_percentage (0-100), fs
    out: (number_of_windows, dataframe) -- long-format, with a 'window_id'
         column (unique across the whole multi-session dataset), windowed
         per session so a window never spans two different recordings
    """
    window_size = int(round(window_length * fs))
    step_size = max(1, int(round(window_size * (1 - overlap_percentage / 100))))

    window_frames = []
    window_id = 0
    for session_id, group in df.groupby(session_col, sort=False):
        group = group.reset_index(drop=True)
        n_samples = len(group)
        start = 0
        while start + window_size <= n_samples:
            chunk = group.iloc[start:start + window_size].copy()
            chunk["window_id"] = window_id
            window_frames.append(chunk)
            window_id += 1
            start += step_size

    if not window_frames:
        return 0, pd.DataFrame(columns=list(df.columns) + ["window_id"])

    return window_id, pd.concat(window_frames, ignore_index=True)


def eda(df: pd.DataFrame, columns: list = ALL_SENSOR_COLS,
        time_col: str = "seconds_elapsed", session_col: str = "session"):
    """
    in: dataframe, columns to summarize
    out: none (prints summary stats + missing-value counts; plots one
         example session's time series per column, plus a correlation
         heatmap across all sessions combined)
    """
    print(df[columns].describe())
    print("\nMissing values per column:")
    print(df[columns].isna().sum())
    print(f"\nSessions: {df[session_col].nunique()}")
    print(f"Rows per session:\n{df.groupby(session_col).size()}")

    example_session = df[session_col].iloc[0]
    example_df = df[df[session_col] == example_session]

    fig, axes = plt.subplots(len(columns), 1, figsize=(10, 2.0 * len(columns)), sharex=True)
    for ax, col in zip(axes, columns):
        ax.plot(example_df[time_col], example_df[col], linewidth=0.5)
        ax.set_ylabel(col)
    axes[0].set_title(f"session {example_session} (example)")
    axes[-1].set_xlabel(time_col)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[columns].corr()
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(columns))); ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=90); ax.set_yticklabels(columns)
    fig.colorbar(im)
    ax.set_title("correlation matrix (all sessions combined)")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# Example flow, continuing straight from build_session_dataset()
# ------------------------------------------------------------------
# dataset_df = build_session_dataset(input_path=DATASET_PATH, save_csv=True)
# df_trimmed  = trim_dataframe(dataset_df, trim_start_sec=3.0, trim_end_sec=3.0, fs=FS)
# df_clean    = handle_missing_values(df_trimmed)
# df_filtered = apply_bandpass_filter(df_clean)
# n_windows, windowed_df = windowing(df_filtered, window_length=1.0, overlap_percentage=80, fs=FS)
# eda(df_filtered)


# ==================================================================
# FEATURE EXTRACTION & FEATURE SELECTION FROM HERE ON OUT:
# -- operating on the DataFrame returned by windowing()
# ==================================================================


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.fft import rfft, rfftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def extract_features(windowed_df: pd.DataFrame, fs: float = FS) -> pd.DataFrame:
    """
    in: dataframe (windowed sensor data), sampling rate (fs)
    out: dataframe with extracted time-domain and frequency-domain (FFT) features,
         retaining 'window_id' and 'session' columns
    """
    print("\n--- 6. Feature Extraction (5ARE0 Course Pipeline) ---")

    # 1. Compute magnitudes and vertical acceleration (orientation-invariant)
    accel_cols = ["accel_x_filt", "accel_y_filt", "accel_z_filt"]
    gyro_cols = ["gyro_x_filt", "gyro_y_filt", "gyro_z_filt"]
    grav_cols = ["gravity_x", "gravity_y", "gravity_z"]

    windowed_df = windowed_df.copy()
    windowed_df["accel_mag"] = np.sqrt((windowed_df[accel_cols] ** 2).sum(axis=1))
    windowed_df["gyro_mag"] = np.sqrt((windowed_df[gyro_cols] ** 2).sum(axis=1))

    # Pure vertical component: projection onto gravity vector
    dot_p = (
        windowed_df["accel_x_filt"] * windowed_df["gravity_x"]
        + windowed_df["accel_y_filt"] * windowed_df["gravity_y"]
        + windowed_df["accel_z_filt"] * windowed_df["gravity_z"]
    )
    grav_mag = np.sqrt((windowed_df[grav_cols] ** 2).sum(axis=1))
    windowed_df["accel_vert"] = dot_p / np.where(grav_mag == 0, 1.0, grav_mag)

    # 2. Vectorized feature extraction per window_id
    signals = ["accel_mag", "gyro_mag", "accel_vert"]
    grouped = windowed_df.groupby("window_id", sort=True)

    feature_dict = {
        "window_id": grouped["window_id"].first(),
        "session": grouped["session"].first(),
    }

    # Time-Domain features
    for s in signals:
        feature_dict[f"{s}_mean"] = grouped[s].mean()
        feature_dict[f"{s}_std"] = grouped[s].std()
        feature_dict[f"{s}_min"] = grouped[s].min()
        feature_dict[f"{s}_max"] = grouped[s].max()
        feature_dict[f"{s}_rms"] = np.sqrt(grouped[s].apply(lambda x: np.mean(x**2)))
        feature_dict[f"{s}_ptp"] = feature_dict[f"{s}_max"] - feature_dict[f"{s}_min"]
        feature_dict[f"{s}_kurtosis"] = grouped[s].apply(lambda x: stats.kurtosis(x))
        feature_dict[f"{s}_skew"] = grouped[s].apply(lambda x: stats.skew(x))

    # 3. Frequency-Domain (FFT) features on accel_vert
    def extract_fft_features(series):
        x = series.values
        n_samples = len(x)
        if n_samples < 2:
            return pd.Series([0.0, 0.0], index=["dom_freq", "vibr_energy"])

        x_centered = x - np.mean(x)
        yf = np.abs(rfft(x_centered)) * (2.0 / n_samples)
        xf = rfftfreq(n_samples, 1.0 / fs)

        dom_freq = xf[np.argmax(yf)] if len(yf) > 0 else 0.0
        mask = (xf >= 5.0) & (xf <= 30.0)
        vibr_energy = np.sum(yf[mask] ** 2) if np.any(mask) else 0.0

        return pd.Series([dom_freq, vibr_energy], index=["dom_freq", "vibr_energy"])

    fft_feats = grouped["accel_vert"].apply(extract_fft_features).unstack()
    feature_dict["accel_vert_dom_freq"] = fft_feats["dom_freq"]
    feature_dict["accel_vert_vibr_energy"] = fft_feats["vibr_energy"]

    x_all = pd.DataFrame(feature_dict).reset_index(drop=True)
    print(f"Extracted {x_all.shape[0]} windows with {x_all.shape[1] - 2} features each.")
    print(x_all.head())

    return x_all


def select_features_and_pca(x_all: pd.DataFrame, variance_threshold: float = 0.95,
                            output_path=None, save_csv: bool = True) -> pd.DataFrame:
    """
    in: dataframe (extracted features), variance_threshold (float), optional output_path,
        save_csv flag
    out: dataframe with principal components (PC_1, PC_2, ...) explaining the target
         cumulative variance, plus 'window_id' and 'session' columns
    """
    print("\n--- 7. Feature Selection: Correlation & PCA ---")
    output_path = Path(output_path) if output_path else Path("OutputData")

    feature_cols = [c for c in x_all.columns if c not in ["window_id", "session"]]
    x = x_all[feature_cols].copy()
    x = x.replace([np.inf, -np.inf], np.nan).fillna(x.mean())

    # Step 1: Multicollinearity filtering (Pearson correlation > 0.90)
    corr_matrix = x.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    cols_to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.90)]

    x_uncorr = x.drop(columns=cols_to_drop)
    print(f"Original Features: {x.shape[1]}")
    print(f"Features after correlation filter (r < 0.90): {x_uncorr.shape[1]}")
    print(f"Dropped as redundant: {cols_to_drop}")

    # Step 2: Standardization & PCA
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_uncorr)

    pca = PCA(n_components=variance_threshold, random_state=42)
    x_pca = pca.fit_transform(x_scaled)

    print(f"\n[PCA] Dimensions reduced to {pca.n_components_} principal components.")
    print(f"Total Explained Variance: {np.sum(pca.explained_variance_ratio_):.2%}")

    # Scree Plot
    plt.figure(figsize=(8, 4))
    plt.plot(
        range(1, pca.n_components_ + 1),
        np.cumsum(pca.explained_variance_ratio_),
        marker="o",
        color="navy",
    )
    plt.axhline(
        y=variance_threshold,
        color="r",
        linestyle="--",
        label=f"{int(variance_threshold * 100)}% Explained Variance",
    )
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Variance")
    plt.title("PCA Explained Variance Ratio (Session 4)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    df_pca = pd.DataFrame(x_pca, columns=[f"PC_{i+1}" for i in range(pca.n_components_)])
    df_pca["window_id"] = x_all["window_id"]
    df_pca["session"] = x_all["session"]

    if save_csv:
        output_path.mkdir(parents=True, exist_ok=True)
        x_all.to_csv(output_path / "extracted_features.csv", index=False)
        print(f"Saved: {output_path / 'extracted_features.csv'}")

        df_pca.to_csv(output_path / "X_selected_pca.csv", index=False)
        print(f"Saved: {output_path / 'X_selected_pca.csv'}")

    return df_pca


# ------------------------------------------------------------------
# Example flow, continuing straight from windowing()
# ------------------------------------------------------------------
# x_all  = extract_features(windowed_df, fs=FS)
# df_pca = select_features_and_pca(x_all, variance_threshold=0.95, output_path=OUTPUT_PATH, save_csv=True)