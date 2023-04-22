import numpy as np
import gzip
from io import StringIO


def parse_header_of_csv(csv_str):
    """
    Helper function to parse a user's data file.

    Written by Yonatan Vaizman, May 2017.

    :param csv_str: CSV header of the user's data file.
    :return:
        feature_names: Names of the sensors (e.g. accelerometer).
        label_names: Names of the context labels (i.e. activity).
    """
    # Isolate the headline columns:
    headline = csv_str[:csv_str.index('\n')]
    columns = headline.split(',')

    # The first column should be timestamped:
    assert columns[0] == 'timestamp'
    # The last column should be label_source:
    assert columns[-1] == 'label_source'

    # Search for the column of the first label:
    first_label_ind = 0  # dummy assignment
    for (ci, col) in enumerate(columns):
        if col.startswith('label:'):
            first_label_ind = ci
            break
        pass

    # Feature columns come after timestamp and before the labels:
    feature_names = columns[1:first_label_ind]
    # Then come the labels, till the one-before-last column:
    label_names = columns[first_label_ind:-1]
    for (li, label) in enumerate(label_names):
        # In the CSV the label names appear with prefix 'label:', but we don't need it after reading the data:
        assert label.startswith('label:');
        label_names[li] = label.replace('label:', '')
        pass;

    return feature_names, label_names


def parse_body_of_csv(csv_str, n_features):
    """
    Helper function to parse a user's data file.

    Written by Yonatan Vaizman, May 2017.

    :param csv_str: CSV representation of features.
    :param n_features: Number of features.
    :return:
        X: Sensor measurements i.e. features corresponding to the user.
        Y: Context labels for measurements at each timestamp.
        M: Whether a label is missing for a particular timestamp's measurements.
        timestamps: Timestamps when the sensor measurements were recorded.
    """
    # Read the entire CSV body into a single numeric matrix:
    full_table = np.loadtxt(StringIO(csv_str), delimiter=',', skiprows=1)

    # Timestamp is the primary key for the records (examples):
    timestamps = full_table[:, 0].astype(int)

    # Read the sensor features:
    X = full_table[:, 1:(n_features + 1)]

    # Read the binary label values, and the 'missing label' indicators:
    trinary_labels_mat = full_table[:, (n_features + 1):-1]  # This should have values of either 0., 1. or NaN
    M = np.isnan(trinary_labels_mat)  # M is the missing label matrix
    Y = np.where(M, 0, trinary_labels_mat) > 0.  # Y is the label matrix

    return X, Y, M, timestamps


def read_user_data(directory, uuid):
    """
    Read the data (precomputed sensor-features and labels) for a user.
    This function assumes the user's data file is present.

    Modified version of the function written by Yonatan Vaizman, May 2017.

    :param directory: Parent directory where user's data file is stored.
    :param uuid: UUID corresponding to the user whose data file is to be read.
    :return:
        X: Sensor measurements i.e. features corresponding to the user.
        Y: Context labels for measurements at each timestamp.
        M: Whether a label is missing for a particular timestamp's measurements.
        timestamps: Timestamps when the sensor measurements were recorded.
        feature_names: Names of the sensors (e.g. accelerometer).
        label_names: Names of the context labels (i.e. activity).
    """
    user_data_file = f'{directory}/{uuid}.features_labels.csv.gz'

    # Read the entire csv file of the user:
    with gzip.open(user_data_file, 'rb') as fid:
        csv_str = fid.read()
        pass

    csv_str = csv_str.decode('UTF-8')

    (feature_names, label_names) = parse_header_of_csv(csv_str)
    n_features = len(feature_names)
    (X, Y, M, timestamps) = parse_body_of_csv(csv_str, n_features)

    return X, Y, M, timestamps, feature_names, label_names


def get_sensor_names_from_features(feature_names):
    feat_sensor_names = np.array([None for feat in feature_names])
    for (fi, feat) in enumerate(feature_names):
        if feat.startswith('raw_acc'):
            feat_sensor_names[fi] = 'Acc'
            pass
        elif feat.startswith('proc_gyro'):
            feat_sensor_names[fi] = 'Gyro'
            pass
        elif feat.startswith('raw_magnet'):
            feat_sensor_names[fi] = 'Magnet'
            pass
        elif feat.startswith('watch_acceleration'):
            feat_sensor_names[fi] = 'WAcc'
            pass
        elif feat.startswith('watch_heading'):
            feat_sensor_names[fi] = 'Compass'
            pass
        elif feat.startswith('location'):
            feat_sensor_names[fi] = 'Loc'
            pass
        elif feat.startswith('location_quick_features'):
            feat_sensor_names[fi] = 'Loc'
            pass
        elif feat.startswith('audio_naive'):
            feat_sensor_names[fi] = 'Aud'
            pass
        elif feat.startswith('audio_properties'):
            feat_sensor_names[fi] = 'AP'
            pass
        elif feat.startswith('discrete'):
            feat_sensor_names[fi] = 'PS'
            pass
        elif feat.startswith('lf_measurements'):
            feat_sensor_names[fi] = 'LF'
            pass
        else:
            raise ValueError("!!! Unsupported feature name: %s" % feat)

        pass

    return feat_sensor_names


def get_uuids_from_filepaths(uuid_filepaths):
    uuids = []
    for filepath in uuid_filepaths:
        with open(filepath, 'r', newline='\n') as f:
            raw_uuids = f.readlines()
            print(f"Adding {len(raw_uuids)} UUIDS from {filepath}")
            clean_uuids = [uuid.strip('\n') for uuid in raw_uuids]
            uuids.extend(clean_uuids)
    return uuids


def project_features_to_selected_sensors(X, feat_sensor_names, sensors_to_use):
    use_feature = np.zeros(len(feat_sensor_names), dtype=bool)
    for sensor in sensors_to_use:
        is_from_sensor = (feat_sensor_names == sensor)
        use_feature = np.logical_or(use_feature, is_from_sensor)
        pass;
    X = X[:, use_feature]
    return X


def estimate_standardization_params(X_train):
    mean_vec = np.nanmean(X_train, axis=0)
    std_vec = np.nanstd(X_train, axis=0)
    return (mean_vec, std_vec)


def standardize_features(X, mean_vec, std_vec):
    # Subtract the mean, to centralize all features around zero:
    X_centralized = X - mean_vec.reshape((1, -1))
    # Divide by the standard deviation, to get unit-variance for all features:
    # * Avoid dividing by zero, in case some feature had estimate of zero variance
    normalizers = np.where(std_vec > 0., std_vec, 1.).reshape((1, -1))
    X_standard = X_centralized / normalizers
    return X_standard


def preprocess_data(X, Y, M, feat_sensor_names, label_names, sensors_to_use, target_labels):
    # Project the feature matrix to the features from the desired sensors:
    X = project_features_to_selected_sensors(X, feat_sensor_names, sensors_to_use)

    # Select target labels
    label_inds = [label_names.index(t) for t in target_labels]
    y = Y[:, label_inds]

    # Select only the examples that are not missing the target labels
    for label_ind in label_inds:
        missing_label = M[:, label_ind]
        existing_label = np.logical_not(missing_label)
        X = X[existing_label, :]
        y = y[existing_label]
        M = M[existing_label]

    # Also, there may be missing sensor-features (represented in the data as NaN).
    # You can handle those by imputing a value of zero (since we standardized,
    # this is equivalent to assuming average value).
    X[np.isnan(X)] = 0.

    return X, y


def get_train_and_test_data(directory, train_uuids, test_uuids, sensors_to_use, target_labels):
    X_train, y_train = [], []
    for uuid in train_uuids:
        X, Y, M, timestamps, feature_names, label_names = read_user_data(directory, uuid)
        feat_sensor_names = get_sensor_names_from_features(feature_names)
        X, y = preprocess_data(X, Y, M, feat_sensor_names, label_names, sensors_to_use, target_labels)
        X_train.extend(X)
        y_train.extend(y)
    print(f"Added data from {len(train_uuids)} users to train data.")

    # Standardize the features (subtract mean and divide by standard deviation),
    # so that all their values will be roughly in the same range:
    mean_vec, std_vec = estimate_standardization_params(X_train)
    X_train = standardize_features(X_train, mean_vec, std_vec)

    X_test, y_test = [], []
    for uuid in test_uuids:
        X, Y, M, timestamps, feature_names, label_names = read_user_data(directory, uuid)
        feat_sensor_names = get_sensor_names_from_features(feature_names)
        X, y = preprocess_data(X, Y, M, feat_sensor_names, label_names, sensors_to_use, target_labels)
        X_test.extend(X)
        y_test.extend(y)
    print(f"Added data from {len(test_uuids)} users to test data.")

    # Apply same standardization for test data
    X_test = standardize_features(X_test, mean_vec, std_vec)

    X_train = np.array(X_train, np.float32)
    y_train = np.array(y_train, dtype=int)
    X_test = np.array(X_test, np.float32)
    y_test = np.array(y_test, dtype=int)

    return X_train, y_train, X_test, y_test
