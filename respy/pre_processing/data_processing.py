import pandas as pd

from respy.python.shared.shared_auxiliary import dist_class_attributes
from respy.pre_processing.data_checking import check_estimation_dataset
from respy.python.shared.shared_constants import DATA_FORMATS_EST
from respy.python.shared.shared_constants import DATA_LABELS_EST


def process_dataset(respy_obj):
    """Process the dataset from disk."""
    # TODO: we should make this function more robust by sorting the columns and rows of
    # the dataframe in the order we need for the likelihood function.
    num_agents_est, file_est, edu_spec, num_periods = dist_class_attributes(
        respy_obj, "num_agents_est", "file_est", "edu_spec", "num_periods"
    )

    # Process dataset from files.
    data_frame = pd.read_csv(file_est, delim_whitespace=True, header=0, na_values=".")
    data_frame.set_index(["Identifier", "Period"], drop=False, inplace=True)

    # We want to allow to estimate with only a subset of periods in the sample.
    cond = data_frame["Period"] < num_periods
    data_frame = data_frame[cond]

    # Only keep the information that is relevant for the estimation.
    # Once that is done,  impose some type restrictions.
    data_frame = data_frame[DATA_LABELS_EST]
    data_frame = data_frame.astype(DATA_FORMATS_EST)

    # We want to restrict the sample to meet the specified initial conditions.
    cond = data_frame["Years_Schooling"].loc[:, 0].isin(edu_spec["start"])
    data_frame.set_index(["Identifier"], drop=False, inplace=True)
    data_frame = data_frame.loc[cond]

    # We now subset the dataframe to include only the number of agents that are
    # requested for the estimation. However, this requires to adjust the
    # num_agents_est as the dataset might actually be smaller as we restrict
    # initial conditions.
    data_frame = data_frame.loc[data_frame.index.unique()[:num_agents_est]]
    data_frame.set_index(["Identifier", "Period"], drop=False, inplace=True)

    # We need to update the number of individuals for the estimation as the
    # whole dataset might actually be lower.
    num_agents_est = data_frame["Identifier"].nunique()

    respy_obj.unlock()
    respy_obj.set_attr("num_agents_est", num_agents_est)
    respy_obj.lock()

    # Check the dataset against the initialization files.
    check_estimation_dataset(data_frame, respy_obj)

    # Finishing
    return data_frame
