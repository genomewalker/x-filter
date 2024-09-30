import os
import numpy as np
from functools import partial
import concurrent.futures


def slice_mmap(arr, arr_name, mask, mmap_folder, dtype):
    mmap_path = os.path.join(mmap_folder, f"{arr_name}.npy")
    filtered_arr = arr[mask]
    sliced_arr = np.memmap(mmap_path, dtype=dtype, mode="w+", shape=filtered_arr.shape)
    sliced_arr[:] = filtered_arr
    sliced_arr.flush()
    del sliced_arr  # Close the memmap file


def parallel_slice_mmap(
    numpy_arrays,
    target_subjects,
    unique_subjects,
    inverse_indices,
    mmap_folder,
    num_threads=1,
):
    os.makedirs(mmap_folder, exist_ok=True)

    # Sort target_subjects to ensure consistent order
    target_subjects = np.sort(target_subjects)

    # Create the mask
    target_indices = np.where(np.isin(unique_subjects, target_subjects))[0]
    if len(target_indices) == 0:
        raise ValueError(
            f"None of the target subjects {target_subjects} were found in unique_subjects."
        )
    mask = np.isin(inverse_indices, target_indices)

    slice_mmap_partial = partial(slice_mmap, mask=mask, mmap_folder=mmap_folder)

    arrays_to_process = [
        (numpy_arrays["subject_numeric_id"], "subject_numeric_id", np.int64),
        (numpy_arrays["subjectStart"], "subjectStart", np.int32),
        (numpy_arrays["subjectEnd"], "subjectEnd", np.int32),
        (numpy_arrays["alnLength"], "alnLength", np.int32),
        (numpy_arrays["qlen"], "qlen", np.int32),
        (numpy_arrays["percIdentity"], "percIdentity", np.float32),
        (numpy_arrays["slen"], "slen", np.int32),
        (numpy_arrays["bitScore"], "bitScore", np.float32),
        (numpy_arrays["query_numeric_id"], "query_numeric_id", np.int64),
        (numpy_arrays["row_hash"], "row_hash", np.int64),
    ]

    # Sort arrays_to_process to ensure consistent order
    arrays_to_process.sort(key=lambda x: x[1])

    if num_threads > 1:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_threads
        ) as executor:
            futures = [
                executor.submit(slice_mmap_partial, arr=arr, arr_name=name, dtype=dtype)
                for arr, name, dtype in arrays_to_process
            ]
            # Wait for all futures to complete in the order they were submitted
            for future in concurrent.futures.as_completed(futures):
                future.result()  # This will raise any exceptions that occurred
    else:
        # Single-threaded execution
        for arr, name, dtype in arrays_to_process:
            slice_mmap_partial(arr=arr, arr_name=name, dtype=dtype)

    # Load memmapped arrays in a consistent order
    for _, name, dtype in arrays_to_process:
        mmap_path = os.path.join(mmap_folder, f"{name}.npy")
        numpy_arrays[name] = np.memmap(mmap_path, dtype=dtype, mode="r")

    return numpy_arrays
