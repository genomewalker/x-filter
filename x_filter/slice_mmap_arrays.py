from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Union, Any, List, Tuple

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

# Type aliases for better readability
ArrayDict = Dict[str, NDArray]
DType = np.dtype[Any]


def slice_mmap(
    arr: NDArray,
    arr_name: str,
    mask: NDArray[np.bool_],
    mmap_folder: Union[str, Path],
    dtype: DType,
    chunk_size: int = 10_000_000,
) -> None:
    """Process a single array in chunks using memory mapping."""
    mmap_folder = Path(mmap_folder)
    mmap_path = mmap_folder / f"{arr_name}.dat"
    total_filtered = np.count_nonzero(mask)

    out_mmap = None
    try:
        out_mmap = np.memmap(mmap_path, dtype=dtype, mode="w+", shape=(total_filtered,))

        output_idx = 0
        for chunk_start in range(0, len(arr), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(arr))
            chunk_mask = mask[chunk_start:chunk_end]
            chunk_data = arr[chunk_start:chunk_end][chunk_mask]

            if len(chunk_data) > 0:
                chunk_end_idx = output_idx + len(chunk_data)
                out_mmap[output_idx:chunk_end_idx] = chunk_data
                output_idx = chunk_end_idx

            del chunk_data

        out_mmap.flush()
    finally:
        if out_mmap is not None:
            del out_mmap


def process_chunk(
    arrays_to_process: List[Tuple[NDArray, str, np.dtype]],
    chunk_start: int,
    chunk_size: int,
    mask: NDArray[np.bool_],
    mmap_folder: Path,
) -> None:
    """Process a chunk of all arrays."""
    chunk_end = min(chunk_start + chunk_size, len(mask))
    chunk_mask = mask[chunk_start:chunk_end]

    for arr, name, dtype in arrays_to_process:
        chunk_data = None
        out_mmap = None
        try:
            chunk_data = arr[chunk_start:chunk_end].copy()
            filtered_data = chunk_data[chunk_mask]

            output_path = mmap_folder / "chunks" / f"{name}_{chunk_start}.dat"
            out_mmap = np.memmap(
                output_path, dtype=dtype, mode="w+", shape=(len(filtered_data),)
            )
            out_mmap[:] = filtered_data
            out_mmap.flush()
        finally:
            if out_mmap is not None:
                del out_mmap
            if chunk_data is not None:
                del chunk_data


def combine_chunks(
    chunk_files: List[Path],
    output_path: Path,
    dtype: np.dtype,
    buffer_size: int = 50_000_000,  # Increased buffer size
) -> np.memmap:
    """Combine chunk files into a single memory-mapped file using efficient buffering."""
    # Calculate total size
    total_size = sum(os.path.getsize(f) // dtype.itemsize for f in chunk_files)

    # Create output memory map
    output_mmap = np.memmap(output_path, dtype=dtype, mode="w+", shape=(total_size,))
    pos = 0

    # Process chunks in batches to reduce file operations
    batch_size = 5  # Number of chunks to process simultaneously
    for batch_start in range(0, len(chunk_files), batch_size):
        batch_end = min(batch_start + batch_size, len(chunk_files))
        batch_files = chunk_files[batch_start:batch_end]
        batch_mmaps = []

        try:
            # Open all files in batch
            for chunk_file in batch_files:
                batch_mmaps.append(np.memmap(chunk_file, dtype=dtype, mode="r"))

            # Copy data from each mmap in batch
            for chunk_mmap in batch_mmaps:
                chunk_len = len(chunk_mmap)
                if chunk_len <= buffer_size:
                    # If chunk is smaller than buffer, copy in one operation
                    output_mmap[pos : pos + chunk_len] = chunk_mmap[:]
                    pos += chunk_len
                else:
                    # For larger chunks, copy in buffer-sized pieces
                    for i in range(0, chunk_len, buffer_size):
                        end = min(i + buffer_size, chunk_len)
                        output_mmap[pos : pos + (end - i)] = chunk_mmap[i:end]
                        pos += end - i

        finally:
            # Clean up batch mmaps
            for mmap_obj in batch_mmaps:
                del mmap_obj

    output_mmap.flush()
    return output_mmap


def cleanup_chunks(chunks_dir: Path) -> None:
    """Clean up all chunk files and the chunks directory."""
    if not chunks_dir.exists():
        return

    for _ in range(3):  # Try a few times with delays
        try:
            for chunk_file in chunks_dir.iterdir():
                try:
                    chunk_file.unlink()
                except Exception as e:
                    print(f"Warning: Failed to delete chunk file {chunk_file}: {e}")
            chunks_dir.rmdir()
            break
        except Exception as e:
            print(f"Warning: Cleanup attempt failed: {e}")
            time.sleep(0.5)
    else:
        print("Warning: Could not fully clean up chunks directory")


def parallel_slice_mmap(
    numpy_arrays: ArrayDict,
    target_subjects: NDArray,
    unique_subjects: NDArray,
    inverse_indices: NDArray,
    mmap_folder: Union[str, Path],
    num_threads: int = 1,
    chunk_size: int = 10_000_000,
) -> ArrayDict:
    """Process large arrays in chunks."""
    mmap_folder = Path(mmap_folder)
    chunks_dir = mmap_folder / "chunks"
    mmap_folder.mkdir(exist_ok=True)
    chunks_dir.mkdir(exist_ok=True)

    print("Creating masks and validating subjects...")
    target_subjects_sorted = np.sort(target_subjects)
    indices = np.searchsorted(unique_subjects, target_subjects_sorted)
    valid_mask = indices < len(unique_subjects)
    valid_indices = indices[valid_mask]
    valid_targets = target_subjects_sorted[valid_mask]

    final_valid_indices = valid_indices[unique_subjects[valid_indices] == valid_targets]

    if len(final_valid_indices) == 0:
        raise ValueError("None of the target subjects were found in unique_subjects.")

    # Create mask
    print("Creating memory-mapped mask...")
    mask = None
    try:
        mask_path = mmap_folder / "temp_mask.dat"
        mask = np.memmap(mask_path, dtype=bool, mode="w+", shape=inverse_indices.shape)

        total_chunks = (len(inverse_indices) + chunk_size - 1) // chunk_size
        for chunk_start in tqdm(
            range(0, len(inverse_indices), chunk_size), total=total_chunks
        ):
            chunk_end = min(chunk_start + chunk_size, len(inverse_indices))
            chunk = inverse_indices[chunk_start:chunk_end]
            mask[chunk_start:chunk_end] = np.isin(chunk, final_valid_indices)

        # Define arrays to process
        arrays_to_process = [
            (
                numpy_arrays["subject_numeric_id"],
                "subject_numeric_id",
                np.dtype(np.int64),
            ),
            (numpy_arrays["subjectStart"], "subjectStart", np.dtype(np.int32)),
            (numpy_arrays["subjectEnd"], "subjectEnd", np.dtype(np.int32)),
            (numpy_arrays["alnLength"], "alnLength", np.dtype(np.int32)),
            (numpy_arrays["qlen"], "qlen", np.dtype(np.int32)),
            (numpy_arrays["percIdentity"], "percIdentity", np.dtype(np.float32)),
            (numpy_arrays["slen"], "slen", np.dtype(np.int32)),
            (numpy_arrays["bitScore"], "bitScore", np.dtype(np.float32)),
            (numpy_arrays["query_numeric_id"], "query_numeric_id", np.dtype(np.int64)),
            (numpy_arrays["row_hash"], "row_hash", np.dtype(np.int64)),
        ]
        arrays_to_process.sort(key=lambda x: x[1])

        print("Processing arrays in chunks...")
        for chunk_start in tqdm(
            range(0, len(mask), chunk_size), desc="Processing chunks"
        ):
            process_chunk(arrays_to_process, chunk_start, chunk_size, mask, mmap_folder)

        print("Merging chunk results...")
        result_arrays = {}
        for _, name, dtype in tqdm(arrays_to_process, desc="Merging results"):
            chunk_files = sorted(list(chunks_dir.glob(f"{name}_*.dat")))
            if not chunk_files:
                continue

            output_path = mmap_folder / f"{name}.dat"
            result_arrays[name] = combine_chunks(chunk_files, output_path, dtype)

        # Clean up after all merging is complete
        print("Cleaning up temporary files...")
        if mask is not None:
            del mask
        cleanup_chunks(chunks_dir)
        try:
            (mmap_folder / "temp_mask.dat").unlink()
        except Exception as e:
            print(f"Warning: Failed to delete mask file: {e}")

        print("Processing complete!")
        return result_arrays

    except Exception as e:
        # Clean up in case of error
        if mask is not None:
            del mask
        cleanup_chunks(chunks_dir)
        try:
            (mmap_folder / "temp_mask.dat").unlink()
        except Exception:
            pass
        raise e
