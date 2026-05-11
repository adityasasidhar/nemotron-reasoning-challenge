"""Metric for NVIDIA (129716)."""

import subprocess
import sys

# Set up environment
commands = [
    'uv pip uninstall torch torchvision torchaudio',
    'tar -cf - -C /kaggle/usr/lib/notebooks/metric/nvidia_metric_utility_script . | tar -xf - -C /tmp',
    'chmod +x /tmp/triton/backends/nvidia/bin/ptxas',
    'chmod +x /tmp/triton/backends/nvidia/bin/ptxas-blackwell',
]
for cmd in commands:
    print(f'Running: {cmd}')
    subprocess.run(cmd, shell=True, check=True)
sys.path.insert(0, '/tmp')

import glob
import math
import multiprocessing
import os
import re
import time
from pathlib import Path

import kagglehub
import pandas as pd
from tqdm import tqdm

# Configuration
MODEL_PATH = kagglehub.model_download(
    'metric/nemotron-3-nano-30b-a3b-bf16/transformers/default'
)
DATA_PATH = Path(kagglehub.dataset_download('metric/nvidia-nemotron-rerun-data-129716'))


class ParticipantVisibleError(Exception):
    pass


def cache_model(
    path: str | Path,
    exts: tuple[str, ...] = ('.bin', '.pt', '.safetensors'),
    num_workers: int | None = None,
    chunk_mb: int = 256,
) -> int:
    """Pre-read model weight files into the OS page cache to speed up later loads.

    Args:
        path        : Directory containing model files, or a single file path.
        exts        : File extensions treated as model weight files.
        num_workers : Number of threads (default = min(CPU cores, 8)).
        chunk_mb    : Size of each read chunk in MB.

    Returns:
        Total bytes read (int).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def warmup_file(fpath: Path) -> tuple[Path, int]:
        """Sequentially read an entire file in chunks."""
        chunk_size = chunk_mb * 1024 * 1024
        total = 0
        try:
            with open(fpath, 'rb') as f:
                while True:
                    data = f.read(chunk_size)
                    if not data:
                        break
                    total += len(data)
        except Exception as e:
            print(f'Error reading {fpath}: {e}')
        return fpath, total

    path = Path(path)
    # Collect files to read
    files: list[Path] = []
    if path.is_dir():
        files = [p for p in path.rglob('*') if p.is_file() and str(p).endswith(exts)]
        files.sort()
    else:
        files = [path] if path.exists() else []

    if not files:
        print(f'No model files found to cache at: {path}')
        return 0

    # Decide number of worker threads
    if num_workers is None:
        try:
            num_workers = min(multiprocessing.cpu_count(), 8)
        except Exception:
            num_workers = 4

    print(f'[cache_model] {len(files)} file(s), {num_workers} worker(s)')
    t0 = time.time()
    total_bytes = 0
    # Read files in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(warmup_file, f): f for f in files}
        for i, fut in enumerate(as_completed(futures), 1):
            fpath, n = fut.result()
            total_bytes += n
            print(f'[{i}/{len(files)}] cached {fpath.name}')

    elapsed = time.time() - t0
    gb = total_bytes / 1024**3
    speed = gb / elapsed if elapsed > 0 else 0
    print(f'[cache_model] total read ≈ {gb:.2f} GB')
    print(f'[cache_model] elapsed {elapsed:.2f} s, ~{speed:.2f} GB/s')
    return total_bytes


def extract_final_answer(text: str | None) -> str:
    r"""Extracts the final answer from the model response.

    Prioritizes extracting answers inside `\boxed{}`.
    If no `\boxed{}` format is found, attempts to extract numbers from other formats.

    Examples:
        >>> extract_final_answer(r"The answer is \boxed{42}")
        '42'
        >>> extract_final_answer("The final answer is: 3.14")
        '3.14'
        >>> extract_final_answer("Just a number 100 in text")
        '100'
        >>> extract_final_answer(None)
        'NOT_FOUND'
    """
    if text is None:
        return 'NOT_FOUND'

    # Search for boxed answer. For each \boxed{ occurrence, take everything up
    # to the last } before the next \boxed{ (or end of text). This handles
    # answers that themselves contain '}' (the model writes them literally,
    # producing e.g. \boxed{}52} for the answer "}52") as well as nested LaTeX
    # like \boxed{\frac{1}{2}}.
    boxed_starts = list(re.finditer(r'\\boxed\{', text))
    matches = []
    for i, m in enumerate(boxed_starts):
        start = m.end()
        end = boxed_starts[i + 1].start() if i + 1 < len(boxed_starts) else len(text)
        segment = text[start:end]
        last_brace = segment.rfind('}')
        matches.append(segment[:last_brace] if last_brace != -1 else segment)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

    # Other common formats if \boxed{} is not found
    patterns = [
        r'The final answer is:\s*([^\n]+)',
        r'Final answer is:\s*([^\n]+)',
        r'Final answer\s*[:：]\s*([^\n]+)',
        r'final answer\s*[:：]\s*([^\n]+)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    # If no structured format is found, extract the last valid number in the text
    matches = re.findall(r'-?\d+(?:\.\d+)?', text)
    if matches:
        return matches[-1]

    # If no numeric answer is found, return the last line of text as a fallback
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else 'NOT_FOUND'


def verify(stored_answer: str, predicted: str) -> bool:
    """Verify if the answer matches.

    For numerical answers, allow them to be judged as equal within a certain relative tolerance (1e-2);
    otherwise, compare strictly as strings (case-insensitive).

    Examples:
        >>> verify("10011000", "10011000")
        True
        >>> verify("10011000", "10011001")
        False
        >>> verify("24.64", "24.6401")
        True
        >>> verify("XLVII", "xlvii")
        True
        >>> verify("11011", "00011011")
        False
    """
    # Clean up strings
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()

    # If the answer is a binary string, compare strictly as strings
    if re.fullmatch(r'[01]+', stored_answer):
        return predicted.lower() == stored_answer.lower()

    try:
        # Try to convert the answers to floating point numbers
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        # Use a small absolute tolerance for numbers near zero
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        # Fallback to case-insensitive string comparison
        return predicted.lower() == stored_answer.lower()


def generate_standard_submission(submission_dir: str):
    """Processes an extracted submission archive to produce a standard submission file."""
    # Locate the LoRA files within the extracted directory
    possible_extraction_dirs = {
        '/kaggle/tmp',
        '/kaggle/working',
        submission_dir,
    }
    adapter_configs = []
    for search_dir in possible_extraction_dirs:
        if os.path.exists(search_dir):
            adapter_configs.extend(
                glob.glob(
                    os.path.join(search_dir, '**/adapter_config.json'), recursive=True
                )
            )
    if not adapter_configs:
        raise ParticipantVisibleError(
            'No adapter_config.json found in submission. Found:\n\n'
            f'{submission_dir} {os.listdir(submission_dir)}\n\n'
            f'/kaggle/tmp {os.listdir("/kaggle/tmp")}\n\n'
            f'/kaggle/input/competition_evaluation {os.listdir("/kaggle/input/competition_evaluation")}'
        )

    lora_path = os.path.dirname(adapter_configs[0])

    # Load test data
    test_df = pd.read_csv(DATA_PATH / 'test.csv', index_col=None)

    row_id_col = str(test_df.columns.to_list()[0])
    predictions = []
    for item in test_df.itertuples(index=False):
        predictions.append(
            {
                row_id_col: getattr(item, row_id_col),
                'prediction': lora_path,
            }
        )

    submission_df = pd.DataFrame(predictions)

    # Write the standard submission file to the current working directory
    submission_df.to_csv('submission.csv', index=False)


def generate_predictions(
    test_df: pd.DataFrame,
    lora_path: str,
    row_id_col: str,
    max_lora_rank: int,
    max_tokens: int,
    top_p: float,
    temperature: float,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    debug: bool = False,
) -> pd.DataFrame:
    """Load the model and generate predictions for the provided test data.

    Args:
        debug: If True, writes a CSV file with raw model outputs and extracted predictions.
    """
    # Cache Model
    cache_model(MODEL_PATH, num_workers=16, chunk_mb=1024)

    os.environ['TRANSFORMERS_NO_TF'] = '1'
    os.environ['TRANSFORMERS_NO_FLAX'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TRITON_PTXAS_PATH'] = '/tmp/triton/backends/nvidia/bin/ptxas'

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    # Initialize vLLM Offline inference Engine
    llm = LLM(
        model=str(MODEL_PATH),
        tensor_parallel_size=1,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype='auto',
        max_model_len=max_model_len,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=max_lora_rank,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    tokenizer = llm.get_tokenizer()
    prompts = []
    for item in test_df.itertuples(index=False):
        user_content = (
            item.prompt
            + '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'
        )
        # Format using the tokenizer's chat template directly
        try:
            prompt = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': user_content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except Exception:
            # Fallback if chat template fails
            prompt = user_content
        prompts.append(prompt)

    # Generate predictions using continuous batching
    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        lora_request=LoRARequest('adapter', 1, lora_path),
    )

    predictions = []
    debug_records = []
    for item, output in zip(test_df.itertuples(index=False), outputs):
        raw_text = output.outputs[0].text
        extracted_answer = extract_final_answer(raw_text)

        row_id_val = getattr(item, row_id_col)

        predictions.append(
            {
                row_id_col: row_id_val,
                'prediction': extracted_answer,
            }
        )

        if debug:
            debug_records.append(
                {
                    row_id_col: row_id_val,
                    'raw_output': raw_text,
                    'extracted_prediction': extracted_answer,
                }
            )

    # Write debug CSV if requested
    if debug and debug_records:
        debug_df = pd.DataFrame(debug_records)
        debug_df.to_csv('debug_predictions.csv', index=False)
        print('Debug data saved to debug_predictions.csv')

    return pd.DataFrame(predictions)


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    max_lora_rank: int = 32,
    max_tokens: int = 3584,
    top_p: float = 1.0,
    temperature: float = 1.0,
    max_num_seqs: int = 128,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 4096,
    debug: bool = False,
) -> float:
    r"""Evaluate the generated predictions against the ground truth.

    Submissions are evaluated based on their **Accuracy** in solving the provided
    tasks. The NVIDIA Nemotron-3-Nano-30B model is loaded with the participant's
    submitted LoRA adapter (which must include an `adapter_config.json`) using
    the vLLM inference engine. For each test case, the model is prompted to
    generate a response and instructed to place its final answer within a `\boxed{}`
    LaTeX command. The metric extracts the final answer from the generated text,
    prioritizing content within the boxed format while falling back to other
    heuristic patterns or the first numeric value found. A prediction is graded as
    correct if it matches the ground truth either exactly as a string or within a
    relative numerical tolerance of $10^{-2}$. The final score is the proportion of
    correctly answered questions.

    Args:
        solution: DataFrame containing the ground truth answers. Must include the
            row_id_column_name and an 'answer' column.
        submission: DataFrame containing the predicted answers. Must include the
            row_id_column_name and a 'prediction' column.
        row_id_column_name: The name of the ID column used to join solution and
            submission.
        max_lora_rank: Maximum rank for LoRA adapters.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top-p sampling parameter.
        temperature: Temperature sampling parameter.
        max_num_seqs: Maximum number of sequences to process concurrently.
        gpu_memory_utilization: Fraction of GPU memory to allocate for the vLLM execution.
        max_model_len: Maximum context length (input + output tokens).
        debug: If True, writes raw outputs and extracted predictions to a CSV file.

    Returns:
        The accuracy score (fraction of matches) as a float.
    """
    lora_path = submission['prediction'].iloc[0]

    # Load test data and filter it to only include rows present in the solution
    test_df = pd.read_csv(DATA_PATH / 'test.csv', index_col=None)
    row_id_col = str(test_df.columns.to_list()[0])
    test_df = test_df[test_df[row_id_col].isin(solution[row_id_column_name])]

    submission = generate_predictions(
        test_df=test_df,
        lora_path=lora_path,
        row_id_col=row_id_column_name,
        max_lora_rank=max_lora_rank,
        max_tokens=max_tokens,
        top_p=top_p,
        temperature=temperature,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        debug=debug,
    )

    dataset = solution.merge(submission, on=row_id_column_name)
    num_correct = 0

    # Verify the predictions
    for item in dataset.itertuples(index=False):
        ground_truth = item.answer
        extracted_answer = item.prediction

        match = verify(str(ground_truth), str(extracted_answer))
        if match:
            num_correct += 1

    accuracy = num_correct / len(solution)
    return float(accuracy)




# Running: uv pip uninstall torch torchvision torchaudio
# Using Python 3.12.12 environment at: /usr
# Uninstalled 3 packages in 824ms
#  - torch==2.9.0+cu126
#  - torchaudio==2.9.0+cu126
#  - torchvision==0.24.0+cu126
# Running: tar -cf - -C /kaggle/usr/lib/notebooks/metric/nvidia_metric_utility_script . | tar -xf - -C /tmp
# Running: chmod +x /tmp/triton/backends/nvidia/bin/ptxas
# Running: chmod +x /tmp/triton/backends/nvidia/bin/ptxas-blackwell