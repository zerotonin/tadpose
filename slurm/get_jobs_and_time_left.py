import subprocess
import time
import os
from datetime import timedelta

def clear_terminal():
    """Clears the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_slurm_queue_length(user_code):
    """Get the number of jobs in the Slurm queue for a specific user."""
    result = subprocess.run(
        ['squeue', '-u', user_code, '-h', '-r'],
        stdout=subprocess.PIPE
    )
    queue_jobs = result.stdout.decode('utf-8').strip().split('\n')
    return len(queue_jobs) if queue_jobs[0] else 0

def format_time(seconds):
    """Format time in seconds to hh:mm:ss."""
    return str(timedelta(seconds=int(seconds)))

def monitor_queue(user_code):
    previous_count = get_slurm_queue_length(user_code)
    start_time = time.time()
    total_time = 0
    job_decrements = 0

    while True:
        current_count = get_slurm_queue_length(user_code)

        if current_count < previous_count:
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            job_decrements += 1
            average_time_per_job = total_time / job_decrements
            remaining_jobs = current_count
            estimated_time_left = average_time_per_job * remaining_jobs

            clear_terminal()
            print(f"Jobs in queue: {current_count}")
            print(f"Average time per job: {format_time(average_time_per_job)}")
            print(f"Estimated time remaining: {format_time(estimated_time_left)}")
            print("-------------------------")

            previous_count = current_count
            start_time = time.time()  # Reset the start time for the next job

        time.sleep(1)  # Check every second

if __name__ == "__main__":
    # Default to the current user; override with $SLURM_USER if set.
    user_code = os.environ.get("SLURM_USER") or os.environ.get("USER", "")
    monitor_queue(user_code)
