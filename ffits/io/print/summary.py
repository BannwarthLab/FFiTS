"""Prints a formatted summary of a completed (or failed) ffits run."""

import datetime


def print_run_summary(
    start_time: float, end_time: float = None, success: bool = True, message: str = None
):
    """
    Prints a formatted summary of the run completion.

    Args:
        start_time (float): Start time as returned by time.time().
        end_time (float, optional): End time as returned by time.time(). Defaults to now.
        success (bool, optional): Whether the run completed successfully. Defaults to True.
        message (str, optional): Additional message to display in the summary.
    """

    if end_time is None:
        end_time = datetime.datetime.now().timestamp()

    elapsed_seconds = end_time - start_time
    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)
    seconds = int(elapsed_seconds % 60)

    if hours > 0:
        time_str = f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        time_str = f"{minutes}m {seconds}s"
    else:
        time_str = f"{seconds}s"

    status = "Calculation COMPLETED" if success else "Calculation FAILED"

    start_dt = datetime.datetime.fromtimestamp(start_time)
    end_dt = datetime.datetime.fromtimestamp(end_time)
    start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")

    print(" " * 70)
    print("=" * 70)
    print(" Run Summary")
    print("=" * 70)
    print(f" Status:       {status}")
    print(f" Start Time:   {start_str}")
    print(f" End Time:     {end_str}")
    print(f" Total Time:   {time_str}")
    if message:
        print(f" Message:      {message}")
    print("=" * 70 + "\n")
