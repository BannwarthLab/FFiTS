import datetime
import textwrap

def print_program_header(version: str = "0.1.0-alpha"):
    """
    Prints a formatted header for the FFiTS program.

    Parameters
    ----------
    version : str
        Program version string.
    """

    # --- Get current date and time ---
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # --- ASCII Art Logo (you can customize this) ---
    logo = r"""
                ███████╗███████╗██╗████████╗███████╗                
                ██╔════╝██╔════╝╚═╝╚══██╔══╝██╔════╝                
                █████╗  █████╗  ██╗   ██║   ███████╗                
                ██╔══╝  ██╔══╝  ██║   ██║   ╚════██║                
                ██║     ██║     ██║   ██║   ███████║                
                ╚═╝     ╚═╝     ╚═╝   ╚═╝   ╚══════╝                
             Force Field-interpolated Transition States             
"""

    # --- Print header ---
    # print("\n" + "=" * 70)
    print(logo)
    print("=" * 70)
    print(f" Program:     FFiTS  —  Force Field-interpolated Transition States")
    print(f" Version:     {version}")
    print(f" Date:        {now}")
    print(f" Author:      Daria Babushkina")
    print(f" License:     tbd")
    print("-" * 70)

    # --- Reference / Citation Info TODO ---
    citation = textwrap.dedent("""
        If you use FFiTS in your work, please cite:
        [1] A future paper 
    """).strip()

    print(citation)
    print("=" * 70 + "\n")
