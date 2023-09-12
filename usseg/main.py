"""Main script for running the file organisation and segmentation"""
#/usr/bin/env python3

# Python imports
import os
import sys
import cProfile
import pstats
import io
from pstats import SortKey

# Module imports
import toml

# Local imports
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
import usseg

def prof(fn, *args, **kwargs):

    pr = cProfile.Profile()

    pr.enable()
    rtn_val = fn(*args, **kwargs)
    pr.disable()

    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()

    with open(f"{fn.__name__}.log", "w+") as f:
        f.write(s.getvalue())

    return rtn_val


def main(root_dir):
    """Main function that performs all of the segmentation on a root directory"""

    # Checks and sets up the tesseract environment
    usseg.setup_tesseract()

    # Gets a list of likely ultrasound images from root dir and saves them to a pickle file.
    filenames = prof(usseg.get_likely_us, root_dir)

    # Segments and digitises the pre-selected ultrasound images.
    prof(usseg.segment, filenames)

    # Generates an output.html of the segmented output
    prof(usseg.generate_html_from_pkl)

if __name__ == "__main__":
    root_dir = toml.load("config.toml")["root_dir"]
    main(root_dir)
