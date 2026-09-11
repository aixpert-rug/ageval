"""
One-time setup helper for METEOR's NLTK WordNet dependency.

Run as: `aixpert-setup-meteor` (after `pip install`), or
        `python -m metrics.common.meteor_setup`

Exists because the raw `nltk.download('wordnet')` path has two real
failure modes end users will hit without guidance:

1. No network access to nltk's data server (common on restricted HPC
   login nodes, some corporate networks) -- nothing this script can fix,
   but it gives a clear, actionable error instead of nltk's default deep
   traceback.
2. A permissions warning ("will not authorize the non-private download
   directory") on any multi-user Unix system where the download
   directory's ancestor (often $HOME) is group- or world-writable --
   NOT unique to HPC, can happen on shared cloud VMs, university
   clusters, some Docker images. This script detects that warning
   specifically and walks the user to a private alternative location,
   rather than leaving them to debug a UserWarning on their own.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings


def _check_available() -> bool:
    import nltk
    try:
        nltk.data.find("corpora/wordnet")
        nltk.data.find("corpora/omw-1.4")
        return True
    except LookupError:
        return False


def _attempt_download(download_dir: str | None) -> tuple[bool, list[str]]:
    """Returns (success, permission_warnings_seen)."""
    import nltk

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ok_wn = nltk.download("wordnet", download_dir=download_dir, quiet=True)
        ok_omw = nltk.download("omw-1.4", download_dir=download_dir, quiet=True)

    permission_warnings = [
        str(w.message) for w in caught
        if "not authorize" in str(w.message) or "world- or group-writable" in str(w.message)
    ]
    return (bool(ok_wn) and bool(ok_omw)), permission_warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Set up the NLTK WordNet corpus needed by METEOREvaluator.")
    parser.add_argument(
        "--nltk-data-dir",
        default=None,
        help=(
            "Directory to install the corpus into. Defaults to nltk's own "
            "default location. If you hit a permissions warning, re-run "
            "with e.g. --nltk-data-dir ~/.cache/aixpert_nltk_data (a "
            "location only you can write to)."
        ),
    )
    args = parser.parse_args()

    if args.nltk_data_dir:
        args.nltk_data_dir = os.path.expanduser(args.nltk_data_dir)

    if _check_available() and not args.nltk_data_dir:
        print("WordNet corpus already available. Nothing to do.")
        return 0

    print("Downloading WordNet corpus for METEOR...")
    try:
        success, permission_warnings = _attempt_download(args.nltk_data_dir)
    except Exception as e:  # network errors, etc. -- surface clearly rather than a raw traceback
        print(
            f"Download failed: {e}\n\n"
            "This is usually a network restriction (e.g. a login node without "
            "outbound access to nltk's data server). Options:\n"
            "  1. Ask your system administrator about outbound access to nltk's "
            "data server, or\n"
            "  2. Download the 'wordnet' and 'omw-1.4' corpora on a machine "
            "with access (see https://github.com/nltk/nltk_data), transfer "
            "the files over, and set the NLTK_DATA environment variable to "
            "point at them.",
            file=sys.stderr,
        )
        return 1

    if permission_warnings:
        print(
            "\nDownload succeeded, but NLTK flagged a permissions issue:\n"
            f"  {permission_warnings[0]}\n\n"
            "This means your default download location (often inside $HOME) "
            "has an ancestor directory that other users on this system could "
            "write to -- a real (if modest) supply-chain concern on shared "
            "systems, not just a noisy warning.\n\n"
            "Fix: re-run this command pointing at a private directory you own, e.g.:\n"
            "    aixpert-setup-meteor --nltk-data-dir ~/.cache/aixpert_nltk_data\n\n"
            "Then add this to your shell profile so METEOREvaluator finds it "
            "in future sessions:\n"
            "    export NLTK_DATA=~/.cache/aixpert_nltk_data",
            file=sys.stderr,
        )
        return 1

    if _check_available():
        print("WordNet corpus installed and verified.")
        if args.nltk_data_dir:
            print(
                f"\nAdd this to your shell profile so it's found in future sessions:\n"
                f"    export NLTK_DATA={args.nltk_data_dir}"
            )
        return 0

    print(
        "nltk reported the download succeeded, but the corpus still isn't "
        "found afterward. In practice this usually also means a network "
        "restriction (nltk can fail to actually fetch data without raising "
        "a normal error) -- same remediation as a download error:\n"
        "  1. Ask your system administrator about outbound access to nltk's "
        "data server, or\n"
        "  2. Download the 'wordnet' and 'omw-1.4' corpora on a machine "
        "with access (see https://github.com/nltk/nltk_data), transfer "
        "the files over, and set the NLTK_DATA environment variable to "
        "point at them.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
