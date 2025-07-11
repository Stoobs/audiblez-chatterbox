#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command-line interface for Audiblez."""

import argparse
import sys
from typing import Dict, Any

import torch

from audiblez.database import load_all_user_settings
from audiblez.core import main

def cli_main() -> None:
    """Main entry point for the command-line interface."""
    
    epilog = (
        'example:\n'
        '  audiblez book.epub --pick\n\n'
        'to run GUI just run:\n'
        '  audiblez-ui\n\n'
        'Note: Chatterbox-TTS uses audio prompt files for voice cloning.\n'
        'Voice selection is handled through the GUI or audio prompt files.'
    )

    # Load settings from database
    db_settings: Dict[str, Any] = load_all_user_settings() or {}

    parser = argparse.ArgumentParser(epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('epub_file_path', help='Path to the epub file')
    parser.add_argument('-p', '--pick', default=False, help='Interactively select which chapters to read in the audiobook', action='store_true')
    parser.add_argument('-c', '--cuda', default=False, help='Use GPU via Cuda in Torch if available', action='store_true')
    parser.add_argument('-o', '--output', default='.', help='Output folder for the audiobook and temporary files', metavar='FOLDER')
    parser.add_argument('--voice-sample', help='Path to audio file for voice cloning (optional)', metavar='FILE')

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    # CUDA/Engine Handling Logic
    use_cuda_from_cli = args.cuda
    engine_from_db = db_settings.get('engine')

    if use_cuda_from_cli:
        if torch.cuda.is_available():
            print('CUDA GPU available (specified by user via --cuda). Using CUDA.')
            torch.set_default_device('cuda')
        else:
            print('CUDA GPU not available (specified by user via --cuda, but unavailable). Defaulting to CPU.')
            torch.set_default_device('cpu')
    elif engine_from_db == 'cuda':
        if torch.cuda.is_available():
            print('CUDA GPU available (from database settings). Using CUDA.')
            torch.set_default_device('cuda')
        else:
            print('CUDA GPU not available (from database settings, but unavailable). Defaulting to CPU.')
            torch.set_default_device('cpu')
    else:
        # Default to CPU if --cuda not used and DB setting is not 'cuda' or not present
        print('Defaulting to CPU (no CUDA specified by user and not set to CUDA in DB).')
        torch.set_default_device('cpu')

    # Call main() with correct parameter names matching the function signature
    main(
        file_path=args.epub_file_path,
        pick_manually=args.pick,
        output_folder=args.output,
        voice_clone_sample=args.voice_sample
    )

if __name__ == '__main__':
    cli_main()
