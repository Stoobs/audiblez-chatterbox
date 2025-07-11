import os
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile


class CliTest(unittest.TestCase):
    def cli(self, args):
        cmd = f'cd .. && python -m audiblez.cli {args}'
        return os.popen(cmd).read()

    def test_help(self):
        out = self.cli('--help')
        self.assertIn('af_sky', out)
        self.assertIn('usage:', out)

    def test_epub(self):
        out = self.cli('epub/mini.epub')
        self.assertIn('Found cover image', out)
        self.assertIn('Creating M4B file', out)
        self.assertTrue(Path('../mini.m4b').exists())
        self.assertTrue(Path('../mini.m4b').stat().st_size > 256 * 1024)

    def test_epub_voice_and_output_folder(self):
        out = self.cli('epub/mini.epub -v af_sky -o test/prova')
        self.assertIn('Found cover image', out)
        self.assertIn('Creating M4B file', out)
        self.assertTrue(Path('./prova/mini.m4b').exists())
        self.assertTrue(Path('./prova/mini.m4b').stat().st_size > 256 * 1024)

    # TODO: Implement markdown and text file support
    # These tests are skipped until the features are implemented
