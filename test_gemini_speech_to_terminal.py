import subprocess
import tempfile
import unittest
from pathlib import Path

from gemini_speech_to_terminal import collect_files, run


class CollectFilesTest(unittest.TestCase):
    def test_git_listed_skipped_directories_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "core.excludesFile", "/dev/null"], cwd=root, check=True)
            paths = [
                "README.md",
                "notes/untracked.md",
                ".playwright-mcp/session/context.json",
                "node_modules/package/index.js",
                "work/.pytest_cache/v/cache/nodeids",
            ]
            for rel in paths:
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(rel)
            subprocess.run(["git", "add", "README.md", "node_modules/package/index.js"], cwd=root, check=True)

            listed = run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root)

            self.assertIsNotNone(listed)
            self.assertEqual(set(paths), set(listed.split("\0")) - {""})
            self.assertEqual(["README.md", "notes/untracked.md"], collect_files(root, 100))


if __name__ == "__main__":
    unittest.main()
