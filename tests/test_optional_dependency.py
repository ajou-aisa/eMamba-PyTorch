import subprocess
import sys
import unittest


class OptionalDependencyTests(unittest.TestCase):
    def test_fp32_model_import_does_not_require_brevitas(self) -> None:
        command = (
            "import sys; sys.modules['brevitas'] = None; "
            "from models.emamba import EMamba; print(EMamba.__name__)"
        )
        result = subprocess.run(
            [sys.executable, "-c", command], capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "EMamba")


if __name__ == "__main__":
    unittest.main()
