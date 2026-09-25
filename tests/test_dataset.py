"""CSV loading and validation."""

import tempfile
import unittest
from pathlib import Path

import support
from src.dataset import DatasetError, load_dataset

_ = support

SUBJECT_CSV = Path(__file__).resolve().parents[1] / "data" / "data.csv"


def _write_csv(contents: str) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
    handle.write(contents)
    handle.close()
    return Path(handle.name)


class LoadDatasetTests(unittest.TestCase):
    def tearDown(self) -> None:
        for path in getattr(self, "_temp_files", []):
            path.unlink(missing_ok=True)

    def _csv(self, contents: str) -> Path:
        path = _write_csv(contents)
        self._temp_files = getattr(self, "_temp_files", [])
        self._temp_files.append(path)
        return path

    def test_loads_subject_dataset(self) -> None:
        mileages, prices = load_dataset(SUBJECT_CSV)
        self.assertEqual(len(mileages), 24)
        self.assertEqual(len(prices), 24)
        self.assertEqual(mileages[0], 240000.0)
        self.assertEqual(prices[0], 3650.0)
        self.assertEqual(min(mileages), 22899.0)
        self.assertEqual(max(mileages), 240000.0)

    def test_missing_file(self) -> None:
        with self.assertRaises(DatasetError):
            load_dataset("/tmp/does-not-exist-ft-linear-regression.csv")

    def test_wrong_header(self) -> None:
        path = self._csv("mileage,cost\n1,2\n3,4\n")
        with self.assertRaises(DatasetError):
            load_dataset(path)

    def test_empty_file(self) -> None:
        path = self._csv("")
        with self.assertRaises(DatasetError):
            load_dataset(path)

    def test_header_only(self) -> None:
        path = self._csv("km,price\n")
        with self.assertRaises(DatasetError):
            load_dataset(path)

    def test_one_valid_row_is_not_enough(self) -> None:
        path = self._csv("km,price\n100,200\n")
        with self.assertRaises(DatasetError):
            load_dataset(path)

    def test_non_numeric_value(self) -> None:
        path = self._csv("km,price\n100,200\nabc,300\n")
        with self.assertRaises(DatasetError):
            load_dataset(path)

    def test_non_finite_value(self) -> None:
        path = self._csv("km,price\n100,200\ninf,300\n")
        with self.assertRaises(DatasetError):
            load_dataset(path)

    def test_negative_mileage(self) -> None:
        path = self._csv("km,price\n-1,200\n100,300\n")
        with self.assertRaises(DatasetError):
            load_dataset(path)

    def test_constant_mileage(self) -> None:
        path = self._csv("km,price\n100,200\n100,400\n")
        with self.assertRaises(DatasetError):
            load_dataset(path)

    def test_skips_blank_lines(self) -> None:
        path = self._csv("km,price\n100,200\n\n300,400\n")
        mileages, prices = load_dataset(path)
        self.assertEqual(mileages, [100.0, 300.0])
        self.assertEqual(prices, [200.0, 400.0])


if __name__ == "__main__":
    unittest.main()
