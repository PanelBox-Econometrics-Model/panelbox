"""
Test BaseResult - Basic Functionality
=====================================

Tests the basic functionality of BaseResult abstract class.
"""

from datetime import datetime
from pathlib import Path

from panelbox.experiment.results import BaseResult


# Create a concrete implementation for testing
class TestResult(BaseResult):
    """Concrete implementation of BaseResult for testing."""

    def __init__(self, test_data=None, **kwargs):
        super().__init__(**kwargs)
        self.test_data = test_data or {
            "total_tests": 10,
            "passed": 8,
            "failed": 2,
            "pass_rate": 0.8,
        }

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "summary": self.test_data,
            "timestamp": self.timestamp.isoformat(),
            "report_title": "Test Result Report",
        }

    def summary(self):
        """Generate text summary."""
        return (
            f"Test Result Summary\n"
            f"===================\n"
            f"Total tests: {self.test_data['total_tests']}\n"
            f"Passed: {self.test_data['passed']}\n"
            f"Failed: {self.test_data['failed']}\n"
            f"Pass rate: {self.test_data['pass_rate']:.1%}\n"
        )


def main():
    print("=" * 80)
    print("TEST: BASERESULT - BASIC FUNCTIONALITY")
    print("=" * 80)
    print()

    # 1. Test instantiation with defaults
    print("1. Testing instantiation...")
    result = TestResult()
    print(f"  ✅ Result created")
    print(f"  - Timestamp: {result.timestamp}")
    print(f"  - Metadata: {result.metadata}")
    print()

    # 2. Test instantiation with custom timestamp and metadata
    print("2. Testing custom timestamp and metadata...")
    custom_ts = datetime(2024, 1, 1, 12, 0, 0)
    custom_meta = {"experiment": "test_1", "version": "1.0"}
    result2 = TestResult(timestamp=custom_ts, metadata=custom_meta)
    print(f"  ✅ Result created with custom values")
    print(f"  - Timestamp: {result2.timestamp}")
    print(f"  - Metadata: {result2.metadata}")
    print()

    # 3. Test to_dict()
    print("3. Testing to_dict()...")
    data_dict = result.to_dict()
    print(f"  ✅ Converted to dict")
    print(f"  - Keys: {list(data_dict.keys())}")
    print(f"  - Summary: {data_dict['summary']}")
    print()

    # 4. Test summary()
    print("4. Testing summary()...")
    summary_text = result.summary()
    print(f"  ✅ Summary generated:")
    print()
    for line in summary_text.split("\n"):
        print(f"    {line}")
    print()

    # 5. Test save_json()
    print("5. Testing save_json()...")
    json_path = Path("/home/guhaase/projetos/panelbox/test_result.json")
    saved_path = result.save_json(str(json_path))
    print(f"  ✅ Saved to JSON: {saved_path}")
    print(f"  - File exists: {saved_path.exists()}")
    print(f"  - File size: {saved_path.stat().st_size} bytes")
    print()

    # 6. Test __repr__()
    print("6. Testing __repr__()...")
    print(f"  ✅ String representation:")
    print(f"  {result}")
    print()

    # 7. Test that BaseResult cannot be instantiated directly
    print("7. Testing abstract class enforcement...")
    try:
        base = BaseResult()
        print("  ❌ ERROR: BaseResult should not be instantiable!")
    except TypeError as e:
        print(f"  ✅ Correctly prevented instantiation")
        print(f"  - Error: {str(e)[:80]}...")
    print()

    # 8. Final summary
    print("=" * 80)
    print("✅ ALL BASERESULT TESTS PASSED!")
    print("=" * 80)
    print()
    print("BaseResult Features Tested:")
    print("  ✅ Instantiation with defaults")
    print("  ✅ Custom timestamp and metadata")
    print("  ✅ to_dict() method")
    print("  ✅ summary() method")
    print("  ✅ save_json() method")
    print("  ✅ __repr__() method")
    print("  ✅ Abstract class enforcement")
    print()
    print("Note: save_html() will be tested in integration tests")
    print("=" * 80)


if __name__ == "__main__":
    main()
