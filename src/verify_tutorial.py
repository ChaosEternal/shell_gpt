
import os
import sys
import pathlib
import shutil

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sadk.tutorial import read_tutorial, tutorial_dir

def test_read_tutorial():
    # 1. Setup temporary tutorial file
    tutorial_dir.mkdir(parents=True, exist_ok=True)
    test_file = tutorial_dir / "test_tutorial.md"
    content = "# Test Tutorial\nThis is a test."
    with open(test_file, "w") as f:
        f.write(content)
    
    print("Testing valid tutorial name...")
    res = read_tutorial("test_tutorial")
    assert res == content
    print("Pass.")

    print("Testing invalid tutorial name (specialchars)...")
    res = read_tutorial("test..tutorial")
    assert "Invalid tutorial name" in res
    print("Pass.")

    print("Testing non-existent tutorial...")
    res = read_tutorial("non_existent_123")
    assert "not found" in res
    print("Pass.")

    # Cleanup
    test_file.unlink()
    print("All tests passed.")

if __name__ == "__main__":
    try:
        test_read_tutorial()
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)
