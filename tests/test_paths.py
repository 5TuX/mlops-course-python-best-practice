import os
import tempfile

import pytest
from src.main import validate_folder_path


def test_validate_folder_path() -> None:
    with pytest.raises(FileNotFoundError):
        validate_folder_path("/a/b/this_dir_does_not_exist")
    with pytest.raises(NotADirectoryError):
        validate_folder_path(42)  # type: ignore
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Temporary directory created at: {temp_dir}")
        # Remove all access rights to the directory
        os.chmod(temp_dir, 0)
        with pytest.raises(PermissionError):
            validate_folder_path(temp_dir)
