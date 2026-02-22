"""
Dataset Manager — Unified Kaggle dataset download & extraction utilities
Eliminates duplicate download logic across training modules
"""
import os
import zipfile
import shutil
from pathlib import Path
from typing import Optional, Union


class DatasetManager:
    """Handles Kaggle dataset downloads with consistent error handling."""
    
    @staticmethod
    def download_kaggle_dataset(
        dataset_id: str,
        dest_dir: Union[str, Path],
        force: bool = False
    ) -> Path:
        """
        Download a Kaggle dataset using the API.
        
        Args:
            dataset_id: Kaggle dataset identifier (e.g., 'username/dataset-name')
            dest_dir: Destination directory for extracted data
            force: If True, re-download even if exists
            
        Returns:
            Path to the downloaded/extracted dataset directory
            
        Raises:
            RuntimeError: If download fails
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already exists
        if not force and dest_dir.exists() and any(dest_dir.iterdir()):
            print(f'  ✅ Dataset already exists: {dest_dir}')
            return dest_dir
        
        print(f'  📥 Downloading Kaggle dataset: {dataset_id}...')
        try:
            import kaggle
            kaggle.api.dataset_download_files(
                dataset_id,
                path=str(dest_dir),
                unzip=True,
                quiet=False
            )
            print(f'  ✅ Dataset downloaded to {dest_dir}')
            return dest_dir
        except Exception as e:
            raise RuntimeError(
                f"Failed to download Kaggle dataset '{dataset_id}': {e}\n"
                f"Make sure:\n"
                f"  1. You have accepted the dataset terms at "
                f"https://www.kaggle.com/datasets/{dataset_id}\n"
                f"  2. Your Kaggle credentials are configured correctly"
            )
    
    @staticmethod
    def download_kaggle_competition(
        competition_id: str,
        dest_dir: Union[str, Path],
        force: bool = False
    ) -> Path:
        """
        Download a Kaggle competition dataset.
        
        Args:
            competition_id: Competition identifier (e.g., 'paddy-disease-classification')
            dest_dir: Destination directory for extracted data
            force: If True, re-download even if exists
            
        Returns:
            Path to the downloaded/extracted dataset directory
            
        Raises:
            RuntimeError: If download fails or competition rules not accepted
        """
        dest_dir = Path(dest_dir)
        zip_file = Path(f'{competition_id}.zip')
        
        # Check if already extracted
        if not force and dest_dir.exists() and any(dest_dir.iterdir()):
            print(f'  ✅ Competition dataset already exists: {dest_dir}')
            return dest_dir
        
        # Check if zip exists and extract
        if zip_file.exists() and not force:
            print(f'  📦 Extracting existing {zip_file}...')
            DatasetManager._extract_zip(zip_file, dest_dir)
            return dest_dir
        
        # Download from Kaggle
        print(f'  📥 Downloading competition: {competition_id}...')
        try:
            ret = os.system(f'kaggle competitions download -c {competition_id} -q')
            if ret != 0:
                raise RuntimeError(f"Kaggle CLI returned exit code {ret}")
            
            # Extract the downloaded zip
            if zip_file.exists():
                DatasetManager._extract_zip(zip_file, dest_dir)
                # Optionally remove zip to save space
                # zip_file.unlink()
                return dest_dir
            else:
                raise RuntimeError(f"Download succeeded but zip file not found: {zip_file}")
                
        except Exception as e:
            raise RuntimeError(
                f"Failed to download competition '{competition_id}': {e}\n"
                f"Make sure:\n"
                f"  1. You have accepted the competition rules at "
                f"https://www.kaggle.com/competitions/{competition_id}/rules\n"
                f"  2. Your Kaggle credentials are configured correctly\n"
                f"  3. The competition allows API downloads"
            )
    
    @staticmethod
    def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
        """Extract a zip file to destination directory."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(str(dest_dir))
            print(f'  ✅ Extracted to {dest_dir}')
        except Exception as e:
            raise RuntimeError(f"Failed to extract {zip_path}: {e}")
    
    @staticmethod
    def ensure_csv_exists(
        csv_path: Union[str, Path],
        dataset_dir: Path,
        expected_name: Optional[str] = None
    ) -> Path:
        """
        Ensure a CSV file exists, handling common download variations.
        
        Some Kaggle datasets have unpredictable CSV filenames.
        This finds and renames them if needed.
        
        Args:
            csv_path: Expected CSV path
            dataset_dir: Directory to search for CSV
            expected_name: If provided, rename found CSV to this name
            
        Returns:
            Path to the CSV file
            
        Raises:
            FileNotFoundError: If no CSV found
        """
        csv_path = Path(csv_path)
        
        if csv_path.exists():
            return csv_path
        
        # Search for any CSV in the directory
        csv_files = list(Path(dataset_dir).glob('*.csv'))
        
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {dataset_dir}. "
                f"Dataset may not have downloaded correctly."
            )
        
        # If expected name provided and CSV found with different name, rename it
        if expected_name and csv_files[0] != csv_path:
            print(f'  🔄 Renaming {csv_files[0].name} → {expected_name}')
            shutil.move(str(csv_files[0]), str(csv_path))
            return csv_path
        
        return csv_files[0]


# Convenience functions for backward compatibility
def download_dataset(dataset_id: str, dest_dir: Union[str, Path]) -> Path:
    """Shorthand for DatasetManager.download_kaggle_dataset()"""
    return DatasetManager.download_kaggle_dataset(dataset_id, dest_dir)


def download_competition(competition_id: str, dest_dir: Union[str, Path]) -> Path:
    """Shorthand for DatasetManager.download_kaggle_competition()"""
    return DatasetManager.download_kaggle_competition(competition_id, dest_dir)
