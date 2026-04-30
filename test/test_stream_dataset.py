from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.load_dataset import load_civil


def main() -> None:
    df = load_civil(stream=True, take=10)

    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()