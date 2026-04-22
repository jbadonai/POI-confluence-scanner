import sys
import re


def extract_symbols(raw_text):
    return re.findall(r'^\s*([A-Z0-9]+USDT)\b', raw_text, re.MULTILINE)


def main():
    print("Paste your table below, then press Ctrl+D (Linux/macOS) or Ctrl+Z + Enter (Windows):\n")

    raw_text = sys.stdin.read()

    symbols = extract_symbols(raw_text)

    print("\n--- Results ---")
    print(symbols)  # Python list
    print("\nSingle line:")
    print(" ".join(symbols))  # space-separated
    print("\nCSV:")
    print(",".join(symbols))  # comma-separated


if __name__ == "__main__":
    main()