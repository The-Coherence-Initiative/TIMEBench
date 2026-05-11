#!/usr/bin/env bash
# Process every raw *_output.json in Outputs/ — judge + structural stats in one pass.
# Requires OPENAI_API_KEY to be set in the environment.

set -euo pipefail
shopt -s nullglob

OUTPUT_DIR="Outputs"

mapfile -t json_files < <(
    find "$OUTPUT_DIR" -maxdepth 1 -type f -name "*_output.json" | sort
)

if [ ${#json_files[@]} -eq 0 ]; then
    echo "No *_output.json files found in $OUTPUT_DIR"
    exit 0
fi

echo "Found ${#json_files[@]} file(s). Processing..."

for json_file in "${json_files[@]}"; do
    echo
    echo "=================================================="
    echo "  Processing: $json_file"
    echo "=================================================="
    python process_output.py -i "$json_file"
done

echo
echo "All files processed. Ready for compare_results.py."
