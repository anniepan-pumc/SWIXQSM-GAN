#!/bin/bash
# T1 Segmentation Script using FreeSurfer recon-all
# This script finds T1 files in MPRAGE folders and runs recon-all for segmentation

# Default base directory (can be overridden by command line argument)
BASE_DIR="${1:-/Volumes/MOVESPEED/QSM_multicohort_data/csvd-bids}"

export FREESURFER_HOME="/Applications/freesurfer/8.1.0"
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export FSLDIR="/Users/panxingqi/fsl"
export FS_LICENSE="/.nofollow/Applications/freesurfer/8.1.0/license.txt"
source $FSLDIR/etc/fslconf/fsl.sh

# Check if base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Base directory does not exist: $BASE_DIR"
    exit 1
fi

# Check if FreeSurfer is available
if ! command -v recon-all &> /dev/null; then
    echo "Error: FreeSurfer recon-all command not found. Please ensure FreeSurfer is installed and sourced."
    exit 1
fi

echo "=========================================="
echo "T1 Segmentation with FreeSurfer recon-all"
echo "=========================================="
echo "Base directory: $BASE_DIR"
echo ""

# Counter for processed subjects
processed=0
skipped=0

# Find all subject directories (sub-*)
for subj_dir in "$BASE_DIR"/sub-*; do
    # Skip if not a directory
    [ ! -d "$subj_dir" ] && continue
    
    subj_name=$(basename "$subj_dir")
    mprage_dir="$subj_dir/MPRAGE"
    
    # Check if MPRAGE directory exists
    if [ ! -d "$mprage_dir" ]; then
        echo "[$subj_name] Skipping: No MPRAGE directory found"
        ((skipped++))
        continue
    fi
    
    # Find T1 file with pattern *_3DT1.nii.gz
    t1_file=$(find "$mprage_dir" -maxdepth 1 -name "*_3DT1.nii.gz" -type f 2>/dev/null | head -n 1)
    
    if [ -z "$t1_file" ]; then
        echo "[$subj_name] Skipping: No *_3DT1.nii.gz file found in MPRAGE"
        ((skipped++))
        continue
    fi
    
    # Check if recon-all output already exists
    reconall_dir="$mprage_dir/reconall"
    if [ -d "$reconall_dir" ] && [ -f "$reconall_dir/scripts/recon-all.done" ]; then
        echo "[$subj_name] Skipping: recon-all already completed"
        ((skipped++))
        continue
    fi
    
    echo "[$subj_name] Processing: $t1_file"
    echo "  Output directory: $mprage_dir"

    export SUBJECTS_DIR="$mprage_dir"
    # Run recon-all
    # -i: input T1 image
    # -subjid: subject ID for output folder
    # -sd: subjects directory (output location)
    # -all: run all recon-all steps
    recon-all -i "$t1_file" -subjid reconall -sd "$mprage_dir" -all
    
    if [ $? -eq 0 ]; then
        echo "[$subj_name] Completed successfully"
        ((processed++))
    else
        echo "[$subj_name] Error: recon-all failed"
    fi
    
    echo ""
done

echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Processed: $processed subjects"
echo "Skipped: $skipped subjects"
echo "Done!"
