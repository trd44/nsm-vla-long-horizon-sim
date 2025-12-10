#!/bin/bash
# Script to collect demonstrations for selected environments
# Usage: bash dataset_making/run_selected_environments.sh

# Navigate to project root
cd "$(dirname "$0")/.." || exit 1

# ============================================
# CONFIGURATION - Customize these as needed
# ============================================

EPISODES=50
# SEED=42
# NOISE_STD=0.03
# NOISY_FRACTION=0.30

# Comment out environments you don't want to run
ENVIRONMENTS=(
    "CubeSorting"
    "HeightStacking"
    "AssemblyLineSorting"
    "PatternReplication"
    "Hanoi"
    "Hanoi4x3"
    # "KitchenEnv"
    # "NutAssembly"
)

# ============================================

echo "Starting data collection..."
echo "Episodes per environment: $EPISODES"
# echo "Random seed: $SEED"
# echo "Noise std: $NOISE_STD"
# echo "Noisy fraction: $NOISY_FRACTION"
echo "=========================================="

# Loop through each environment
for ENV in "${ENVIRONMENTS[@]}"; do
    echo ""
    echo "Processing: $ENV"
    echo "----------------------------------------"
    
    # Generate a name
    NAME="${ENV}_${EPISODES}ep_box_movement"
    
    # Run the data collection with all parameters
    python -m dataset_making.main \
        --env "$ENV" \
        --episodes "$EPISODES" \
        --name "$NAME"
        # --seed "$SEED" \
        # --noise-std "$NOISE_STD" \
        # --noisy-fraction "$NOISY_FRACTION" \
        # --verbose \
        # --save-full-res-vid
    
    # Check result
    if [ $? -eq 0 ]; then
        echo "✓ Completed $ENV"
    else
        echo "✗ Failed $ENV"
    fi
    
    echo "=========================================="
done

echo ""
echo "All done!"

