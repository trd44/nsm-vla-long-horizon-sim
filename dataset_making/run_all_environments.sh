#!/bin/bash
# Script to collect demonstrations for all environments
# Usage: bash dataset_making/run_all_environments.sh

# Configuration
EPISODES=50
SEED=42

# Navigate to project root
cd "$(dirname "$0")/.." || exit 1

# Define environments to run
ENVIRONMENTS=(
    "CubeSorting"
    "HeightStacking"
    "AssemblyLineSorting"
    "PatternReplication"
    "NutAssembly"
    "Hanoi"
    "Hanoi4x3"
    "KitchenEnv"
)

echo "Starting data collection for all environments..."
echo "Episodes per environment: $EPISODES"
echo "Random seed: $SEED"
echo "=========================================="

# Loop through each environment
for ENV in "${ENVIRONMENTS[@]}"; do
    echo ""
    echo "Processing environment: $ENV"
    echo "----------------------------------------"
    
    # Generate a timestamp-based name
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    NAME="${ENV}_${EPISODES}ep_${TIMESTAMP}"
    
    # Run the data collection
    python -m dataset_making.main \
        --env "$ENV" \
        --episodes "$EPISODES" \
        --seed "$SEED" \
        --name "$NAME" \
        --verbose
    
    # Check if the command succeeded
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed $ENV"
    else
        echo "✗ Failed to complete $ENV"
        echo "Continuing to next environment..."
    fi
    
    echo "=========================================="
done

echo ""
echo "Data collection complete!"
echo "Check the output directory for results."

