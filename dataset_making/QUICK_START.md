# Quick Start - New Environments

## What Was Added

Your `dataset_making` package now supports 4 additional environments from `auto_demonstration.py`:

1. **CubeSorting** - Sort cubes by size to different platforms
2. **HeightStacking** - Stack cubes from smallest to largest
3. **AssemblyLineSorting** - Sort cubes by type matching
4. **PatternReplication** - Replicate a reference pattern

## Files Modified

1. `dataset_making/record_demos.py` - Added environment configs and goal generation logic
2. `dataset_making/main.py` - Added detector imports and CLI support

## Data Format Preserved

✅ Your existing RLDS data format is **unchanged**  
✅ Recording architecture remains the same  
✅ Noise scheduling preserved  

## Usage

```bash
# Activate your conda environment
conda activate dataset_making

# Record demos for any of the new environments
cd /home/train/vlas/CyclicLxM

python -m dataset_making.main --env HeightStacking --episodes 10
python -m dataset_making.main --env CubeSorting --episodes 10
python -m dataset_making.main --env AssemblyLineSorting --episodes 10
python -m dataset_making.main --env PatternReplication --episodes 10
```

## Prerequisites

Before running, ensure you have:
1. PDDL files in `planning/PDDL/{env_name}/` for each environment
2. Detectors implemented in robosuite for each environment
3. Environments registered in robosuite

## Next Steps

1. Check if PDDL files exist for the new environments
2. Test with a single episode: `python -m dataset_making.main --env CubeSorting --episodes 1 --verbose`
3. If successful, scale up to more episodes

See `MERGE_SUMMARY.md` for detailed technical documentation.

