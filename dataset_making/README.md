# Dataset Making Instructions

## 1. Generate data

From CyclicLxM Directory:

Example usage
```bash
# Single environment
python -m dataset_making.main --env AssemblyLineSorting --episodes 50 --name box_movement_50ep

# Run all environments at once
bash dataset_making/run_all_environments.sh

# Run selected environments (edit the script to choose which ones)
bash dataset_making/run_selected_environments.sh
```

## 2. Convert to RLDS
Move the .npy files generated in the previous step to a directory in rlds_dataset_builder following the same pattern as other examples in the directory.

1. Name the directory the name of the dataset.
2. Move the npy files into the data/train data/test and data/val directories. 
3. Rename the X_dataset_builder.py file to match the parent directory name. 
4. Rename the class in X_dataset_builder.py to match the parent directory name in PascalCase reather than snake_case.
5. Run the build instruction below. (Only create the conda env if you havent before)

```bash
cd rlds_dataset_builder
conda env create -f environment_ubuntu.yml
conda activate rlds_env
cd your_dataset
tfds build --overwrite
```
or if you have more than one to build use:
```bash
bash rlds_dataset_builder/build_all_dataset.sh
```

## 3. Convert to LeRobot
