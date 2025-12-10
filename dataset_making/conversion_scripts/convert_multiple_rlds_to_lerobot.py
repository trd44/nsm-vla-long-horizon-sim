import tensorflow_datasets as tfds
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import HfApi
from requests import HTTPError


HF_USERNAME = "tduggan93"  # Change to your HuggingFace username
ROBOT_TYPE = "kinova3"
TFDS_PATH = "/home/hrilab/tensorflow_datasets/"
CODEBASE_VERSION = "1.0.0"
PUSH_TO_HUB = True
BRANCH = None
DATASETS = [
    # ("assembly_line_sorting", TFDS_PATH + "assembly_line_sorting/1.0.0"),
    # ("cube_sorting",          TFDS_PATH + "cube_sorting/1.0.0"),
    # ("height_stacking",       TFDS_PATH + "height_stacking/1.0.0"),
    # ("pattern_replication",   TFDS_PATH + "pattern_replication/1.0.0"),
    # ("hanoi_50",              TFDS_PATH + "hanoi50/1.0.0"),
    ("hanoi4x3_50",           TFDS_PATH + "hanoi4x350/1.0.0"),
]

def convert_individual_datasets():
    """Create a separate LeRobot dataset for each task type."""
    
    for task_name, dataset_path in DATASETS:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {task_name}")
        print(f"{'='*60}")
        
        # Create a separate dataset for this task
        repo_id = f"{HF_USERNAME}/{task_name}"
        print(f"Creating LeRobot dataset: {repo_id}")
        
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            robot_type=ROBOT_TYPE,
            fps=10,
            features={
                "image": {
                    "dtype": "image", 
                    "shape": (256, 256, 3), 
                    "names": ["height", "width", "channel"]
                },
                "wrist_image": {
                    "dtype": "image", 
                    "shape": (256, 256, 3), 
                    "names": ["height", "width", "channel"]
                },
                "state": {
                    "dtype": "float32", 
                    "shape": (8,), 
                    "names": ["state"]
                },
                "actions": {
                    "dtype": "float32", 
                    "shape": (7,), 
                    "names": ["actions"]
                },
                # "task": {
                #     "dtype": "string",
                #     "shape": (),
                #     "names": ["task"]
                # },
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )
        
        # Load RLDS dataset
        builder = tfds.builder_from_directory(dataset_path)
        ds = builder.as_dataset(split="train", as_supervised=False)
        
        # Convert episodes
        for epi_idx, ex in enumerate(tfds.as_numpy(ds)):
            # Add all frames from this episode
            for step in ex["steps"]:            
                task_str = step["language_instruction"].decode() if isinstance(step["language_instruction"], bytes) else str(step["language_instruction"])
                dataset.add_frame({
                    "image": step["observation"]["image"],
                    "wrist_image": step["observation"]["wrist_image"],
                    "state": step["observation"]["state"],
                    "actions": step["action"],
                }, task=task_str)  # Use fine-grained instruction as task
            
            dataset.save_episode()
            
            if (epi_idx + 1) % 10 == 0:
                print(f"  Converted {epi_idx + 1} episodes")
        
        print(f"✓ Finished converting {task_name}")
        # print(f"  Pushing {repo_id} to HuggingFace Hub using upload_large_folder...")

        # if PUSH_TO_HUB:
        #     hub_api = HfApi()
        #     try:
        #         hub_api.delete_tag(repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
        #     except HTTPError as e:
        #         print(f"tag={CODEBASE_VERSION} probably doesn't exist. Skipping exception ({e})")
        #         pass
        #     hub_api.delete_files(
        #         delete_patterns=["data/chunk*/episode_*", "meta/*.jsonl", "videos/chunk*"],
        #         repo_id=repo_id,
        #         revision=BRANCH,
        #         repo_type="dataset",
        #     )
        #     hub_api.create_tag(repo_id, tag=CODEBASE_VERSION, revision=BRANCH, repo_type="dataset")

        #     LeRobotDataset(repo_id).push_to_hub()
        
        # # Push to HuggingFace using upload_large_folder for better handling of large datasets
        # api = HfApi()
        # local_dir = dataset.root  # Get the local directory where dataset is stored
        # api.upload_large_folder(
        #     repo_id=repo_id,
        #     folder_path=local_dir,
        #     repo_type="dataset",
        # )
        # print(f"✓ Successfully pushed {repo_id}")
    
    print(f"\n{'='*60}")
    print("All datasets converted!")
    print(f"{'='*60}")

if __name__ == "__main__":
    convert_individual_datasets()