# OpLearn

## Dependencies

Clone MimicGen in /OpLearn
```bash
gh repo clone https://github.com/helenlu66/mimicgen
```

Clone RoboSuite in /OpLearn
```bash
gh repo clone https://github.com/ARISE-Initiative/robosuite.git
```

Install tarski - coming soon

## Docker Container (Optional)
There is an optional docker container for you to use. The docker-compose is setup to use Nvidia GPUs. You will need to install the Nvidia Container Toolkit to use this. 

### Installing Nvidia Container Toolkit
Instructions for installing the Nvidia Container Toolkit can be found here:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html 

### Building and running the Docker container

Building
```bash
docker compose build
```

Running
```bash
docker compose up
```

Closing runaway containers. If you get an error on a subsequent docker build. It's likely because part of the container is still running in the background. You can use this command to stop it.

```bash
docker compose down -v
```

### Connecting to VS Code (Optional)
Install the dev containers extension in VS Code

While the container is up, press F1 in VS Code and type in the command pallete "Dev Containers: Attach to running container..."

Select the running container. In my case it says "/oplearn_oplearn_1"

Open the /home/user/oplearn directory to start working on the files from within the container.

## Run Demo
You can test your mujoco installation first with 
```bash
python test_mujoco.py
```

You can test your Robosuite installation with 
```bash
python robosuite/robosuite/demos/demo_random_action.py
```

You can test your MimicGen installation with 
```bash
python mimicgen/mimicgen/scripts/demo_random_action.py
```