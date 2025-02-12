import warnings
warnings.filterwarnings("ignore")

import argparse
import robosuite as suite
import numpy as np
from robosuite.wrappers import GymWrapper
from robosuite.utils.detector import HanoiDetector
from robosuite.wrappers.hanoi.hanoi_reach_and_pick import ReachAndPickWrapper
# diffusion policy import
from ultralytics import YOLO
import cv2



from robosuite.devices import Keyboard
from robosuite.utils.input_utils import input2action

# env import
import gymnasium as gym
from digits_detection.src.object_detection.with_onnx import OnnxObjectDetector

if __name__ == "__main__":
    # Define the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()
    np.random.seed(args.seed)

    # # Load ONNX models
    # preprocess_model = ort.InferenceSession("./segmentation/preprocessing.onnx")
    # yolo_model = ort.InferenceSession("./segmentation/yolo.onnx")
    # nms_model = ort.InferenceSession("./segmentation/nms.onnx")
    # postprocess_model = ort.InferenceSession("./segmentation/postprocessing.onnx")

    # digit_detector = OnnxObjectDetector(preprocessing_path="./segmentation/preprocessing.onnx",yolo_path="./segmentation/yolo.onnx",nms_path="./segmentation/nms.onnx",postprocessing_path="./segmentation/postprocessing.onnx")

    # Load YOLOv8 pre-trained model
    model = YOLO("/home/lorangpi/CyclicLxM/detection/runs/detect/train/weights/best.pt")  # Use "yolov8n.pt" or train your own model on numbers


    # Load the controller config
    controller_config = suite.load_controller_config(default_controller='OSC_POSITION')

    env = suite.make(
        "Hanoi",
        robots="Kinova3",
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=True,
        horizon=1000000,
        use_camera_obs=True,
        camera_heights=256,
        camera_widths=256,
        use_object_obs=False,
        #camera_segmentations='element',
        render_camera="agentview",#"robot0_eye_in_hand", # Available "camera" names = ('frontview', 'birdview', 'agentview', 'robot0_robotview', 'robot0_eye_in_hand')
    )

    # Wrap the environment
    env = GymWrapper(env, proprio_obs=False)
    env = ReachAndPickWrapper(env)

    device = Keyboard()
    env.viewer.add_keypress_callback(device.on_press)
    device.start_control()

    #Reset the environment
    try:
        obs, info = env.reset()
    except Exception as e:
        obs = env.reset()
        info = None
    
    state = info['state']
    obj_to_pick = 'RoundNut'
    gripper_body = env.sim.model.body_name2id('gripper0_eef')


    while True:
        # Set active robot
        active_robot = env.robots[0]

        # Get the newest action
        action, grasp = input2action(
            device=device, robot=active_robot, active_arm="right", env_configuration="single-arm-opposed"
        )

        # If action is none, then this a reset so we should break
        if action is None:
            break

        # If the current grasp is active (1) and last grasp is not (-1) (i.e.: grasping input just pressed),
        # toggle arm control and / or camera viewing angle if requested

        # Update last grasp
        last_grasp = grasp

        # Fill out the rest of the action space if necessary
        rem_action_dim = env.action_dim - action.size
        if rem_action_dim > 0:
            # Initialize remaining action space
            rem_action = np.zeros(rem_action_dim)
            # This is a multi-arm setting, choose which arm to control and fill the rest with zeros
            action = np.concatenate([action, rem_action])

        elif rem_action_dim < 0:
            # We're in an environment with no gripper action space, so trim the action space to be the action dim
            action = action[: env.action_dim]

        # Step through the simulation and render
        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except:
            obs, reward, done, info = env.step(action)
        

        # Load image
        #image = cv2.imread("./segmentation/number1.png")
        # reshape image to 256x256X3
        image = cv2.flip(obs.reshape(256, 256, 3), 0)
        confidence_threshold = 0.10

        # Run YOLO inference
        # Perform prediction
        results = model.predict(source=image, conf=confidence_threshold)
        # Keep only the highest confidence detection per class
        filtered_results = []
        for result in results:
            # Create a dictionary to store the highest confidence detection for each class
            highest_confidence_detections = {}
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = box.conf[0]
                if cls not in highest_confidence_detections or conf > highest_confidence_detections[cls].conf[0]:
                    highest_confidence_detections[cls] = box
            filtered_results.append(highest_confidence_detections.values())

        # Draw bounding boxes around detected numbers
        for r in filtered_results:
            for box in r:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, str(int(box.cls[0])), (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Detected Numbers", image)
        cv2.waitKey(1)

        # # Draw bounding boxes around detected numbers
        # for r in filtered_results:
        #     for i, box in enumerate(r.boxes):
        #         x1, y1, x2, y2 = map(int, box.xyxy[0])
        #         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #         cv2.putText(image, r.names[int(box.cls[0])], (x1, y1 - 10),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # cv2.imshow("Detected Numbers", image)
        # cv2.waitKey(1)

        # # Run digit detection
        # detected_digits = digit_detector(image).visualize()
        # print("Detected Digits:", detected_digits)
        # print("Detected Digits:", detected_digits.shape)

        # print("Detected Digits:", detected_digits)
        # # Show detected digits
        # cv2.imshow("Detected Digits", image)
        # cv2.imshow("Detected Digit", detected_digits)
        # cv2.waitKey(1)


        # # Load image
        # #image = cv2.imread("./segmentation/number1.png")
        # # reshape image to 256x256X3
        # image = cv2.flip(obs.reshape(256, 256, 3), 0)

        # # Load pre-trained model (assume you have a trained model)
        # model = load_model("digit_recognition_model.h5")

        # # Load image
        # image = cv2.resize(image, (28, 28))  # Resize to MNIST size
        # image = image.reshape(1, 28, 28, 1) / 255.0  # Normalize

        # # Predict
        # prediction = model.predict(image)
        # digit = np.argmax(prediction)

        # print("Predicted Digit:", digit)


        # Load and process the observation image
        # obs = obs.reshape(256, 256, 3)
        # obs = cv2.flip(obs, 0)  # Mirror vertically

        # # Load image
        # obs = obs.reshape(256, 256, 3)
        # obs = cv2.flip(obs, 0)

        # # Convert to grayscale
        # obs_gray = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)

        # # Apply OCR to detect numbers
        # custom_config = r'--oem 3 --psm 6 outputbase digits'  # Detect only digits
        # numbers = pytesseract.image_to_data(obs_gray, config=custom_config, output_type=pytesseract.Output.DICT)

        # # Draw bounding boxes around detected numbers
        # for i in range(len(numbers['text'])):
        #     if numbers['text'][i].strip().isdigit():
        #         x, y, w, h = numbers['left'][i], numbers['top'][i], numbers['width'][i], numbers['height'][i]
        #         cv2.rectangle(obs, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #         cv2.putText(obs, numbers['text'][i], (x, y - 10),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # cv2.imshow("Detected Numbers", obs)
        # cv2.waitKey(1)






        new_state = info['state']
        
        if new_state != state:
            # CHeck if the change is linked a grounded predicate 'on(o1,o2)' or 'clear(o1)'
            diff = {k: new_state[k] for k in new_state if k not in state or new_state[k] != state[k]}
            # If any key in diff has 'on' or 'clear' in it, print the change and the new state
            #if any(['on' in k or 'clear' in k for k in diff]):
            if any(['on' in k for k in diff]):
                print("Change detected: {}".format(diff))
                print("State: {}".format(new_state))
                print("\n\n")
                state = new_state


        #print("Obs: {}\n\n".format(obs))
        env.render()
