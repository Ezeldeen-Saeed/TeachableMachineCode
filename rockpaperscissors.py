import random
import cv2
import time
from teachable import TeachableModel, CameraHandler

model_path = "./converted_savedmodel/converted_savedmodel/model.savedmodel/"
labels_path = "./converted_savedmodel/converted_savedmodel/labels.txt"

model = TeachableModel(model_path, labels_path)
camera_path = CameraHandler.get_camera_path()

choices = ["rock", "paper", "scissors"]

def play_game():
    if camera_path is None:
        print("No camera found!")
        return

    computer_choice = random.choice(choices)

    print("\nGet ready! Show your hand to the camera...")
    for i in range(3, 0, -1):
        print(i)
        time.sleep(1)

    cap = cv2.VideoCapture(camera_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to capture image.")
        return

    # Process frame
    image_tensor = model.preprocess_image(frame)
    predictions = model.get_prediction(image_tensor)
    player_choice, confidence = model.get_classification(predictions)

    if not player_choice:
        print("Could not detect your move clearly. Try again!")
        return

    player_choice = player_choice.lower()
    print(f"\nYou played: {player_choice} ({confidence:.2f})")
    print(f"Computer played: {computer_choice}")

    if player_choice == computer_choice:
        print("It's a tie!")
    elif (player_choice == "rock" and computer_choice == "scissors") or \
         (player_choice == "scissors" and computer_choice == "paper") or \
         (player_choice == "paper" and computer_choice == "rock"):
        print("You win! 🎉")
    else:
        print("Computer wins! 🤖")

while True:
    play_game()
    again = input("\nDo you want to play again? (y/n): ").strip().lower()
    if again != "y":
        print("Thanks for playing! Goodbye 👋")
        break