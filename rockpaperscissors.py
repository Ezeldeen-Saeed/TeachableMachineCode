import random
import cv2
from teachable import TeachableModel

model_path = "./converted_savedmodel/converted_savedmodel/model.savedmodel/"
labels_path = "./converted_savedmodel/converted_savedmodel/labels.txt"

model = TeachableModel(model_path, labels_path)

choices = ["rock", "paper", "scissors"]

def play_game():
    computer_choice = random.choice(choices)

    print("Get ready! Show your hand to the camera...")
    for i in range(4, 0, -1):
        print(i)
        time.sleep(1)

    prediction = model.predict()
    player_choice = prediction.lower()
    print(f"\nYou played: {player_choice}")
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