import random
import cv2
from teachable import TeachableModel

model_path = "./converted_savedmodel/converted_savedmodel/model.savedmodel/"
labels_path = "./converted_savedmodel/converted_savedmodel/labels.txt"

model = TeachableModel(model_path, labels_path)

rps_choices = ["rock", "paper", "scissors"]

def get_winner(player, computer):
    if player == computer:
        return "Draw!"
    elif (player == "rock" and computer == "scissors") or \
         (player == "paper" and computer == "rock") or \
         (player == "scissors" and computer == "paper"):
        return "You Win!"
    else:
        return "Computer Wins!"

def play_game():
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit the game.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_tensor = model.preprocess_image(frame)
        predictions = model.get_prediction(image_tensor)
        player_move, confidence = model.get_classification(predictions)

        if player_move:
            computer_move = random.choice(rps_choices)
            result = get_winner(player_move, computer_move)
            text = f"You: {player_move} | Computer: {computer_move} -> {result}"
            cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Rock Paper Scissors", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    play_game()
