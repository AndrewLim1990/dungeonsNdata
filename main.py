from creatures import leotris
from creatures import vampire
from actions import vampire_bite
from actions import Move
from environments import square_room


def main():
    vampire.use_action(vampire_bite, target_creature=leotris)
    vampire.use_action(Move(direction="up", distance=10, environment=square_room))


if __name__ == "__main__":
    main()
