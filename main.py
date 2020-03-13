from creatures import leotris
from creatures import vampire
from actions import vampire_bite


def main():
    vampire.use_action(vampire_bite, target_creature=leotris)


if __name__ == "__main__":
    main()
