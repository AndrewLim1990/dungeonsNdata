class Arena:
    """
    Represents the terrain. Ex: 30x30 ft grid
    """
    def __init__(self, name, *args, **kwargs):
        self.name = name
    pass


class SquareRoom(Arena):
    """
    Square room
    Okay, technically a rectangle might've been the better name for this
    """
    def __init__(self, room_width=30, room_length=30, *args, **kwargs):
        super().__init__(name="Square Room", *args, **kwargs)
        self.room_width = room_width
        self.room_length = room_length

    def check_if_legal(self, target_location):
        """
        Checks to see if movement is legal

        Todo: Implement this
        """
        x = target_location[0]
        y = target_location[1]
        if (x < 5) or (x >= (self.room_width - 5)):
            return False
        elif (y < 5) or (y >= (self.room_length - 5)):
            return False
        else:
            return True

    def draw_board(self, console, charset="+-| "):
        width = int(self.room_width / 5)
        length = int(self.room_length / 5)

        h_line = charset[0] + charset[1] * (width - 2) + charset[0]
        v_line = charset[2] + charset[3] * (width - 2) + charset[2]

        console.clear()
        console.addstr(0, 0, h_line)
        for line in range(1, length):
            console.addstr(line, 0, v_line)

        console.addstr(length - 1, 0, h_line)
        console.refresh()


# Todo: Place this into DB
square_room = SquareRoom(room_width=50, room_length=50)
