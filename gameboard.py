import numpy as np

print('\n'*3)
class Gameboard:
    def __init__(self):
        self.reset()

    def reset(self):
        self.grid = np.zeros((6, 7), dtype=int)
        self.grid[self.grid == ''] = 0  # Ensure the grid is filled with '0'
        self.red_score = 0
        self.yellow_score = 0
        self.turns_played = 0
        self.done = False
        self.winner = None

    def __str__(self):
        return str(self.grid)
        
    def get_state(self):
        return self.grid

    def get_score(self, player=None):
        if player == 'red':
            return self.red_score - self.yellow_score
        if player == 'yellow':
            return self.yellow_score - self.red_score
        else:
            return 0

    def turn(self, player, column):
        # Drop piece
        if player == 'red':
            piece = 1
        else:
            piece = 2
        for row in range(5, -1, -1):
            if self.grid[row][column] == 0:  # Change from != 0 to == '0'
                self.grid[row][column] = piece
                break
        # Calculate scores
        red_vert = self.score_vert('red', 1)
        yellow_vert = self.score_vert('yellow', 2)
        red_hort = self.score_hort('red', 1)
        yellow_hort = self.score_hort('yellow', 2)
        red_sw = self.score_sw('red', 1)
        yellow_sw = self.score_se('yellow', 2)
        red_se = self.score_sw('red', 1)
        yellow_se = self.score_se('yellow', 2)
        
        self.red_score = red_vert + red_hort + red_sw + red_se
        self.yellow_score = yellow_vert + yellow_hort + yellow_sw + yellow_se
        
        return self.get_score(player), self.done, self.winner
        # Check if done
    
    def score_vert(self, player, piece):
        counts = []
        for column in range(7):
            for row in range(6):
                count = 0
                while(row<6):
                    if self.grid[row][column] != piece:
                        break
                    count += 1
                    row += 1
                counts.append(count)
        twos = counts.count(2)
        threes = counts.count(3)
        fours = 0
        if counts.count(4) > 0:
            fours = 1
            self.winner = player
            self.done = True
        score = twos + 10*threes + 100*fours
        return score
    
    def score_hort(self, player, piece):
        counts = []
        for row in range(6):
            for column in range(6):
                count = 0
                while(column<7):
                    if self.grid[row][column] != piece:
                        break
                    count += 1
                    column += 1
                counts.append(count)
        twos = counts.count(2)
        threes = counts.count(3)
        fours = 0
        if counts.count(4) > 0:
            fours = 1
            self.winner = player
            self.done = True
        score = twos + 10*threes + 100*fours
        return score
    
    def score_sw(self, player, piece):
        counts = []
        for row in range(6):
            for column in range(7):
                count = 0
                while(column<7 and row > -1):
                    if self.grid[row][column] != piece:
                        break
                    count += 1
                    column += 1
                    row  -= 1
                counts.append(count)
        twos = counts.count(2)
        threes = counts.count(3)
        fours = 0
        if counts.count(4) > 0:
            fours = 1
            self.winner = player
            self.done = True
        score = twos + 10*threes + 100*fours
        return score
        
    def score_se(self, player, piece):
        counts = []
        for row in range(7):
            for column in range(6):
                count = 0
                while(column<7 and row < 6):
                    if self.grid[row][column] != piece:
                        break
                    count += 1
                    column += 1
                    row  += 1
                counts.append(count)
        twos = counts.count(2)
        threes = counts.count(3)
        fours = 0
        if counts.count(4) > 0:
            fours = 1
            self.winner = player
            self.done = True
        score = twos + 10*threes + 100*fours
        return score
# Create an instance of Gameboard
board = Gameboard()

# Call turn method on the instance
board.turn('red', 2)
board.turn('red',2)

# Print the board
player = 'red'
print(board)
print(f'Score: {board.get_score(player)}')
