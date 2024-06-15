import numpy as np

print('\n'*3)
class Gameboard:
    def __init__(self):
        self.reset()

    def reset(self):
        self.grid = np.zeros((6, 7), dtype=int)
        self.red_score = 0
        self.yellow_score = 0
        self.turns_played = 0
        self.done = False
        self.winner = None
        self.wincon = 'none'

    def __str__(self):
        string = ''
        for row in range(6):
            string += '\n| '
            for column in range(7):
                string += str(self.grid[row][column]) + ' '
            string += '|'
        return string
        
    def get_state(self):
        return self.grid.reshape(-1)
    
    def get_grid(self):
        return self.grid

    def get_score(self, player=None):
        if player == 1:
            return self.red_score - self.yellow_score
        if player == 2:
            return self.yellow_score - self.red_score
        else:
            return 0
        
        
    def is_valid_action(self, action):
        # Check if the top row of the selected column is empty
        return self.grid[0][action] == 0


    def turn(self, player, column):
        # Drop piece
        self.turns_played += 1
        if player == 1:
            piece = 1
        else:
            piece = 2
        for row in range(5, -1, -1):
            if self.grid[row][column] == 0:  # Change from != 0 to == '0'
                self.grid[row][column] = piece
                break
        # Calculate scores
        red_vert = self.score_vert(1, 1)
        yellow_vert = self.score_vert(2, 2)
        red_hort = self.score_hort(1, 1)
        yellow_hort = self.score_hort(2, 2)
        red_sw = self.score_sw(1, 1)
        yellow_sw = self.score_se(2, 2)
        red_se = self.score_sw(1, 1)
        yellow_se = self.score_se(2, 2)
        
        self.red_score = 10*(red_vert + red_hort + red_sw + red_se) + self.score_count(1)
        self.yellow_score = 10*(yellow_vert + yellow_hort + yellow_sw + yellow_se) + self.score_count(2)
        if self.turns_played>=42:
            self.done = True
            self.wincon = 'draw'
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
            self.wincon = 'vert'
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
            self.wincon = 'hort'
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
            self.wincon = 'sw'
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
            self.wincon = 'se'
        score = twos + 10*threes + 100*fours
        return score
    def score_count(self, piece):
        count = np.count_nonzero(self.grid == piece)
        return count
    
    def is_done(self):
        return self.done
# # Create an instance of Gameboard
# board = Gameboard()

# # Call turn method on the instance
# board.turn(1, 2)
# board.turn(1,2)

# # Print the board
# player = 1
# print(board)
# print(f'Score: {board.get_score(player)}')
