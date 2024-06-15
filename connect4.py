import numpy as np

class Gameboard:
    def __init__(self, grid, p1, p1id, p2, p2id):
        self.grid = grid
        self.p1 = p1
        self.p1id = p1id
        self.p2 = p2
        self.p2id = p2id    
    def __str__(self):
        string = ''
        for row in range(6):
            string += '\n| '
            for column in range(7):
                string += str(self.grid[row][column]) + ' '
            string += '|'
        return string
    
    def check_vert_win(self, player):
        if player == self.p1:
            id = self.p1id
        elif player == self.p2:
            id = self.p2id
        else:
            return 'Error'
        count = 0
        win = False
        for column in range(7):
            count = 0
            for row in range(6):
                if board[row][column] == id:
                    count+=1
                if count == 4:
                    win = True
        return win
    
    def check_hort_win(self, player):
        if player == self.p1:
            id = self.p1id
        elif player == self.p2:
            id = self.p2id
        else:
            return 'Error'
        count = 0
        win = False
        for row in range(6):
            count = 0
            for column in range(7):
                if board[row][column] == id:
                    count+=1
                if count == 4:
                    win = True
        return win
    
    def check_nw_se_win(self, player):
        if player == self.p1:
            id = self.p1id
        elif player == self.p2:
            id = self.p2id
        else:
            return 'Error'
        win = False
        count = 0
        for row in range(3):
          for column in range(4):
                if board[row][column] == id:
                    if board[row+1][column+1] == id:
                        if board[row+2][column+2] == id:
                           if board[row+3][column+3] == id:
                                win = True
        return win
    
    def check_ne_sw_win(self, player):
            if player == self.p1:
                id = self.p1id
            elif player == self.p2:
                id = self.p2id
            else:
                return 'Error'
            win = False
            count = 0
            for row in range(3):
                for column in range(3,7):
                    if board[row][column] == id:
                        if board[row-1][column-1] == id:
                            if board[row-2][column-2] == id:
                                if board[row-3][column-3] == id:
                                    win = True
            return win
    
    def check_win(self, player):
        if player == self.p1:
                id = self.p1id
        elif player == self.p2:
            id = self.p2id
        else:
            return 'Error'
        win = False
        count = 0
        for column in range(7):#ccheck vert
            count = 0
            for row in range(6):
                if self.grid[row][column] == id:
                    count+=1
                if count == 4:
                    win = True
        for row in range(6):#check hort
            count = 0
            for column in range(7):
                if board[row][column] == id:
                    count+=1
                if count == 4:
                    win = True
        for row in range(3): #check nw-se
            for column in range(4):
                if board[row][column] == id:
                    if board[row+1][column+1] == id:
                        if board[row+2][column+2] == id:
                           if board[row+3][column+3] == id:
                                win = True
        for row in range(3):    #check ne-sw1
                for column in range(3,7):
                    if board[row][column] == id:
                        if board[row-1][column-1] == id:
                            if board[row-2][column-2] == id:
                                if board[row-3][column-3] == id:
                                    win = True
        return win 
            
    def drop_piece(self, player, column):
        if player == self.p1:
                id = self.p1id
        elif player == self.p2:
            id = self.p2id
        else:
            return 'Error'
        for row in range(5,-1,-1):
            if self.grid[row][column] == 0:
                self.grid[row][column] = id
                break
                



#NEED TO CHANGE BOARD TO SELF.GRID IN CLASS DEFINITION




board = np.array([[0,0,0,0,0,0,0],
                     [0,0,0,0,0,0,1],
                     [0,0,0,1,0,0,1],
                     [0,0,1,0,1,0,1],
                     [0,1,0,0,0,1,1],
                     [1,1,1,1,1,0,1]])

board = Gameboard(board, 'red', '1', 'yellow', '2')

print(board)
board.drop_piece('yellow',0)
print(board)