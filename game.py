import numpy as np
from abc import ABC, abstractmethod

RANKS = {0:8, 1:7, 2:6, 3:5, 4:4, 5:3, 6:2, 7:1}
FILES = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h'}

class Chess():
    def __init__(self):
        # {8'p':2, 10'R':14, 10'N':8, 10'B'13:, 9'Q'27:, 1'K'10:} : 619
        self.row_count = 8
        self.column_count = 8
        self.action_size = 619

    def __repr__(self):
        return "Chess_Game"

    def get_initial_state(self):
        return np.array([
                [Square('bR'), Square('bN'), Square('bB'), Square('bQ'), Square('bK'), Square('bB'), Square('bN'), Square('bR')],
                [Square('bp'), Square('bp'), Square('bp'), Square('bp'), Square('bp'), Square('bp'), Square('bp'), Square('bp')],
                [Square(),     Square(),     Square(),     Square(),     Square(),     Square(),     Square(),     Square()],
                [Square(),     Square(),     Square(),     Square(),     Square(),     Square(),     Square(),     Square()],
                [Square(),     Square(),     Square(),     Square(),     Square(),     Square(),     Square(),     Square()],
                [Square(),     Square(),     Square(),     Square(),     Square(),     Square(),     Square(),     Square()],
                [Square('wp'), Square('wp'), Square('wp'), Square('wp'), Square('wp'), Square('wp'), Square('wp'), Square('wp')],
                [Square('wR'), Square('wN'), Square('wB'), Square('wQ'), Square('wK'), Square('wB'), Square('wN'), Square('wR')],
            ])

    def get_next_state(self, state, action, player):    
        action.pieceMoved.makeMove(action, state)
        return state

    def get_valid_moves(self, state):
        self.resetRange(state)
        self.getAllEnemysMoves(state)
        moves = self.getAllPlayerMoves(state)
        for move in reversed(moves):
            
            move.pieceMoved.makeMove(move, state)
            
            state = self.change_perspective(state, -1)
            
            enemyMoves = self.getAllPlayerMoves(state)
            if self.validate_danger(enemyMoves):
                moves.remove(move)
                
            state = self.change_perspective(state, -1)
            
            move.pieceMoved.undoMoves(move, state) 
            
        return moves
    
    def resetRange(self, state):
        for r in range(len(state)):
            for c in range(len(state[r])):
                state[r,c].inDanger = False
                if isinstance(state[r,c].piece, King):
                    state[r,c].piece.kingInDanger = False
                    
    def getAllEnemysMoves(self, state):
        moves = []       
        for r in range(len(state)):
            for c in range(len(state[r])):
                if state[r,c] != None: 
                    turn = state[r,c].piece.playerColor
                    if (turn == 'b'):
                        moves += state[r,c].piece.getMoves(r, c, state)
        for move in moves:
            state[move.endRow, move.endCol].inDanger = True
            if move.enemyKingInRange:
                move.pieceCaptured.kingInDanger = True

    def getAllPlayerMoves(self, state):
        moves = []
        for r in range(len(state)):
            for c in range(len(state[r])):
                if state[r,c] != None: 
                    turn = state[r,c].piece.playerColor
                    if (turn == 'w'):
                        moves += state[r,c].piece.getMoves(r, c, state)
        return moves
    
    def validate_danger(self, enemyMoves):
        for move in enemyMoves:
            if isinstance(move, Move):
                if move.enemyKingInRange:
                    return True
        return False

    def check_win(self, state, action):
        valid_moves = self.get_valid_moves(state)
        if len(valid_moves) <= 0:
            return True
        return False
    
    def check_tie(self, state, action):
        pieces = 0
        for r in range(len(state)):
            for c in range(len(state[r])):
                if state[r,c] != None: 
                    pieces += 1
                if pieces > 2:
                    return False
        return True

    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if self.check_tie(state, action):
            return 0, True          
        return 0, False

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value

    def change_perspective(self, state, player):
        for i in range(len(state)):
            for j in range(len(state[i])):
                piece = state[i,j].piece
                if player == -1 and piece != None:
                    piece.playerColor = 'w' if piece.playerColor == 'b' else 'b'
        return state

    def get_encoded_state(self, state):
        piecesDict = {'p':2, 'R':3, 'N':4, 'B':5, 'Q':6, 'K':7}
        encoded_board = np.zeros((3, 8, 8)).astype(np.float32)
        for i in range(len(encoded_board)):
            for j in range(len(encoded_board[i])):
                for k in range(len(encoded_board[i][j])):
                    piece = state[j,k].piece
                    if piece != None:
                        if i == 0 and piece.playerColor == 'w':
                            encoded_board[i,j,k] = piecesDict[piece.pieceName]
                        elif i == 2 and piece.playerColor == 'b':
                            encoded_board[i,j,k] = -piecesDict[piece.pieceName]
                    elif i == 1:
                        encoded_board[i,j,k] = 1
        return encoded_board

class Square():
    def __init__(self, piece=None):
        if piece == None:
            self.piece = None
        elif piece[1] == 'p':
            self.piece = Pawn(piece)
        elif piece[1] == 'R':
            self.piece = Rook(piece)
        elif piece[1] == 'N':
            self.piece = Knight(piece)
        elif piece[1] == 'B':
            self.piece = Bishop(piece)
        elif piece[1] == 'Q':
            self.piece = Queen(piece)
        elif piece[1] == 'K':
            self.piece = King(piece)
        self.inDanger = False
    
    def __eq__(self, value) -> bool:
        return self.piece == value
    
    def __repr__(self) -> str:
        if (self.piece != None):
            return self.piece.__repr__()
        else:
            return '--'
    
    def resetSquare(self):
        self.inDanger = False

class Move():
    # maps keys to values
    # key : value
    ranksToRows = {"1":7, "2":6, "3":5, "4":4,
                   "5":3, "6":2, "7":1, "8":0}
    rowsToRanks = {v:k for k, v in ranksToRows.items()}
    filesToCols = {"a":0, "b":1, "c":2, "d":3,
                   "e":4, "f":5, "g":6, "h":7}
    colsToFiles = {v:k for k, v in filesToCols.items()}

    def __init__(self, startSq, endSq, board):
        self.startRow = startSq[0]
        self.startCol = startSq[1]
        self.endRow = endSq[0]
        self.endCol = endSq[1]
        self.pieceMoved = board[self.startRow][self.startCol].piece
        self.pieceCaptured = board[self.endRow][self.endCol].piece
        self.enemyKingInRange = self.checkKing()
        self.moveID = '{}{}{}{}'.format(self.startRow,self.startCol,self.endRow,self.endCol)

    def __eq__(self, other) -> bool:
        if isinstance(other, Move):
            return int(self.moveID) == int(other.moveID)
        
    def __repr__(self) -> str:
        return '{}{}{}{}'.format(FILES[self.startCol],RANKS[self.startRow],FILES[self.endCol],RANKS[self.endRow])      

    def getChessNotation(self):
        return self.getRankFile(self.startRow, self.startCol) + self.getRankFile(self.endRow, self.endCol)
    
    def getRankFile(self, r, c):
        return self.colsToFiles[c] + self.rowsToRanks[r]
    
    def checkKing(self):
        if isinstance(self.pieceCaptured, King):
            return True
        return False

class Pieces(ABC):
    def __init__(self, name):
        self.playerColor = name[0]
        self.pieceName = name[1]
        self.pieceMoves = []
        self.moveUp = False if self.playerColor == 'b' else True

    @abstractmethod
    def getMoves(self, r, c, board):
        pass

    def createMoves(self, r, c, board, direction):
        i = direction[0]
        j = direction[1]
        condition = False
        while not condition:
            condition = 0 > (r + i) or (r + i) > 7 or 0 > (c + j) or (c + j) > 7

            if condition:
                break

            if board[r+i][c+j] != None and board[r+i][c+j].piece.playerColor == self.playerColor:
                break

            self.pieceMoves.append(Move((r,c), (r+i,c+j), board))
            if board[r+i][c+j] != None and board[r+i][c+j].piece.playerColor != self.playerColor:
                break

            i += direction[0]
            j += direction[1]

    def makeMove(self, move, board):
        board[move.startRow][move.startCol].piece = None
        board[move.endRow][move.endCol].piece = move.pieceMoved

    def undoMoves(self, move, board):
        board[move.startRow][move.startCol].piece = move.pieceMoved
        board[move.endRow][move.endCol].piece = move.pieceCaptured

    def __repr__(self) -> str:
        return self.playerColor + self.pieceName

class Pawn(Pieces):
    def __init__(self, name):
        super().__init__(name)

    class Move(Move):
        def __init__(self, startSq, endSq, board):
            super().__init__(startSq, endSq, board)
            # En passant Move
            self.enPassant = False
            self.passantRow = None
            self.passantCol = None
            # Pawn promotion Move
            self.pawnPromotion = False
            self.piecePromoted = None  

    def getMoves(self, r, c, board):
        self.pieceMoves = []

        if self.moveUp:
            if r-1 >= 0 and board[r-1][c] == None:
                self.createMoves(r, c, r-1, c, board)

                if r == 6 and board[r-2][c] == None:
                    self.createMoves(r, c, r-2, c, board)

            if c-1 >= 0 and r-1 >= 0:
                canEnPassant = self.canEnPassant(r, c-1)
                if board[r-1][c-1] != None:
                    if board[r-1][c-1].piece.playerColor != self.playerColor:
                        self.createMoves(r, c, r-1, c-1, board)
                elif canEnPassant:
                    move = self.Move((r, c), (r-1, c-1), board)
                    self.pieceMoves.append(self.enPassant(r,c-1, board[r][c-1].piece, move))

            if c+1 <= 7 and r-1 >= 0:
                canEnPassant = self.canEnPassant(r, c+1)
                if board[r-1][c+1] != None:
                    if board[r-1][c+1].piece.playerColor != self.playerColor:
                        self.createMoves(r, c, r-1, c+1, board)
                elif canEnPassant:
                    move = self.Move((r, c), (r-1, c+1), board)
                    self.pieceMoves.append(self.enPassant(r, c+1, board[r][c+1].piece, move))

        elif not self.moveUp:
            if r+1 <= 7 and board[r+1][c] == None:
                self.createMoves(r, c, r+1, c, board)

                if r == 1 and board[r+2][c] == None:
                    self.createMoves(r, c, r+2, c, board)

            if c-1 >= 0 and r+1 <= 7:
                canEnPassant = self.canEnPassant(r, c-1)
                if board[r+1][c-1] != None:
                    if board[r+1][c-1].piece.playerColor != self.playerColor:
                        self.createMoves(r, c, r+1, c-1, board)
                elif canEnPassant:
                    move = self.Move((r, c), (r+1, c-1), board)
                    self.pieceMoves.append(self.enPassant(r, c-1, board[r][c-1].piece, move))

            if c+1 <= 7 and r+1 <= 7:
                canEnPassant = self.canEnPassant(r, c+1)
                if board[r+1][c+1] != None:
                    if board[r+1][c+1].piece.playerColor != self.playerColor:
                        self.createMoves(r, c, r+1, c+1, board)
                elif canEnPassant:
                    move = self.Move((r, c), (r+1, c+1), board)
                    self.pieceMoves.append(self.enPassant(r, c+1, board[r][c+1].piece, move))

        return self.pieceMoves 
    
    def createMoves(self, old_r, old_c, new_r, new_c, board):
        if self.canPawnPromoted(new_r):
            pieces = (Bishop(self.playerColor+'B'), Queen(self.playerColor+'Q'), Knight(self.playerColor+'N'), Rook(self.playerColor+'R'))
            for piece in pieces:
                move = self.Move((old_r, old_c), (new_r, new_c), board)
                self.pieceMoves.append(self.pawnPromotion(move, piece))
        else:
            self.pieceMoves.append(self.Move((old_r, old_c), (new_r, new_c), board))        

    def canPawnPromoted(self, r):
        if (r == 0 or r == 7) and self.pieceName == 'p':
            return True
        return False  
    
    def pawnPromotion(self, move, piece):
        move.piecePromoted = piece
        move.pawnPromotion = True
        return move

    def canEnPassant(self, r, c):
        return False
        #TODO:
        #if gameState == None:
        #    return False
        #lastMove = gameState.getLastMove()
        #if lastMove == None:
        #    return False
        #lastMoveCondition = abs(lastMove.endRow - lastMove.startRow) == 2 and lastMove.pieceMoved.pieceName == 'p'
        #currentMoveCondition = lastMove.endRow == r and c == lastMove.endCol
        #if lastMoveCondition and currentMoveCondition and lastMove.pieceMoved.playerColor != self.playerColor:
        #    return True
        #return False
    
    def enPassant(self, r, c, piece, move):
        move.pieceCaptured = piece
        move.passantRow = r
        move.passantCol = c
        move.enPassant = True
        return move
    
    def makeMove(self, move, board):
        super().makeMove(move, board)
        if move.enPassant:
            pass
            #TODO:
            #board[move.passantRow][move.passantCol].piece = None
        if move.pawnPromotion:
            board[move.endRow][move.endCol].piece = move.piecePromoted

    def undoMoves(self, move, board):
        super().undoMoves(move, board)
        if move.enPassant:
            board[move.endRow][move.endCol].piece = None
            board[move.passantRow][move.passantCol].piece = move.pieceCaptured

    def __eq__(self, value) -> bool:
        return self.playerColor + self.pieceName == value
    
    def __str__(self) -> str:
        return self.playerColor + self.pieceName


class Rook(Pieces):
    def __init__(self, name):
        super().__init__(name)
        self.rookMovements = []    

    def getMoves(self, r, c, board):
        self.pieceMoves = []
        directions = ((1,0),(-1,0),(0,1),(0,-1))

        if self.moveUp:
            for d in directions:
                self.createMoves(r, c, board, d)

        elif not self.moveUp:
            for d in directions:
                self.createMoves(r, c, board, d)

        return self.pieceMoves
    
    def hasRookMove(self):
        if len(self.rookMovements) >= 1:
            return True
        return False
    
    def makeMove(self, move, board):
        super().makeMove(move, board)
        self.rookMovements.append(move)

    def undoMoves(self, move, board):
        super().undoMoves(move, board)
        self.rookMovements.remove(move)

    def __eq__(self, value) -> bool:
        return self.playerColor + self.pieceName == value
    
    def __str__(self) -> str:
        return self.playerColor + self.pieceName

class Knight(Pieces):
    def __init__(self, name):
        super().__init__(name)

    def getMoves(self, r, c, board):
        self.pieceMoves = []
        directions = ((1,2),(-1,2),(1,-2),(-1,-2),(2,1),(-2,1),(2,-1),(-2,-1))

        if self.moveUp:
            for d in directions:
                self.createMoves(r, c, board, d)

        elif not self.moveUp:
            for d in directions:
                self.createMoves(r, c, board, d)

        return self.pieceMoves

    def createMoves(self, r, c, board, direction):
        i = direction[0]
        j = direction[1]
        condition = 0 > (r + i) or (r + i) > 7 or 0 > (c + j) or (c + j) > 7
        if not condition and (board[r+i][c+j] == None or board[r+i][c+j].piece.playerColor != self.playerColor):
            self.pieceMoves.append(Move((r,c), (r+i, c+j), board))
    
    def makeMove(self, move, board):
        super().makeMove(move, board)

    def undoMoves(self, move, board):
        super().undoMoves(move, board)

    def __eq__(self, value) -> bool:
        return self.playerColor + self.pieceName == value
    
    def __str__(self) -> str:
        return self.playerColor + self.pieceName
    
class Bishop(Pieces):
    def __init__(self, name):
        super().__init__(name)

    def getMoves(self, r, c, board):
        self.pieceMoves = []
        directions = ((1, 1), (-1, 1), (1, -1), (-1, -1))

        if self.moveUp:
            for d in directions:
                self.createMoves(r, c, board, d)

        elif not self.moveUp:
            for d in directions:
                self.createMoves(r, c, board, d)

        return self.pieceMoves
    
    def makeMove(self, move, board):
        super().makeMove(move, board)

    def undoMoves(self, move, board):
        super().undoMoves(move, board)

    def __eq__(self, value) -> bool:
        return self.playerColor + self.pieceName == value
    
    def __str__(self) -> str:
        return self.playerColor + self.pieceName

class Queen(Pieces):
    def __init__(self, name):
        super().__init__(name)

    def getMoves(self, r, c, board):
        self.pieceMoves = []
        directions = ((1, 1), (-1, 1), (1, -1), (-1, -1), (1,0), (-1,0), (0,1), (0,-1))

        if self.moveUp:
            for d in directions:
                self.createMoves(r, c, board, d)

        elif not self.moveUp:
            for d in directions:
                self.createMoves(r, c, board, d)

        return self.pieceMoves
    
    def makeMove(self, move, board):
        super().makeMove(move, board)

    def undoMoves(self, move, board):
        super().undoMoves(move, board)

    def __eq__(self, value) -> bool:
        return self.playerColor + self.pieceName == value
    
    def __str__(self) -> str:
        return self.playerColor + self.pieceName
    
class King(Pieces):
    def __init__(self, name):
        super().__init__(name)
        self.kingMovements = []
        self.kingInDanger = False

    class Move(Move):
        def __init__(self, startSq, endSq, board, kingSq=None, rookSq=None):
            super().__init__(startSq, endSq, board)
            # Castling Move
            self.realKingRow, self.realKingCol, self.realRookRow, self.realRookCol = self.canCastling(kingSq, rookSq)

        def canCastling(self, kingSq, rookSq):
            if kingSq == None or rookSq == None:
                return None, None, None, None
            return kingSq[0], kingSq[1], rookSq[0], rookSq[1]

    def getMoves(self, r, c, board):
        self.pieceMoves = []
        directions = ((1, 1), (-1, 1), (1, -1), (-1, -1), (1,0), (-1,0), (0,1), (0,-1))

        if self.moveUp:
            for d in directions:
                self.createMoves(r, c, board, d)

        elif not self.moveUp:
            for d in directions:
                self.createMoves(r, c, board, d)
            
        self.castlingKing(r, c, board)

        return self.pieceMoves

    def createMoves(self, r, c, board, direction):
        i = direction[0]
        j = direction[1]
        if (0 <= (r + i) <= 7 and 0 <= (c + j) <= 7) and (board[r+i][c+j] == None or board[r+i][c+j].piece.playerColor != self.playerColor):
            self.pieceMoves.append(self.Move((r, c), (r+i, c+j), board))

    def castlingKing(self,r, c, board):
        
        def checkMove(j:int, finishCol, kingRow, rookRow):
            for i in range(c+j,finishCol,j):
                square = board[r][i]
                if square.inDanger:
                    return
                if square.piece == None:
                    continue
                return
            piece = board[r][finishCol].piece
            if piece == None:
                return
            if piece.playerColor != self.playerColor:
                return
            if not isinstance(piece, Rook):
                return
            if piece.hasRookMove():
                return
            self.pieceMoves.append(self.Move((r, c), (r, finishCol), board, (r,kingRow),(r,rookRow)))

        if self.hasKingMove() or self.kingInDanger or c != 4:
            return
        
        checkMove(1, 7, 6, 5)
        checkMove(-1, 0, 2, 3)
            
    def hasKingMove(self):
        if len(self.kingMovements) >= 1:
            return True
        return False
    
    def makeMove(self, move, board):
        self.kingMovements.append(move)
        if move.realKingRow != None and move.realRookRow != None:
            board[move.realKingRow][move.realKingCol].piece = move.pieceMoved
            board[move.realRookRow][move.realRookCol].piece = move.pieceCaptured
            board[move.startRow][move.startCol].piece = None
            board[move.endRow][move.endCol].piece = None
        else:
            super().makeMove(move, board)

    def undoMoves(self, move, board):
        super().undoMoves(move, board)
        self.kingMovements.remove(move)
        if move.realKingRow != None and move.realRookRow != None:
            board[move.realKingRow][move.realKingCol].piece = None
            board[move.realRookRow][move.realRookCol].piece = None 

    def __eq__(self, value) -> bool:
        return self.playerColor + self.pieceName == value
    
    def __str__(self) -> str:
        return self.playerColor + self.pieceName

