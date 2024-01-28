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
                ['bR', 'bN', 'bB', 'bQ', 'bK', 'bB', 'bN', 'bR'],
                ['bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp'],
                ['--', '--', '--', '--', '--', '--', '--', '--'],
                ['--', '--', '--', '--', '--', '--', '--', '--'],
                ['--', '--', '--', '--', '--', '--', '--', '--'],
                ['--', '--', '--', '--', '--', '--', '--', '--'],
                ['wpU', 'wpU', 'wpU', 'wpU', 'wpU', 'wpU', 'wpU', 'wpU'],
                ['wR', 'wN', 'wB', 'wQ', 'wK', 'wB', 'wN', 'wR'],
        ], dtype=np.object_)

    def get_next_state(self, state, action, player):    
        state = self.make_move(action, state)
        return state

    def get_valid_moves(self, state):
        self.resetRange(state)
        self.getAllEnemysMoves(state)
        moves = self.getAllPlayerMoves(state)
        for move in reversed(moves):
            
            self.make_move(move, state)
            
            state = self.change_perspective(state, -1)
            
            enemyMoves = self.getAllPlayerMoves(state)
            if self.validate_danger(enemyMoves):
                moves.remove(move)
                
            state = self.change_perspective(state, -1)
            
            self.undo_moves(move, state) 
        self.resetRange(state)
            
        return moves
    
    def resetRange(self, state):
        for r in range(len(state)):
            for c in range(len(state[r])):
                state[r,c] = state[r,c].replace('D', '')
                #if 'K' in state[r,c]:
                 #   state[r,c] = False
                    
    def getAllEnemysMoves(self, state):
        moves = []       
        for r in range(len(state)):
            for c in range(len(state[r])):
                if '--' not in state[r,c]: 
                    turn = state[r,c][0]
                    if (turn == 'b'):
                        moves += self.get_moves(r, c, state)
        for move in moves:
            if 'R' in state[move.endRow, move.endCol] or 'K' in state[move.endRow, move.endCol]:
                if 'N' in state[move.endRow, move.endCol]:
                    piece = state[move.endRow, move.endCol].rsplit('N', 1)
                    state[move.endRow, move.endCol] = f'{piece[0]}DN{piece[1]}'
                    continue          
            state[move.endRow, move.endCol] += 'D'

    def getAllPlayerMoves(self, state):
        moves = []
        for r in range(len(state)):
            for c in range(len(state[r])):
                if '--' not in state[r,c]: 
                    turn = state[r,c][0]
                    if (turn == 'w'):
                        moves += self.get_moves(r, c, state)
        return moves
    
    def validate_danger(self, enemyMoves):
        for move in enemyMoves:
            if isinstance(move, Move):
                if move.enemyKingInRange:
                    return True
        return False
    
    def get_moves(self, r, c, state):
        playerPiece = state[r, c][1]

        if 'p' in playerPiece:
            return self.pawns_move(r, c, state)
        elif 'R' in playerPiece:
            return self.rooks_move(r, c, state)
        elif 'N' in playerPiece:
            return self.knight_move(r, c, state)
        elif 'B' in playerPiece:
            return self.bishop_moves(r, c, state)
        elif 'Q' in playerPiece:
            return self.queen_moves(r, c, state)
        elif 'K' in playerPiece:
            return self.king_moves(r, c, state)   
        else:
            return []
          
    def pawns_move(self, r, c, board):
        pieceMoves = []
        playerColor = board[r,c][0]

        if 'U' in board[r,c]:
            if r-1 >= 0 and '--' in board[r-1][c]:
                self.createMoves(r, c, board, (r-1, c), pieceMoves)

                if r == 6 and '--' in board[r-2][c]:
                    self.createMoves(r, c, board, (r-2, c), pieceMoves)

            if c-1 >= 0 and r-1 >= 0:
                canEnPassant = self.canEnPassant(r, c-1)
                if '--' not in board[r-1][c-1]:
                    if board[r-1][c-1][0] not in playerColor:
                        self.createMoves(r, c, board, (r-1, c-1), pieceMoves)
                elif canEnPassant:
                    move = Move((r, c), (r-1, c-1), board)
                    pieceMoves.append(self.enPassant(r, c-1, board[r][c-1], move))

            if c+1 <= 7 and r-1 >= 0:
                canEnPassant = self.canEnPassant(r, c+1)
                if '--' not in board[r-1][c+1]:
                    if board[r-1][c+1][0] not in playerColor:
                        self.createMoves(r, c, board, (r-1, c+1), pieceMoves)
                elif canEnPassant:
                    move = Move((r, c), (r-1, c+1), board)
                    pieceMoves.append(self.enPassant(r, c+1, board[r][c+1], move))

        else:
            if r+1 <= 7 and '--' in board[r+1][c]:
                self.createMoves(r, c, board, (r+1, c), pieceMoves)

                if r == 1 and '--' in board[r+2][c]:
                    self.createMoves(r, c, board, (r+2, c), pieceMoves)

            if c-1 >= 0 and r+1 <= 7:
                canEnPassant = self.canEnPassant(r, c-1)
                if '--' not in board[r+1][c-1]:
                    if board[r+1][c-1][0] not in playerColor:
                        self.createMoves(r, c, board, (r+1, c-1), pieceMoves)
                elif canEnPassant:
                    move = Move((r, c), (r+1, c-1), board)
                    pieceMoves.append(self.enPassant(r, c-1, board[r][c-1], move))

            if c+1 <= 7 and r+1 <= 7:
                canEnPassant = self.canEnPassant(r, c+1)
                if '--' not in board[r+1][c+1]:
                    if board[r+1][c+1][0] not in playerColor:
                        self.createMoves(r, c, board, (r+1, c+1), pieceMoves)
                elif canEnPassant:
                    move = Move((r, c), (r+1, c+1), board)
                    pieceMoves.append(self.enPassant(r, c+1, board[r][c+1], move))

        return pieceMoves 
    
    def canPawnPromoted(self, r):
        if (r == 0 or r == 7):
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
    
    def rooks_move(self, r, c, board):
        pieceMoves = []
        directions = ((1,0),(-1,0),(0,1),(0,-1))

        for d in directions:
            self.createMoves(r, c, board, d, pieceMoves)

        return pieceMoves
    
    def knight_move(self, r, c, board):
        pieceMoves = []
        directions = ((1,2),(-1,2),(1,-2),(-1,-2),(2,1),(-2,1),(2,-1),(-2,-1))

        for d in directions:
            self.createMoves(r, c, board, d, pieceMoves)

        return pieceMoves
    
    def bishop_moves(self, r, c, board):
        pieceMoves = []
        directions = ((1, 1), (-1, 1), (1, -1), (-1, -1))

        for d in directions:
            self.createMoves(r, c, board, d, pieceMoves)

        return pieceMoves
    
    def queen_moves(self, r, c, board):
        pieceMoves = []
        directions = ((1, 1), (-1, 1), (1, -1), (-1, -1), (1,0), (-1,0), (0,1), (0,-1))

        for d in directions:
            self.createMoves(r, c, board, d, pieceMoves)

        return pieceMoves
    
    def king_moves(self, r, c, board):
        pieceMoves = []
        directions = ((1, 1), (-1, 1), (1, -1), (-1, -1), (1,0), (-1,0), (0,1), (0,-1))

        for d in directions:
            self.createMoves(r, c, board, d, pieceMoves)
            
        self.castlingKing(r, c, board, pieceMoves)

        return pieceMoves
    
    def castlingKing(self, r, c, board, pieceMoves):
        
        def checkMove(j:int, finishCol, kingRow, rookRow):
            for i in range(c+j,finishCol,j):
                square = board[r][i]
                if 'D' in square:
                    return
                if '--' in square:
                    continue
                return
            piece = board[r][finishCol]
            if '--' in piece:
                return
            if piece[0] not in board[r][c]:
                return
            if 'R' not in piece[1]:
                return
            if self.hasRookMove(piece):
                return
            pieceMoves.append(Move((r, c), (r, finishCol), board, (r,kingRow),(r,rookRow)))

        if self.hasKingMove(board[r][c]) or 'D' in board[r][c] or c != 4:
            return
        
        checkMove(1, 7, 6, 5)
        checkMove(-1, 0, 2, 3)
        
    def hasKingMove(self, kingPiece):
        if 'N' in kingPiece:
            piece = kingPiece.rsplit('N', 1)
            if int(piece[1]) >= 1:
                return True
            return False
            
    def hasRookMove(self, rookPiece):
        if 'N' in rookPiece:
            piece = rookPiece.rsplit('N', 1)
            if int(piece[1]) >= 1:
                return True
            return False
    
    def createMoves(self, r, c, board, direction, pieceMoves):
        new_r = direction[0]
        new_c = direction[1]
        playerColor = board[r, c][0]
        playerPiece = board[r, c][1]

        if 'p' in playerPiece:
            if self.canPawnPromoted(new_r):
                pieces = (f'{playerColor}B', f'{playerColor}Q', f'{playerColor}N', f'{playerColor}R')
                for piece in pieces:
                    move = Move((r, c), (new_r, new_c), board)
                    pieceMoves.append(self.pawnPromotion(move, piece))
            else:
                pieceMoves.append(Move((r, c), (new_r, new_c), board))  
        elif 'N' in playerPiece:
            condition = 0 > (r + new_r) or (r + new_r) > 7 or 0 > (c + new_c) or (c + new_c) > 7
            if not condition and ('--' in board[r+new_r][c+new_c] or board[r+new_r][c+new_c][0] not in playerColor):
                pieceMoves.append(Move((r,c), (r+new_r, c+new_c), board))
        elif 'K' in playerPiece:
            condition = (0 <= (r + new_r) <= 7 and 0 <= (c + new_c) <= 7)
            if condition and ('--' in board[r+new_r][c+new_c] or board[r+new_r][c+new_c][0] not in playerColor):
                pieceMoves.append(Move((r, c), (r+new_r, c+new_c), board))
        else:
            condition = False
            while not condition:
                condition = 0 > (r + new_r) or (r + new_r) > 7 or 0 > (c + new_c) or (c + new_c) > 7

                if condition:
                    break

                if '--' not in board[r+new_r][c+new_c] and board[r+new_r][c+new_c][0] in playerColor:
                    break

                pieceMoves.append(Move((r, c), (r+new_r, c+new_c), board))
                if '--' not in board[r+new_r][c+new_c] and board[r+new_r][c+new_c][0] not in playerColor:
                    break

                new_r += direction[0]
                new_c += direction[1]   

    def make_move(self, action, state):
        if 'R' in action.pieceMoved or 'K' in action.pieceMoved:
            if 'N' in action.pieceMoved:
                piece = action.pieceMoved.rsplit('N', 1)
                count = int(piece[1]) + 1
                action.pieceMoved = f'{piece[0]}N{count}'
            else: 
               action.pieceMoved += 'N0'
               
        state[action.startRow][action.startCol] = '--'
        state[action.endRow][action.endCol] = action.pieceMoved
        
        if action.enPassant:
            pass
            #TODO:
            #board[move.passantRow][move.passantCol].piece = None
        if action.pawnPromotion:
            state[action.endRow][action.endCol] = action.piecePromoted
        if action.realKingRow != None and action.realRookRow != None:
            state[action.realKingRow][action.realKingCol] = action.pieceMoved
            state[action.realRookRow][action.realRookCol] = action.pieceCaptured
            state[action.startRow][action.startCol] = '--'
            state[action.endRow][action.endCol] = '--'
        return state

    def undo_moves(self, action, state):
        if 'R' in action.pieceMoved or 'K' in action.pieceMoved:
            if 'N' in action.pieceMoved:
                piece = action.pieceMoved.rsplit('N', 1)
                count = 0 if int(piece[1]) - 1 <= 0 else int(piece[1]) - 1
                action.pieceMoved = f'{piece[0]}N{count}'
                
        state[action.startRow][action.startCol] = action.pieceMoved
        state[action.endRow][action.endCol] = action.pieceCaptured
        
        if action.enPassant:
            state[action.endRow][action.endCol] = '--'
            state[action.passantRow][action.passantCol] = action.pieceCaptured              
        if action.realKingRow != None and action.realRookRow != None:
            state[action.realKingRow][action.realKingCol] = '--'
            state[action.realRookRow][action.realRookCol] = '--' 
        return state

    def check_win(self, state, action):
        valid_moves = self.get_valid_moves(state)
        if len(valid_moves) <= 0:
            return True
        return False
    
    def check_tie(self, state, action):
        pieces = 0
        for r in range(len(state)):
            for c in range(len(state[r])):
                if '--' not in state[r,c]: 
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
                piece = state[i,j]
                if player == -1 and '--' not in piece:
                    state[i,j] = 'w' + piece[1:] if piece[0] == 'b' else 'b' + piece[1:]
        return state

    def get_encoded_state(self, state):
        piecesDict = {'p':2, 'R':3, 'N':4, 'B':5, 'Q':6, 'K':7}
        if len(state.shape) == 3:
            encoded_board = np.zeros((state.shape[0], 3, 8, 8)).astype(np.float32)
        else:
            encoded_board = np.zeros((1, 3, 8, 8)).astype(np.float32)
            state = np.expand_dims(state, axis=0)
            
        for i in range(len(encoded_board)):
            for j in range(len(encoded_board[i])):
                for k in range(len(encoded_board[i][j])):
                    for l in range(len(encoded_board[i][j][k])):
                        piece = state[i,k,l]
                        if '--' not in piece:
                            if j == 0 and piece[0] == 'w':
                                encoded_board[i,j,k,l] = piecesDict[piece[1]]
                            elif j == 2 and piece[0] == 'b':
                                encoded_board[i,j,k,l] = -piecesDict[piece[1]]
                        elif j == 1:
                            encoded_board[i,j,k,l] = 1
                            
        return encoded_board

class Move():
    # maps keys to values
    # key : value
    ranksToRows = {"1":7, "2":6, "3":5, "4":4,
                   "5":3, "6":2, "7":1, "8":0}
    rowsToRanks = {v:k for k, v in ranksToRows.items()}
    filesToCols = {"a":0, "b":1, "c":2, "d":3,
                   "e":4, "f":5, "g":6, "h":7}
    colsToFiles = {v:k for k, v in filesToCols.items()}

    def __init__(self, startSq, endSq, board,  kingSq=None, rookSq=None):
        self.startRow = startSq[0]
        self.startCol = startSq[1]
        self.endRow = endSq[0]
        self.endCol = endSq[1]
        self.pieceMoved = board[self.startRow][self.startCol]
        self.pieceCaptured = board[self.endRow][self.endCol]
        self.enemyKingInRange = self.checkKing()
        # En passant Move
        self.enPassant = False
        self.passantRow = None
        self.passantCol = None
        # Pawn promotion Move
        self.pawnPromotion = False
        self.piecePromoted = None  
        # Castling Move
        self.realKingRow, self.realKingCol, self.realRookRow, self.realRookCol = self.canCastling(kingSq, rookSq)
        self.moveID = '{}{}{}{}'.format(self.startRow,self.startCol,self.endRow,self.endCol)

    def canCastling(self, kingSq, rookSq):
        if kingSq == None or rookSq == None:
            return None, None, None, None
        return kingSq[0], kingSq[1], rookSq[0], rookSq[1]

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
        if 'K' in self.pieceCaptured:
            return True
        return False