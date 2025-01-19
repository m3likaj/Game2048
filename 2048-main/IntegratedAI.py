
import model
import copy

INF = 2**64


def ddqnHeuristic(board, agent):
    import numpy as np
    state = np.array(board.board).flatten().reshape(1, -1)
    q_values = agent.q_network.model.predict(state, verbose=0)
    return float(np.max(q_values))

def expectiminimax(board, depth, dir=None, agent=None):
    if board.checkLoss():
        return -INF, dir
    elif depth < 0:
        # Use ONLY DDQN heuristic
        return ddqnHeuristic(board, agent), dir

    a = 0
    if depth != int(depth):
        # Player's turn: pick max
        a = -INF
        for move_dir in model.directions:
            simBoard = copy.deepcopy(board)
            score, hadMovement = simBoard.move(move_dir, False)
            if hadMovement:
                res = expectiminimax(simBoard, depth - 0.5, move_dir, agent)[0]
                if res > a:
                    a = res
    else:
        # Nature's turn: calculate average
        a = 0
        openTiles = board.getOpenTiles()
        for addTileLoc in openTiles:
            board.addTile(addTileLoc, 2)
            a += 1.0 / len(openTiles) * expectiminimax(board, depth - 0.5, dir, agent)[0]
            board.addTile(addTileLoc, 0)

    return a, dir

def getNextBestMoveExpectiminimax(board, pool, agent, depth=2):
    bestScore = -INF
    bestNextMove = model.directions[0]
    results = []

    for dir in model.directions:
        simBoard = copy.deepcopy(board)
        score, validMove = simBoard.move(dir, False)
        if not validMove:
            continue
        results.append(pool.apply_async(expectiminimax, (simBoard, depth, dir, agent)))

    results = [res.get() for res in results]
    for res in results:
        if res[0] >= bestScore:
            bestScore = res[0]
            bestNextMove = res[1]

    return bestNextMove
