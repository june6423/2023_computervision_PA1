import numpy as np
from tqdm import tqdm


def aggregate_cost_volume(cost_volume):
    #Semi Global Matching을 수행한다.
    aggregated_costs = np.full((cost_volume.shape[0],cost_volume.shape[1],cost_volume.shape[2],8),np.inf)

    forward_pass = ((-1,-1),(-1,0),(-1,1),(0,-1))
    backward_pass = ((1,-1),(1,0),(1,1),(0,1))
    #8개의 방향을 forward_pass와 backward_pass로 묶어서 두 번의 iteration을 통해 aggregated_costs를 계산한다.
    
    for y in range(cost_volume.shape[0]):
        for x in range(cost_volume.shape[1]):
            for dis in range(cost_volume.shape[2]):
                for idx, (dy, dx) in enumerate(forward_pass):
                    if(y+dy >= cost_volume.shape[0] or x+dx >= cost_volume.shape[1] or y+dy < 0 or x+dx < 0):
                        aggregated_costs[y][x][dis][idx] = cost_volume[y][x][dis]
                    elif(dis == 0):
                        a = min(aggregated_costs[y+dy,x+dx,:,idx])
                        if(a == np.inf):
                            aggregated_costs[y][x][dis][idx] = cost_volume[y][x][dis]
                            continue
                        b = min(aggregated_costs[y+dy,x+dx,dis+2:,idx])
                        aggregated_costs[y][x][dis][idx] = cost_volume[y][x][dis] + min(aggregated_costs[y+dy][x+dx][dis][idx],aggregated_costs[y+dy][x+dx][dis+1][idx]+5,b+150) - a
                    elif(dis+1 == cost_volume.shape[2]):
                        a = min(aggregated_costs[y+dy,x+dx,:,idx])
                        if(a == np.inf):
                            aggregated_costs[y][x][dis][idx] = cost_volume[y][x][dis]
                            continue
                        aggregated_costs[y][x][dis][idx] = cost_volume[y][x][dis] + min(aggregated_costs[y+dy][x+dx][dis][idx],aggregated_costs[y+dy][x+dx][dis-1][idx]+5)- a
                    elif(dis+2 == cost_volume.shape[2]):
                        a = min(aggregated_costs[y+dy,x+dx,:,idx])
                        if(a == np.inf):
                            aggregated_costs[y][x][dis][idx] = cost_volume[y][x][dis]
                            continue
                        aggregated_costs[y][x][dis][idx] = cost_volume[y][x][dis] + min(aggregated_costs[y+dy][x+dx][dis][idx],aggregated_costs[y+dy][x+dx][dis-1][idx]+5,aggregated_costs[y+dy][x+dx][dis+1][idx]+5)- a
                    else:
                        a = min(aggregated_costs[y+dy,x+dx,:,idx])
                        if(a == np.inf):
                            aggregated_costs[y][x][dis][idx] = cost_volume[y][x][dis]
                            continue
                        b = min(aggregated_costs[y+dy,x+dx,dis+2:,idx])
                        aggregated_costs[y][x][dis][idx] = cost_volume[y][x][dis] + min(aggregated_costs[y+dy][x+dx][dis][idx],aggregated_costs[y+dy][x+dx][dis-1][idx]+5,aggregated_costs[y+dy][x+dx][dis+1][idx]+5,b+150) - a

    for y in reversed(range(cost_volume.shape[0])):
        for x in reversed(range(cost_volume.shape[1])):
            for dis in range(cost_volume.shape[2]):
                for idx, (dy, dx) in enumerate(backward_pass):
                    if(y+dy >= cost_volume.shape[0] or x+dx >= cost_volume.shape[1] or y+dy < 0 or x+dx < 0):
                        aggregated_costs[y][x][dis][idx+4] = cost_volume[y][x][dis]
                    elif(dis == 0):
                        a = min(aggregated_costs[y+dy,x+dx,:,idx+4])
                        if(a == np.inf):
                            aggregated_costs[y][x][dis][idx+4] = cost_volume[y][x][dis]
                            continue
                        b = min(aggregated_costs[y+dy,x+dx,dis+2:,idx+4])
                        aggregated_costs[y][x][dis][idx+4] = cost_volume[y][x][dis] + min(aggregated_costs[y+dy][x+dx][dis][idx+4],aggregated_costs[y+dy][x+dx][dis+1][idx+4]+5,b+150)- a
                    elif(dis+1 == cost_volume.shape[2]):
                        a = min(aggregated_costs[y+dy,x+dx,:,idx+4])
                        if(a == np.inf):
                            aggregated_costs[y][x][dis][idx+4] = cost_volume[y][x][dis]
                            continue
                        aggregated_costs[y][x][dis][idx+4] = cost_volume[y][x][dis] + min(aggregated_costs[y+dy][x+dx][dis][idx+4],aggregated_costs[y+dy][x+dx][dis-1][idx+4]+5) - a
                    elif(dis+2 == cost_volume.shape[2]):
                        a = min(aggregated_costs[y+dy,x+dx,:,idx+4])
                        if(a == np.inf):
                            aggregated_costs[y][x][dis][idx+4] = cost_volume[y][x][dis]
                            continue
                        aggregated_costs[y][x][dis][idx+4] = cost_volume[y][x][dis] + min(aggregated_costs[y+dy][x+dx][dis][idx+4],aggregated_costs[y+dy][x+dx][dis-1][idx+4]+5,aggregated_costs[y+dy][x+dx][dis+1][idx+4]+5) - a
                    else:
                        a = min(aggregated_costs[y+dy,x+dx,:,idx+4])
                        if(a == np.inf):
                            aggregated_costs[y][x][dis][idx+4] = cost_volume[y][x][dis]
                            continue
                        b = min(aggregated_costs[y+dy,x+dx,dis+2:,idx+4])
                        aggregated_costs[y][x][dis][idx+4] = cost_volume[y][x][dis] + min(aggregated_costs[y+dy][x+dx][dis][idx+4],aggregated_costs[y+dy][x+dx][dis-1][idx+4]+5,aggregated_costs[y+dy][x+dx][dis+1][idx+4]+5,b+150) - a

    aggregated_cost_sum = aggregated_costs.sum(axis=3)
    #모든 방향에 대해 aggregated_costs를 더한다.
    return aggregated_cost_sum
