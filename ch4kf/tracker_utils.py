import numpy as np
from scipy.optimize import linear_sum_assignment

def cost_matrix(tracks, detections):
    M = len(tracks)
    N = len(detections)
    print("Number of tracks:", M)
    print("Tracks:", tracks)
    print("Number of detections:", N)
    print("Detections:", detections)
    if N == 0:
        # No detections, set all costs to zero
        return np.zeros((M, M))
    cost = np.zeros(shape=(M, N))    
    for i in range(M):
        for j in range(N):                 
            dist = np.array(tracks[i].predicted[:2] - detections[j].reshape((2,1)))            
            cost[i][j] = np.sqrt(dist[0] ** 2 + dist[1] ** 2)
    print("Cost matrix:")
    print(cost)
    # Normalize cost matrix                
    max_cost = np.max(cost)
    print("Maximum cost:", max_cost)
    cost = cost / max_cost
    # print("Normalized cost matrix:")
    print(cost)
    return cost

            
def track_association(tracks, detections, dist_th=0.5):
    if not tracks:        
        return [], np.arange(len(detections)), []

    cost = cost_matrix(tracks, detections)            
    print("Cost matrix in track_association:")
    print(cost)

    # Use the Hungarian algorithm to find the optimal matches 
    row_ind, col_ind = linear_sum_assignment(cost)
    paired_indices =list(zip(row_ind, col_ind))
    print("Paired indices:")
    print(paired_indices)

    # Find unpaired detections and trackers
    unpaired_tracks=[d for d, _ in enumerate(tracks) if d not in row_ind]
    unpaired_detections=[t for t, _ in enumerate(detections) if t not in col_ind]
    print("Unpaired tracks:", unpaired_tracks)
    print("Unpaired detections:", unpaired_detections)

    # Filter out matches with a distance greater than the threshold
    pairs = []
    for i,j in paired_indices:
        if cost[i][j] < dist_th:
            pairs.append((i,j))
        else:            
            unpaired_tracks.append(i)
            unpaired_detections.append(j)
    print("Filtered pairs:")
    print(pairs)
    print("Updated unpaired tracks:", unpaired_tracks)
    print("Updated unpaired detections:", unpaired_detections)
    return pairs, unpaired_detections, unpaired_tracks
