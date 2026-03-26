import pickle

with open("outputs/nominal/cagrad_trajectories.pkl", "rb") as f:
    obj = pickle.load(f)

# If it's a list or tuple
# find the last non-[nan, nan] entry in the 'traj' key of the first element
for i in range(len(obj[0]['traj'])-1, -1, -1):
    if not (obj[0]['traj'][i][0] != obj[0]['traj'][i][0] and obj[0]['traj'][i][1] != obj[0]['traj'][i][1]):
        print(f"Last non-nan entry index: {i}, value: {obj[0]['traj'][i]}")
        break
