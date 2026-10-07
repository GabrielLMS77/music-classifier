from pathlib import Path

data_path = Path ("data/genres_original")

print("Dataset path:", data_path)


print("Does the dataset exist?", data_path.exists())

items = list(data_path.iterdir())

genre_counts = {}

for genre_folder in items:
    
    genre_counts[genre_folder.name] = 0
    


    files = list(genre_folder.iterdir())

    for file in files:


        if file.suffix == ".wav":
           genre_counts[genre_folder.name] += 1

for genre, count in genre_counts.items():
    print(f"{genre}: {count}")






