import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

if (len(sys.argv) < 2):
    print("Usage: python3 ./gen_plot.py <directory>")
    exit(1)
directory = sys.argv[1]
# directory = 'imagedim64'

# Specify the column headers
columns = [
    'Network Arch', 'Batch Size', 'Image Size', 'Epoch', 'Sampler', 'dataload time',
    'cache update time', 'data process time', 'exec time', 'rss(MiB)', 'vms(MiB)',
    'max rss(MiB)', 'max vms(MiB)'
]

# Initialize a list to hold all rows of data
data_rows = []
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)

    if os.path.isfile(file_path) and file_path.endswith('.tsv'):
        with open(file_path, 'r') as file:
            line = file.readline().strip()
            data = line.split('\t')
            data_rows.append(data)
            data_rows[-1][2] = data_rows[-1][2].split(",")[2]

df = pd.DataFrame(data_rows, columns=columns)

# save the dataframe in case for later
combined_rows_file = os.path.join(directory, 'combined_rows.csv')
print("dumped:", combined_rows_file)

# sort
df['dataload time'] = pd.to_numeric(df['dataload time'], errors='coerce')
df_sorted = df.sort_values(
    by=['Batch Size', 'Image Size', 'Network Arch'])

kept_groups = []

df_sorted.to_csv(combined_rows_file, index=False, header=False)

# this is to filter out grouped df whose len == 1 --> meaning the one only with `graddistbg` dose not count
for (batch_size, image_size, net_archj), bs_is_group in df_sorted.groupby(['Batch Size', 'Image Size', 'Network Arch']):
    if (len(bs_is_group) != 1):
        kept_groups.append(bs_is_group)

filtered_df = pd.concat(kept_groups)
# print(filtered_df)


def normalize_by_default(group):
    # If the 'dataload time' for `default` sample is not found, then cannot normalize, just return
    if (len(group[group['Sampler'] == 'default']['dataload time'].values)) == 0:
        return group
    default_time = group[group['Sampler'] ==
                         'default']['dataload time'].values[0]
    # Calculate normalized speed up 'dataload time'
    group['Normalized Speed Up'] = 1 / \
        (group['dataload time'] / default_time)
    return group


# to normalize data load time by default sampler, and convert to speed up
normalized_filtered_df = filtered_df.groupby(
    ['Batch Size', 'Image Size', 'Network Arch']).apply(normalize_by_default)

normalized_filtered_df = normalized_filtered_df.reset_index(drop=True)
filtered_file_path = os.path.join(directory, 'filtered.csv')
print("dumped:", filtered_file_path)
normalized_filtered_df.to_csv(filtered_file_path, index=False, header=False)

for (batch_size, image_size), bs_is_group in normalized_filtered_df.groupby(['Batch Size', 'Image Size']):
    # print(bs_is_group)
    pivot = bs_is_group.pivot_table(
        index='Network Arch', columns='Sampler', values='Normalized Speed Up', aggfunc='first')

    # Plot
    pivot.plot(kind='bar', figsize=(16, 9))
    # plt.title(f'Batch Size: {batch_size}, Image Size: {image_size}')
    plt.ylabel('Normalized Speed Up')
    plt.xlabel('Network Arch')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot with a descriptive filename
    filename = f'bs_{batch_size}_imgdim_{image_size}_dataloadtime.png'
    file_path = os.path.join(directory, filename)
    print("dumped:", file_path)
    plt.savefig(file_path)
    plt.close()  # Close the plot to free up memory and avoid display issues in some environments
