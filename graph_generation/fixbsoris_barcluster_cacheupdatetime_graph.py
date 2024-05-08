import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('4a4000_size1920_alldata.tsv', delimiter=',')

# Unique network architectures
network_archs = df['Network Arch'].unique()

# Define a colormap for the samplers, assuming known sampler names
sampler_colors = {
    'default': 'red',
    'dali': 'blue',
    'shade': 'green',
    'graddistbg': 'yellow'
}

hatchlist = ['\\', '\\', '\\', '\\', '/', '/', '/', '/', 'o', 'o', 'o', 'o', 'x', 'x', 'x', 'x']
labellist = ["MobileNet-V2", "ResNet-18", "ResNet-50", "ResNet-101"]
# preprocess


def normalize_by_default(group):
    # If the 'exec time' for `default` sample is not found, then cannot normalize, just return
    # if (len(group[group['Sampler'] == 'default']['exec time'].values)) == 0:
    #     return group
    default_time = group[group['Sampler'] == 'default']['cache update time'].values[0]
    # Calculate normalized speed up 'exec time'
    group['normed'] = group['cache update time']
    return group


df['cache update time'] = pd.to_numeric(df['cache update time'], errors='coerce')
# to normalize data load time by default sampler, and convert to speed up
normalized_filtered_df = df.groupby(
    ['Batch Size', 'Image Size', 'Network Arch']).apply(normalize_by_default)

normalized_filtered_df = normalized_filtered_df.reset_index(drop=True)
normalized_filtered_df.to_csv('normalized.csv', index=False, header=False)
print("dumped:", 'normalized.csv')
df = normalized_filtered_df
df['normed'] = pd.to_numeric(df['normed'], errors='coerce')

# dump each <network, image size> for later
for (net_arch, img_size), sub_group in df.groupby(['Network Arch', 'Image Size']):
    file_name = f'{net_arch}_{img_size}.csv'
    # print(file_name)
    sub_group.to_csv(file_name, index=False, header=False)
# exit(0)

df_chosen = None


def gen_plot_4_fixed_image_size():
    IMG_SZ = 128  # 32, 64, 128, 256, 512
    for image_size, sub_group in df.groupby('Image Size'):
        if image_size == IMG_SZ:
            df_chosen = sub_group
            break
    for batch_size, sub_group in df_chosen.groupby('Batch Size'):
        pivot = sub_group.pivot_table(
            index='Network Arch', columns='Sampler', values='normed', aggfunc='first')
        # sort to create the sequence mobilenet, resnet18, resnet50, resnet101
        pivot = pivot.sort_values(by=["Network Arch"], key=lambda x: x.map({"mobilenet_v2": 0, "resnet18": 1, "resnet50": 2, "resnet101": 3}))
        pivot = pivot.reindex(columns=["default", "dali", "shade", "graddistbg"])
        # Plot
        ax = pivot.plot(kind='bar', figsize=(5, 3))
        bars = ax.patches
        for bar, hatch in zip(bars, hatchlist):
            bar.set_hatch(hatch)
        ax.set_xticks(list(range(len(labellist))))
        ax.set_xticklabels(labellist)

        plt.title(f'Batch Size: {batch_size}, Image Size: {IMG_SZ}')
        plt.xlabel('Network Arch')
        plt.ylabel('Cache Update Time')
        plt.legend(loc='upper right', ncol=2)
        plt.xticks(rotation=0)
        # plt.ylim([0, 1.75])
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(f'figures/cacheupdatetime_bs{batch_size}_is{image_size}.png')
    return


def gen_plot_4_fixed_batch_size():
    BATCH_SZ = 32  # 2, 4, 8, 16, 32, 64
    for batch_size, sub_group in df.groupby('Batch Size'):
        if batch_size == BATCH_SZ:
            df_chosen = sub_group
            break
    for image_size, sub_group in df_chosen.groupby('Image Size'):
        pivot = sub_group.pivot_table(
            index='Network Arch', columns='Sampler', values='normed', aggfunc='first')
        pivot = pivot.sort_values(by=["Network Arch"], key=lambda x: x.map({"mobilenet_v2": 0, "resnet18": 1, "resnet50": 2, "resnet101": 3}))
        pivot = pivot.reindex(columns=["default", "dali", "shade", "graddistbg"])
        # Plot
        ax = pivot.plot(kind='bar', figsize=(5, 3))
        bars = ax.patches
        for bar, hatch in zip(bars, hatchlist):
            bar.set_hatch(hatch)
        ax.set_xticks(list(range(len(labellist))))
        ax.set_xticklabels(labellist)
        # Plot
        plt.title(f'Batch Size: {BATCH_SZ}, Image Size: {image_size}')
        plt.xlabel('Network Arch')
        plt.ylabel('Cache Update Time')
        plt.legend(loc='upper right', ncol=2)
        plt.xticks(rotation=0)
        plt.grid(axis='y')
        # plt.ylim([0, 1.75])
        plt.tight_layout()
        plt.savefig(f'figures/cacheupdatetime_bs{batch_size}_is{image_size}.png')
    return


def gen_stacked_plot():
    for arch in network_archs:
        # print(arch)
        arch_df = df[df['Network Arch'] == arch]
        image_sizes = arch_df['Image Size'].unique()
        image_sizes.sort()

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.set_title(f'Dataload Time for Network Arch: {arch}')

        group_offset = np.arange(len(image_sizes))
        bar_width = 0.15

        # Keep track of which samplers have been added to the legend
        added_to_legend = set()

        for i, size in enumerate(image_sizes):
            size_df = arch_df[arch_df['Image Size'] == size]
            batch_sizes = size_df['Batch Size'].unique()
            batch_sizes.sort()

            for j, batch_size in enumerate(batch_sizes):
                batch_df = size_df[size_df['Batch Size'] == batch_size]
                bottom_offset = 0

                for sampler, color in sampler_colors.items():
                    sampler_df = batch_df[batch_df['Sampler'] == sampler]

                    if not sampler_df.empty:
                        dataload_time = sampler_df['normed'].iloc[0]
                        # print(dataload_time)
                        label = sampler if sampler not in added_to_legend else ""
                        ax.bar(i + j*bar_width, dataload_time, bar_width, bottom=bottom_offset,
                               color=color, label=label)
                        bottom_offset += dataload_time

                        if label:  # This means the sampler was not previously added to the legend
                            added_to_legend.add(sampler)

        ax.set_xticks(group_offset + bar_width / 2 * (len(batch_sizes)-2))
        ax.set_xticklabels(image_sizes)

        ax.set_xlabel('Image Size')
        ax.set_ylabel('Normalized Dataload Time Speed Up')

        # Now, only unique legend labels are added
        plt.legend(ncol=4)

        # plt.show()
        plt.tight_layout()
        plt.savefig(f'figures/{arch}.png')


gen_plot_4_fixed_image_size()
gen_plot_4_fixed_batch_size()
# gen_stacked_plot()
