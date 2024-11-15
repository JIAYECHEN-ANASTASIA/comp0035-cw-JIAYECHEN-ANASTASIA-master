from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.max_columns = None
data_folder = Path(__file__).parent.joinpath('data')
dataframes = []
# merge data
if not data_folder.exists():
    print("Data folder not found.")
else:
    csv_files = list(data_folder.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in the folder.")
    else:

        for file in csv_files:
            df = pd.read_csv(file)
            dataframes.append(df)
merged_data = pd.concat(dataframes, ignore_index=True)

num_rows = len(merged_data)
# show merged_data length
print("The length of merged_data is %d." % num_rows)

# show merged_data info
print(merged_data.info())

missing_data = merged_data.isnull().sum()
print("The missing_data：\n", missing_data)

cleaned_data = merged_data.dropna()

duplicates = cleaned_data.duplicated().sum()
# There is no duplicate.
print(f"The number of duplicates: {duplicates}")
cleaned_data = cleaned_data.drop_duplicates()

# show cleaned_data describe
print(cleaned_data.describe())

area_counts = cleaned_data['ADMINISTRATIVE_AREA'].value_counts()
print("The amount of datas in each ADMINISTRATIVE_AREA：\n", area_counts)
area_counts.plot(kind='bar', color='skyblue', figsize=(10, 6), title="Number of datas per ADMINISTRATIVE_AREA")
plt.xlabel("ADMINISTRATIVE_AREA")
plt.ylabel("Number of datas")
plt.tight_layout()
plt.show()


def iqr_filter(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound


peak_lower, peak_upper = iqr_filter(cleaned_data, 'PEAK_KW')
kwh_lower, kwh_upper = iqr_filter(cleaned_data, 'KWH')

filtered_data = cleaned_data[
    (cleaned_data['PEAK_KW'] >= peak_lower) & (cleaned_data['PEAK_KW'] <= peak_upper) &
    (cleaned_data['KWH'] >= kwh_lower) & (cleaned_data['KWH'] <= kwh_upper)
    ]


filtered_data['PEAK_KW'].plot(kind='hist', bins=30, alpha=0.7, color='orange', figsize=(8, 5),
                              title="PEAK_KW Distribution")
plt.xlabel("PEAK_KW")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

filtered_data['KWH'].plot(kind='hist', bins=30, alpha=0.7, color='blue', figsize=(8, 5), title="KWH Distribution")
plt.xlabel("KWH")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

area_avg_consumption = cleaned_data.groupby('ADMINISTRATIVE_AREA')['KWH'].mean().sort_values(ascending=False)

top_5_areas = area_avg_consumption.head(5)
print("The five regions with the highest average KWH:\n", top_5_areas)


top_5_areas.plot(kind='bar', color='green', figsize=(10, 6), title="Top 5 Areas by Average KWH")
plt.xlabel("Area")
plt.ylabel("Average KWH")
plt.tight_layout()
plt.show()

top_5_data = cleaned_data[cleaned_data['ADMINISTRATIVE_AREA'].isin(top_5_areas.index)]
print(top_5_data.head())

top_5_area_counts = top_5_data['ADMINISTRATIVE_AREA'].value_counts()
top_5_area_counts.plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8), title="Proportion of Records by Area")
plt.ylabel("")
plt.show()

grouped_stats = top_5_data.groupby('ADMINISTRATIVE_AREA').agg({
    'KWH': ['max', 'min', 'sum'],
    'RESI_COUNT': 'sum',
    'NONRESI_COUNT': 'sum'
})


grouped_stats.columns = ['Max Energy (KWH)', 'Min Energy (KWH)', 'Total Energy (KWH)', 'Total Residential Count', 'Total Non-Residential Count']
print(grouped_stats)

grouped_stats[['Max Energy (KWH)']].plot(
    kind='line',
    figsize=(10, 6),
    title="Max Energy Statistics by Area"
)
plt.ylabel("Energy (KWH)")
plt.xlabel("Administrative Area")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

grouped_stats[['Min Energy (KWH)']].plot(
    kind='line',
    figsize=(10, 6),
    title="Min Energy Statistics by Area"
)
plt.ylabel("Energy (KWH)")
plt.xlabel("Administrative Area")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()


ward_stats = cleaned_data.groupby('WARD').agg({
    'KWH': 'mean',
    'PEAK_KW': 'mean'
}).sort_values(by='KWH', ascending=False)

ward_stats.rename(columns={
    'KWH': 'Avg Energy Demand (KWH)',
    'PEAK_KW': 'Avg Peak Demand (KW)'
}, inplace=True)

top_5_ward_stats = ward_stats.head(5)

top_5_ward_stats['Avg Energy Demand (KWH)'].plot(kind='barh', figsize=(12, 8))
plt.title("Average Energy Demand by Ward", fontsize=16)
plt.xlabel("Average Energy Demand (KWH)")
plt.ylabel("Ward")
plt.tight_layout()
plt.show()

top_5_ward_stats['Avg Peak Demand (KW)'].plot(kind='barh', figsize=(12, 8))
plt.title("Average Peak Demand by Ward", fontsize=16)
plt.xlabel("Average Peak Demand (KW)")
plt.ylabel("Ward")
plt.tight_layout()
plt.show()
print(top_5_ward_stats)

top_5_ward_data = top_5_data[top_5_data['WARD'].isin(top_5_ward_stats.index)]

peak_lower, peak_upper = iqr_filter(top_5_ward_data, 'PEAK_KW')

kwh_lower, kwh_upper = iqr_filter(top_5_ward_data, 'KWH')

top_5_ward_filtered_data = top_5_ward_data[
    (top_5_ward_data['PEAK_KW'] >= peak_lower) & (top_5_ward_data['PEAK_KW'] <= peak_upper) &
    (top_5_ward_data['KWH'] >= kwh_lower) & (top_5_ward_data['KWH'] <= kwh_upper)
    ]

top_5_ward_filtered_data.boxplot(column='KWH', by='WARD', grid=False, patch_artist=True,
                     medianprops=dict(color='red', linewidth=2),
                     boxprops=dict(facecolor='lightblue', color='black'),
                     whiskerprops=dict(color='black', linewidth=1),
                     flierprops=dict(markerfacecolor='red', marker='o', markersize=6))
plt.title("Boxplot of Energy Demand (KWH) by Ward", fontsize=16)
plt.suptitle('')
plt.xlabel("Ward")
plt.ylabel("Energy Demand (KWH)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


top_5_ward_filtered_data.boxplot(column='PEAK_KW', by='WARD', grid=False, patch_artist=True,
                                 medianprops=dict(color='red', linewidth=2),
                                 boxprops=dict(facecolor='lightblue', color='black'),
                                 whiskerprops=dict(color='black', linewidth=1),
                                 flierprops=dict(markerfacecolor='red', marker='o', markersize=6))
plt.title("Boxplot of PEAK_KW by Ward", fontsize=16)
plt.suptitle('')
plt.xlabel("Ward")
plt.ylabel("PEAK_KW")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


