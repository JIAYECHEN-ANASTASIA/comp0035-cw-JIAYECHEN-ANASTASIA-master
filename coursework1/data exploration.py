from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


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
plt.savefig("number of datas per ADMINISTRATIVE_AREA.png")
plt.close()

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
plt.savefig("PEAK_KW Distribution.png")
plt.close()


filtered_data['KWH'].plot(kind='hist', bins=30, alpha=0.7, color='blue', figsize=(8, 5), title="KWH Distribution")
plt.xlabel("KWH")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("KWH Distribution.png")
plt.close()


area_avg_consumption = cleaned_data.groupby('ADMINISTRATIVE_AREA')['KWH'].mean().sort_values(ascending=False)

top_5_areas = area_avg_consumption.head(5)
print("The five areas with the highest average KWH：\n", top_5_areas)

top_5_areas.plot(kind='bar', color='green', figsize=(10, 6), title="Top 5 Areas by Average KWH")
plt.xlabel("Area")
plt.ylabel("Average KWH")
plt.tight_layout()
plt.savefig("Top 5 Areas by Average KWH.png")
plt.close()

top_5_data = cleaned_data[cleaned_data['ADMINISTRATIVE_AREA'].isin(top_5_areas.index)]
print(top_5_data.head())
top_5_area_counts = top_5_data['ADMINISTRATIVE_AREA'].value_counts()
top_5_area_counts.plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8), title="Proportion of Records by Area")
plt.ylabel("")
plt.savefig("Proportion of Records by Area.png")
plt.close()

top_5_data.to_csv("top_5_datasets.csv", index=False)
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
plt.savefig("Max Energy Statistics by Area.png")
plt.close()


grouped_stats[['Min Energy (KWH)']].plot(
    kind='line',
    figsize=(10, 6),
    title="Min Energy Statistics by Area"
)
plt.ylabel("Energy (KWH)")
plt.xlabel("Administrative Area")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("Min Energy Statistics by Area.png")
plt.close()


#      Q1
ward_stats = top_5_data.groupby('WARD').agg({
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
plt.savefig("Average Energy Demand by Ward.png")
plt.close()


top_5_ward_stats['Avg Peak Demand (KW)'].plot(kind='barh', figsize=(12, 8))
plt.title("Average Peak Demand by Ward", fontsize=16)
plt.xlabel("Average Peak Demand (KW)")
plt.ylabel("Ward")
plt.tight_layout()
plt.savefig("Average Peak Demand by Ward.png")
plt.close()
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
plt.savefig("Boxplot of Energy Demand by Ward.png")
plt.close()



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
plt.savefig("Boxplot of PEAK_KW by Ward.png")
plt.close()


top_5_ward_filtered_data.plot.scatter(
    x='LONGITUDE',
    y='LATITUDE',
    c='KWH',
    cmap='Reds',
    s=80,
    figsize=(12, 8)
)

plt.title("Heatmap of Energy Demand (KWH) by Geographic Location", fontsize=16)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig("Heatmap of Energy Demand by Geographic Location.png")
plt.close()


ward_energy = top_5_data.groupby('WARD').agg(
    Total_KWH=('KWH', 'sum'),
    Total_PEAK_KW=('PEAK_KW', 'sum'),
    Total_Resi_Count=('RESI_COUNT', 'sum'),
    Total_NonResi_Count=('NONRESI_COUNT', 'sum')
).reset_index()

high_energy_threshold = ward_energy['Total_KWH'].quantile(0.8)
high_energy_wards = ward_energy[ward_energy['Total_KWH'] >= high_energy_threshold]

correlation_matrix = high_energy_wards[['Total_KWH', 'Total_PEAK_KW', 'Total_Resi_Count', 'Total_NonResi_Count']].corr()
print("Correlation Matrix:")
print(correlation_matrix)

#      Q2
msoa_data = top_5_data.groupby('MSOA').agg(
    Total_Resi_Count=('RESI_COUNT', 'sum'),
    Total_NonResi_Count=('NONRESI_COUNT', 'sum'),
    Avg_KWH=('KWH', 'mean'),
    Avg_PEAK_KW=('PEAK_KW', 'mean')
).reset_index()
msoa_data['Total_Buildings'] = msoa_data['Total_Resi_Count'] + msoa_data['Total_NonResi_Count']
print(msoa_data.head())

X_kwh = msoa_data[['Total_Resi_Count', 'Total_NonResi_Count', 'Total_Buildings']]
y_kwh = msoa_data['Avg_KWH']


model_kwh = sm.OLS(y_kwh, sm.add_constant(X_kwh)).fit()
print(model_kwh.summary())

X_peak = msoa_data[['Total_Resi_Count', 'Total_NonResi_Count', 'Total_Buildings']]
y_peak = msoa_data['Avg_PEAK_KW']

model_peak = sm.OLS(y_peak, sm.add_constant(X_peak)).fit()
print(model_peak.summary())


residuals_kwh = y_kwh - model_kwh.predict(sm.add_constant(X_kwh))
residuals_kwh.plot(kind='hist', bins=30, figsize=(8, 6), title="Residuals for Avg_KWH Prediction", color='skyblue')
plt.savefig("Residuals for Avg_KWH Prediction.png")
plt.close()


residuals_peak = y_peak - model_peak.predict(sm.add_constant(X_peak))
residuals_peak.plot(kind='hist', bins=30, figsize=(8, 6), title="Residuals for Avg_PEAK_KW Prediction", color='orange')
plt.savefig("Residuals for Avg_PEAK_KW Prediction.png")
plt.close()


#  Q3
peak_lower, peak_upper = iqr_filter(top_5_data, 'PEAK_KW')

kwh_lower, kwh_upper = iqr_filter(top_5_data, 'KWH')
epcs_lower, epcs_upper = iqr_filter(top_5_data, 'EPCS')

top_5_filtered_data = top_5_data[
    (top_5_data['PEAK_KW'] >= peak_lower) & (top_5_data['PEAK_KW'] <= peak_upper) &
    (top_5_data['KWH'] >= kwh_lower) & (top_5_data['KWH'] <= kwh_upper)&
    (top_5_data['EPCS'] >= epcs_lower) & (top_5_data['EPCS'] <= epcs_upper)
    ]



low_efficiency_data = top_5_filtered_data[top_5_filtered_data['EPCS'] < 2]


low_efficiency_by_area = low_efficiency_data.groupby('WARD')['EPCS'].count()
top_low_efficiency = low_efficiency_by_area.sort_values(ascending=False).head(10)
print(top_low_efficiency)

top_low_efficiency.plot(kind='bar', figsize=(12, 18), color='orange', title="Low Efficiency Buildings by Area (EPCS < 5)")
plt.ylabel("Number of Low Efficiency Buildings")
plt.xticks(rotation=45)
plt.savefig("Low Efficiency Buildings by Area.png")
plt.close()


X = sm.add_constant(top_5_filtered_data['EPCS'])
y = top_5_filtered_data['KWH']


model = sm.OLS(y, X).fit()


print(model.summary())


y_pred = model.predict(X)

plt.figure(figsize=(8, 6))
plt.scatter(top_5_filtered_data['EPCS'], top_5_filtered_data['KWH'], alpha=0.6, color='blue', label='Data Points')
plt.plot(top_5_filtered_data['EPCS'], y_pred, color='red', label='Regression Line')
plt.xlabel('Energy Efficiency Rating (EPCS)')
plt.ylabel('Energy Consumption (KWH)')
plt.title("EPCS vs Energy Consumption with Regression Line")
plt.legend()
plt.savefig("EPCS vs Energy Consumption with Regression Line.png")
plt.close()


