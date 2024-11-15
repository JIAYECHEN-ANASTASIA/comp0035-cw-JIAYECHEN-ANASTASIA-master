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

# 定义 IQR 过滤函数
def iqr_filter(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound

# 计算 PEAK_KW 和 KWH 的 IQR 上下界，并输出边界值
peak_lower, peak_upper = iqr_filter(cleaned_data, 'PEAK_KW')
kwh_lower, kwh_upper = iqr_filter(cleaned_data, 'KWH')

print(f"PEAK_KW lower bound: {peak_lower}, upper bound: {peak_upper}")
print(f"KWH lower bound: {kwh_lower}, upper bound: {kwh_upper}")

# 过滤数据
filtered_data = cleaned_data[
    (cleaned_data['PEAK_KW'] >= peak_lower) & (cleaned_data['PEAK_KW'] <= peak_upper) &
    (cleaned_data['KWH'] >= kwh_lower) & (cleaned_data['KWH'] <= kwh_upper)
]

# 检查过滤后的数据量
print(f"Filtered data length: {len(filtered_data)}")

# 检查过滤后的数据是否为空并绘制直方图
if not filtered_data.empty:
    filtered_data['PEAK_KW'].plot(kind='hist', bins=30, alpha=0.7, color='orange', figsize=(8, 5),
                                  title="PEAK_KW Distribution")
    plt.xlabel("PEAK_KW")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
else:
    print("Filtered data is empty after applying IQR. Consider adjusting the filtering criteria.")

# 绘制 KWH 的直方图
if not filtered_data.empty:
    filtered_data['KWH'].plot(kind='hist', bins=30, alpha=0.7, color='blue', figsize=(8, 5), title="KWH Distribution")
    plt.xlabel("KWH")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
else:
    print("Filtered data is empty after applying IQR. Consider adjusting the filtering criteria.")
