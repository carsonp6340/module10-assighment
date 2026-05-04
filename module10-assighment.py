# Module 10 Assignment: Data Manipulation and Cleaning with Pandas
# UrbanStyle Customer Data Cleaning

# Import required libraries
import pandas as pd
import numpy as np
from datetime import datetime

# Welcome message
print("=" * 60)
print("URBANSTYLE CUSTOMER DATA CLEANING")
print("=" * 60)

# ----- USE THE FOLLOWING CODE TO SIMULATE A CSV FILE (DO NOT MODIFY) -----
from io import StringIO

csv_content = """customer_id,first_name,last_name,email,phone,join_date,last_purchase,total_purchases,total_spent,preferred_category,satisfaction_rating,age,city,state,loyalty_status
CS001,John,Smith,johnsmith@email.com,(555) 123-4567,2023-01-15,2023-12-01,12,"1,250.99",Menswear,4.5,35,Tampa,FL,Gold
CS002,Emily,Johnson,emily.j@email.com,555.987.6543,01/25/2023,10/15/2023,8,$875.50,Womenswear,4,28,Miami,FL,Silver
CS003,Michael,Williams,mw@email.com,(555)456-7890,2023-02-10,2023-11-20,15,"2,100.75",Footwear,5,42,Orlando,FL,Gold
CS004,JESSICA,BROWN,jess.brown@email.com,5551234567,2023-03-05,2023-12-10,6,659.25,Womenswear,3.5,31,Tampa,FL,Bronze
CS005,David,jones,djones@email.com,555-789-1234,2023-03-20,2023-09-18,4,350.00,Menswear,,45,Jacksonville,FL,Bronze
CS006,Sarah,Miller,sarah_miller@email.com,(555) 234-5678,2023-04-12,2023-12-05,10,1450.30,Accessories,4,29,Tampa,FL,Silver
CS007,Robert,Davis,robert.davis@email.com,555.444.7777,04/30/2023,11/25/2023,7,$725.80,Footwear,4.5,38,Miami,FL,Silver
CS008,Jennifer,Garcia,jen.garcia@email.com,(555)876-5432,2023-05-15,2023-10-30,3,280.50,ACCESSORIES,3,25,Orlando,FL,Bronze
CS009,Michael,Williams,m.williams@email.com,5558889999,2023-06-01,2023-12-07,9,1100.00,Menswear,4,39,Jacksonville,FL,Silver
CS010,Emily,Johnson,emilyjohnson@email.com,555-321-6547,2023-06-15,2023-12-15,14,"1,875.25",Womenswear,4.5,27,Miami,FL,Gold
CS006,Sarah,Miller,sarah_miller@email.com,(555) 234-5678,2023-04-12,2023-12-05,10,1450.30,Accessories,4,29,Tampa,FL,Silver
CS011,Amanda,,amanda.p@email.com,(555) 741-8529,2023-07-10,,2,180.00,womenswear,3,32,Tampa,FL,Bronze
CS012,Thomas,Wilson,thomas.w@email.com,,2023-07-25,2023-11-02,5,450.75,menswear,4,44,Orlando,FL,Bronze
CS013,Lisa,Anderson,lisa.a@email.com,555.159.7530,08/05/2023,,0,0.00,Womenswear,,30,Miami,FL,
CS014,James,Taylor,jtaylor@email.com,555-951-7530,2023-08-20,2023-10-10,11,"1,520.65",Footwear,4.5,,Jacksonville,FL,Gold
CS015,Karen,Thomas,karen.t@email.com,(555) 357-9512,2023-09-05,2023-12-12,6,685.30,Womenswear,4,36,Tampa,FL,Silver
"""

customer_data_csv = StringIO(csv_content)
# ----- END OF SIMULATION CODE -----


# TODO 1: Load and Explore the Dataset
raw_df = pd.read_csv(customer_data_csv)
initial_missing_counts = raw_df.isna().sum()
initial_duplicate_count = raw_df.duplicated().sum()

# TODO 2: Handle Missing Values
missing_value_report = raw_df.isna().sum()

# 2.2 Fill missing satisfaction_rating with median
satisfaction_median = raw_df["satisfaction_rating"].median()
raw_df["satisfaction_rating"].fillna(satisfaction_median, inplace=True)

# 2.3 Fill missing last_purchase with backward fill
date_fill_strategy = "backward_fill"
raw_df["last_purchase"] = raw_df["last_purchase"].bfill()

# 2.4 Fill other missing values
raw_df["last_name"].fillna("Unknown", inplace=True)
raw_df["phone"].fillna("Unknown", inplace=True)
raw_df["loyalty_status"].fillna("Bronze", inplace=True)
raw_df["age"].fillna(raw_df["age"].median(), inplace=True)

# Clean total_spent
raw_df["total_spent"] = (
    raw_df["total_spent"]
    .astype(str)
    .str.replace(r"[\$,]", "", regex=True)
    .astype(float)
)

df_no_missing = raw_df

print("\nMissing values after cleaning:")
print(df_no_missing.isna().sum())

# TODO 3: Correct Data Types
df_typed = df_no_missing.copy()
df_typed["join_date"] = pd.to_datetime(df_typed["join_date"], errors="coerce")
df_typed["last_purchase"] = pd.to_datetime(df_typed["last_purchase"], errors="coerce")
df_typed["total_purchases"] = pd.to_numeric(df_typed["total_purchases"], errors="coerce")
df_typed["age"] = pd.to_numeric(df_typed["age"], errors="coerce")

# TODO 4: Clean and Standardize Text Data
df_text_cleaned = df_typed.copy()
df_text_cleaned["first_name"] = df_text_cleaned["first_name"].str.title()
df_text_cleaned["last_name"] = df_text_cleaned["last_name"].str.title()
df_text_cleaned["preferred_category"] = df_text_cleaned["preferred_category"].str.title()

phone_format = "(XXX) XXX-XXXX"
df_text_cleaned["phone"] = (
    df_text_cleaned["phone"]
    .astype(str)
    .str.replace("[^0-9]", "", regex=True)
    .str.replace(r"(\d{3})(\d{3})(\d{4})", r"(\1) \2-\3", regex=True)
)

# TODO 5: Remove Duplicates
duplicate_count = df_text_cleaned.duplicated().sum()
df_no_duplicates = df_text_cleaned.drop_duplicates(subset="customer_id", keep="first")

# TODO 6: Add Derived Features
today = pd.Timestamp.today()
df_no_duplicates["days_since_last_purchase"] = (
    (today - df_no_duplicates["last_purchase"]).dt.days
)
df_no_duplicates["average_purchase_value"] = (
    df_no_duplicates["total_spent"] / df_no_duplicates["total_purchases"]
)

def categorize_purchases(x):
    if x >= 10:
        return "High"
    elif x >= 5:
        return "Medium"
    else:
        return "Low"

df_no_duplicates["purchase_frequency_category"] = (
    df_no_duplicates["total_purchases"].apply(categorize_purchases)
)

# TODO 7: Clean Up the DataFrame
df_renamed = df_no_duplicates.rename(
    columns={
        "customer_id": "Customer ID",
        "first_name": "First Name",
        "last_name": "Last Name",
        "email": "Email",
        "phone": "Phone",
        "join_date": "Join Date",
        "last_purchase": "Last Purchase",
        "total_purchases": "Total Purchases",
        "total_spent": "Total Spent",
        "preferred_category": "Preferred Category",
        "satisfaction_rating": "Satisfaction Rating",
        "age": "Age",
        "city": "City",
        "state": "State",
        "loyalty_status": "Loyalty Status",
        "days_since_last_purchase": "Days Since Last Purchase",
        "average_purchase_value": "Average Purchase Value",
        "purchase_frequency_category": "Purchase Frequency Category",
    }
)

df_final = df_renamed.drop(columns=["Email", "Phone"])
df_final = df_final.sort_values(by="Total Spent", ascending=False)

# TODO 8: Generate Insights
avg_spent_by_loyalty = df_final.groupby("Loyalty Status")["Total Spent"].mean()
category_revenue = df_final.groupby("Preferred Category")["Total Spent"].sum().sort_values(ascending=False)
satisfaction_spend_corr = df_final["Satisfaction Rating"].corr(df_final["Total Spent"])

# TODO 9: Generate Final Report
print("\n" + "=" * 60)
print("URBANSTYLE CUSTOMER DATA CLEANING REPORT")
print("=" * 60)

print("Data Quality Issues:")
print(f"- Missing Values: {missing_value_report.sum()} total missing entries")
print(f"- Duplicates: {initial_duplicate_count} duplicate records found")
print("- Data Type Issues: join_date, last_purchase, and total_spent required conversion\n")

print("Standardization Changes:")
print("- Names: Converted to proper case")
print("- Categories: Capitalized consistently (e.g., Menswear, Womenswear, Accessories)")
print(f"- Phone Numbers: Standardized to {phone_format}\n")

print("Key Business Insights:")
print(f"- Customer Base: {df_final.shape[0]} total customers")
print(f"- Revenue by Loyalty:\n{avg_spent_by_loyalty}")
top_category = category_revenue.index[0]
top_revenue = category_revenue.iloc[0]
print(f"- Top Category: {top_category} with ${top_revenue:,.2f} revenue")
print(f"- Correlation (Satisfaction vs Spending): {satisfaction_spend_corr:.2f}\n")

print("First 5 Lines of Cleaned Data:")
print(df_final.head())
