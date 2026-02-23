"""Merge raw Excel tables into a single master dataset.

Place the raw files in `data/` and run this script to create
`master_tourism_dataset.csv` used by the rest of the project.
"""

import pandas as pd

# Load datasets
transaction = pd.read_excel("data/Transaction.xlsx")
user = pd.read_excel("data/User.xlsx")
city = pd.read_excel("data/City.xlsx")
country = pd.read_excel("data/Country.xlsx")
region = pd.read_excel("data/Region.xlsx")
continent = pd.read_excel("data/Continent.xlsx")
item = pd.read_excel("data/Item.xlsx")
type_df = pd.read_excel("data/Type.xlsx")
visitmode = pd.read_excel("data/visitmode.xlsx")

print("All files loaded successfully!")

# Merge transaction + user
df = transaction.merge(user, on="UserId", how="left")

# Merge visit mode (if VisitModeId exists)
if "VisitModeId" in df.columns:
    df = df.merge(visitmode, on="VisitModeId", how="left")

# Merge attraction (item)
df = df.merge(item, on="AttractionId", how="left")

# Merge attraction type
df = df.merge(type_df, on="AttractionTypeId", how="left")

# Merge city
df = df.merge(city, left_on="AttractionCityId", right_on="CityId", how="left")

# Fix duplicate country columns
if "CountryId_y" in df.columns:
    df.rename(columns={"CountryId_y": "CountryId"}, inplace=True)

if "CountryId_x" in df.columns:
    df.drop(columns=["CountryId_x"], inplace=True)

# Merge country
df = df.merge(country, left_on="CountryId", right_on="CountryId", how="left")

# Fix duplicate RegionId columns
if "RegionId_y" in df.columns:
    df.rename(columns={"RegionId_y": "RegionId"}, inplace=True)

if "RegionId_x" in df.columns:
    df.drop(columns=["RegionId_x"], inplace=True)

df = df.merge(region, left_on="RegionId", right_on="RegionId", how="left")
 
if "ContinentId_y" in df.columns:
    df.rename(columns={"ContinentId_y": "ContinentId"}, inplace=True)

if "ContinentId_x" in df.columns:
    df.drop(columns=["ContinentId_x"], inplace=True)
# Merge continent
df = df.merge(continent, on="ContinentId", how="left")

print("Merging completed!")
print("Final shape:", df.shape)

# Save final dataset
df.to_csv("master_tourism_dataset.csv", index=False)

print("Master dataset saved successfully!")